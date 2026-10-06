"""DDP setup and hardware-adaptive precision.

torchrun sets WORLD_SIZE/RANK/LOCAL_RANK; without them we run single-process. Precision:
Ampere+ (sm >= 80) -> bf16 autocast; Turing (T4, sm 75) -> fp16 autocast + GradScaler
(T4 has no bf16); CPU -> fp32. Master weights are always fp32.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist


@dataclass
class DistInfo:
    rank: int = 0
    local_rank: int = 0
    world: int = 1
    device: torch.device = torch.device("cpu")

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def enabled(self) -> bool:
        return self.world > 1


def setup() -> DistInfo:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    else:
        device = torch.device("cpu")
    if world > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    return DistInfo(rank, local, world, device)


def teardown(info: DistInfo):
    if info.enabled and dist.is_initialized():
        dist.destroy_process_group()


def barrier(info: DistInfo):
    if info.enabled:
        dist.barrier()


def all_mean(info: DistInfo, t: torch.Tensor) -> torch.Tensor:
    if info.enabled:
        t = t.clone()
        dist.all_reduce(t)
        t /= info.world
    return t


def any_flag(info: DistInfo, flag: bool) -> bool:
    """True if the flag is set on any rank (all ranks get the same answer)."""
    if not info.enabled:
        return flag
    t = torch.tensor([1.0 if flag else 0.0], device=info.device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return bool(t.item() > 0)


def pick_precision(device: torch.device, requested: str = "auto") -> Optional[torch.dtype]:
    """Autocast dtype, or None for fp32."""
    if requested == "fp32" or device.type != "cuda":
        return None
    if requested == "bf16":
        return torch.bfloat16
    if requested == "fp16":
        return torch.float16
    major, _ = torch.cuda.get_device_capability(device)
    return torch.bfloat16 if major >= 8 else torch.float16


PEAK_FLOPS = {"T4": 65e12, "L4": 121e12, "A100": 312e12, "H100": 989e12, "A10": 125e12, "V100": 125e12}


def peak_flops(device: torch.device) -> Optional[float]:
    if device.type != "cuda":
        return None
    name = torch.cuda.get_device_name(device)
    for k, v in PEAK_FLOPS.items():
        if k in name:
            return v
    return None
