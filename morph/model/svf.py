"""Transformer² / Singular Value Fine-tuning (Sun et al. 2025, arXiv 2501.06252) — Stage 3.

Every adapted weight is decomposed once, W = U diag(s) Vᵀ, and frozen. An *expert* is one
vector z per matrix that rescales the singular values: W' = U diag(s ⊙ z) Vᵀ (z = 1 is the
base model). Experts are tiny (one number per singular value) and compose by linear
interpolation, which is what the two-pass inference uses: pass 1 reads the prompt and picks
mixing weights, pass 2 runs with z = Σ_k α_k z_k.
"""
from __future__ import annotations

import json
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

Expert = Dict[str, torch.Tensor]          # module name -> z vector


class SVFLinear(nn.Module):
    """A frozen linear layer re-expressed by its SVD; only the singular-value scales z train."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        W = linear.weight.detach().float()
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        self.register_buffer("U", U.contiguous())
        self.register_buffer("S", S.contiguous())
        self.register_buffer("Vh", Vh.contiguous())
        self.bias = None if linear.bias is None else nn.Parameter(linear.bias.detach().clone(), requires_grad=False)
        self.z = nn.Parameter(torch.ones_like(S))
        self.in_features, self.out_features = linear.in_features, linear.out_features

    def weight(self) -> torch.Tensor:
        with torch.autocast(device_type=self.U.device.type, enabled=False):
            return (self.U * (self.S * self.z.float())) @ self.Vh

    def forward(self, x):
        return F.linear(x, self.weight().to(x.dtype), None if self.bias is None else self.bias.to(x.dtype))


def _targets(model) -> List[str]:
    """Names of the block linears that SVF adapts (mixers + MLPs; not embeddings, LM head, aux heads)."""
    out = []
    for name, m in model.model.layers.named_modules():
        if isinstance(m, nn.Linear) and ".router." not in f".{name}.":
            out.append(f"model.layers.{name}")
    return out


def apply_svf(model, names: Optional[Iterable[str]] = None) -> List[str]:
    """Replace the selected nn.Linear modules with SVFLinear in place, freeze everything but z."""
    names = list(names) if names is not None else _targets(model)
    for full in names:
        parent_name, _, child = full.rpartition(".")
        parent = model.get_submodule(parent_name)
        setattr(parent, child, SVFLinear(getattr(parent, child)))
    for n, p in model.named_parameters():
        p.requires_grad = n.endswith(".z")
    return names


def svf_modules(model) -> Dict[str, SVFLinear]:
    return {n: m for n, m in model.named_modules() if isinstance(m, SVFLinear)}


def get_expert(model) -> Expert:
    return {n: m.z.detach().float().cpu().clone() for n, m in svf_modules(model).items()}


@torch.no_grad()
def set_expert(model, expert: Optional[Expert]) -> None:
    """Load an expert (None = base model, z = 1)."""
    for n, m in svf_modules(model).items():
        m.z.copy_(torch.ones_like(m.z) if expert is None else expert[n].to(m.z.device, m.z.dtype))


def identity_expert(model) -> Expert:
    return {n: torch.ones_like(m.z).cpu() for n, m in svf_modules(model).items()}


def mix_experts(experts: Sequence[Expert], weights: Sequence[float]) -> Expert:
    """Linear interpolation z = Σ w_k z_k (weights are used as given; pass a softmax for a convex mix)."""
    if len(experts) != len(weights):
        raise ValueError("one weight per expert")
    return {n: sum(float(w) * e[n] for e, w in zip(experts, weights)) for n in experts[0]}


def save_expert(path: str, expert: Expert, meta: Optional[dict] = None) -> None:
    from safetensors.torch import save_file
    save_file({k: v.contiguous() for k, v in expert.items()}, path,
              metadata={"meta": json.dumps(meta or {})})


def load_expert(path: str) -> Expert:
    from safetensors.torch import load_file
    return load_file(path)


def n_expert_params(model) -> int:
    return sum(m.z.numel() for m in svf_modules(model).values())
