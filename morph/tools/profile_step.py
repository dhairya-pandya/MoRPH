"""Profile fwd+bwd of one micro-batch and print the top CUDA ops (where training time goes).

    PYTHONPATH=. python -m morph.tools.profile_step --config configs/model/xs.json --micro 8
"""
from __future__ import annotations

import argparse
import time

import torch
from torch.profiler import ProfilerActivity, profile

from morph.model import MorphConfig, MorphForCausalLM
from morph.train.distributed import pick_precision


def run(model, x, amp):
    with torch.autocast("cuda", dtype=amp):
        out = model(x, x)
    out["loss"].backward()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/model/xs.json")
    ap.add_argument("--micro", type=int, default=8)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--ckpt", action="store_true")
    ap.add_argument("--mset", action="append", default=[])
    args = ap.parse_args(argv)
    from morph.train.config import apply_overrides
    cfg = apply_overrides(MorphConfig.from_json(args.config), args.mset)
    dev = torch.device("cuda")
    model = MorphForCausalLM(cfg).to(dev)
    model.set_grad_checkpointing(args.ckpt)
    amp = pick_precision(dev)
    x = torch.randint(0, cfg.vocab_size, (args.micro, args.seq), device=dev)
    for _ in range(3):
        run(model, x, amp)
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(3):
        run(model, x, amp)
    torch.cuda.synchronize()
    dt = (time.time() - t) / 3
    print(f"[profile] {args.config} {args.mset} micro={args.micro}: {dt * 1e3:.0f} ms/step, "
          f"{args.micro * args.seq / dt / 1e3:.1f}k tok/s", flush=True)
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        run(model, x, amp)
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=35, max_name_column_width=60), flush=True)


if __name__ == "__main__":
    main()
