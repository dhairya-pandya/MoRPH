"""Local learning coefficient (LLC / RLCT estimate) via SGLD — the paper's complexity measure.

    llc = n·β · ( E_{w ~ SGLD around w*}[ L(w) ] − L(w*) )

SGLD step:  w ← w − (ε/2)·( n·β·∇L_batch(w) + γ·(w − w*) ) + N(0, ε).
Only the language-modeling CE is used (the landscape of the task, as in the paper). Results are
comparable across runs only with identical (ε, γ, n·β, steps, batches); use `--calibrate` to find
an ε range where the estimate is flat (Lau et al. 2023; devinterp practice).

    PYTHONPATH=. python -m morph.eval.llc --ckpt runs/proxy_R1/checkpoints/ckpt_*.pt --data_dir DATA
"""
from __future__ import annotations

import argparse
import json
import math
from contextlib import nullcontext
from typing import List

import numpy as np
import torch

from morph.data.shards import ShardSet


def _ce(model, mb, amp_dtype):
    ctx = torch.autocast(mb.device.type, dtype=amp_dtype) if amp_dtype else nullcontext()
    with ctx:
        return model(mb[:, :-1], mb[:, 1:])["ce"].float()


def _ce_grad(model, mb):
    from morph.model.losses import chunked_cross_entropy
    hidden, _ = model.model(mb[:, :-1])
    ce, _ = chunked_cross_entropy(hidden, model.lm_head.weight, mb[:, 1:], model.cfg.ce_chunk)
    return ce


def estimate_llc(model, batches: List[torch.Tensor], eps: float = 1e-4, gamma: float = 100.0,
                 nbeta: float | None = None, steps: int = 200, burn_in: int = 50, chains: int = 2,
                 amp_dtype=None, seed: int = 0) -> dict:
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    w_star = [p.detach().clone() for p in params]
    n = batches[0][:, 1:].numel()
    nbeta = nbeta if nbeta is not None else n / math.log(n)
    with torch.no_grad():
        l0 = float(np.mean([float(_ce(model, b, amp_dtype)) for b in batches]))
    gen = torch.Generator(device=w_star[0].device).manual_seed(seed)
    chain_means = []
    for c in range(chains):
        with torch.no_grad():
            for p, w in zip(params, w_star):
                p.copy_(w)
        losses = []
        for t in range(steps):
            b = batches[(t + c) % len(batches)]
            model.zero_grad(set_to_none=True)
            ctx = torch.autocast(b.device.type, dtype=amp_dtype) if amp_dtype else nullcontext()
            with ctx:
                loss = _ce_grad(model, b)
            loss.backward()
            with torch.no_grad():
                for p, w in zip(params, w_star):
                    g = p.grad if p.grad is not None else torch.zeros_like(p)
                    p.add_(-(eps / 2) * (nbeta * g.float() + gamma * (p - w)))
                    p.add_(torch.randn(p.shape, device=p.device, generator=gen) * math.sqrt(eps))
            if not math.isfinite(loss.item()):
                break
            if t >= burn_in:
                losses.append(loss.item())
        chain_means.append(float(np.mean(losses)) if losses else float("nan"))
    with torch.no_grad():
        for p, w in zip(params, w_star):
            p.copy_(w)
    llcs = [nbeta * (m - l0) for m in chain_means]
    return {"llc": float(np.mean(llcs)), "llc_std": float(np.std(llcs)), "l0": l0, "nbeta": nbeta,
            "eps": eps, "gamma": gamma, "steps": steps, "chains": chains}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n_batches", type=int, default=16)
    ap.add_argument("--eps", type=float, default=1e-4)
    ap.add_argument("--gamma", type=float, default=100.0)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--chains", type=int, default=2)
    ap.add_argument("--calibrate", action="store_true", help="sweep eps over 3 decades")
    args = ap.parse_args(argv)
    from morph.eval.quality_bench import load_model
    from morph.train.distributed import pick_precision
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(argparse.Namespace(ckpt=args.ckpt, weights=None, config=None), device)
    ds = ShardSet(args.data_dir, "train", args.seq_len)
    rng = np.random.default_rng(0)
    ids = rng.choice(ds.n_windows, size=args.batch * args.n_batches, replace=False)
    batches = [torch.from_numpy(np.stack([ds.window(int(w)) for w in ids[i:i + args.batch]])).to(device)
               for i in range(0, len(ids), args.batch)]
    amp = pick_precision(device)
    amp = amp if amp == torch.bfloat16 else None   # no loss scaling in SGLD: fp16 grads would underflow
    for eps in ([args.eps / 10, args.eps / 3, args.eps, args.eps * 3] if args.calibrate else [args.eps]):
        print(json.dumps(estimate_llc(model, batches, eps=eps, gamma=args.gamma, steps=args.steps,
                                      burn_in=args.steps // 4, chains=args.chains, amp_dtype=amp)), flush=True)


if __name__ == "__main__":
    main()
