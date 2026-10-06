"""Time fwd+bwd of each model component in isolation (GPU), to find the bottleneck.

    PYTHONPATH=. python -m morph.tools.bench_components --config configs/model/xs.json --micro 8
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from morph.model import MorphConfig
from morph.model.losses import chunked_cross_entropy
from morph.model.mamba2_block import Mamba2Block
from morph.model.mla_attention import MLAAttention
from morph.model.mlp import SwiGLU
from morph.model.ssd import ssd_chunked
from morph.train.config import apply_overrides


def bench(name, fn, params, iters=5):
    for _ in range(2):
        fn().backward()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(iters):
        fn().backward()
    torch.cuda.synchronize()
    ms = (time.time() - t) / iters * 1e3
    print(f"{name:34s} {ms:8.1f} ms  peak {torch.cuda.max_memory_allocated() / 1e9:5.2f} GB", flush=True)
    torch.cuda.reset_peak_memory_stats()
    return ms


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/model/xs.json")
    ap.add_argument("--micro", type=int, default=8)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--mset", action="append", default=[])
    args = ap.parse_args(argv)
    cfg = apply_overrides(MorphConfig.from_json(args.config), args.mset)
    dev, B, L, d = torch.device("cuda"), args.micro, args.seq, cfg.d_model
    amp = torch.float16 if torch.cuda.get_device_capability()[0] < 8 else torch.bfloat16
    x = torch.randn(B, L, d, device=dev, requires_grad=True)
    ac = lambda: torch.autocast("cuda", dtype=amp)
    print(f"== {args.config} {args.mset} B={B} L={L} amp={amp}", flush=True)

    mamba = Mamba2Block(cfg).to(dev)
    mlp = SwiGLU(cfg).to(dev)
    attn = MLAAttention(cfg, is_producer=True).to(dev)

    def f_mamba():
        with ac():
            return mamba(x).float().square().mean()

    def f_mlp():
        with ac():
            return mlp(x).float().square().mean()

    def f_attn():
        with ac():
            return attn(x)[0].float().square().mean()

    H, P, N = cfg.ssm_nheads, cfg.ssm_headdim, cfg.ssm_d_state
    xs = torch.randn(B, L, H, P, device=dev, dtype=amp, requires_grad=True)
    dt = F.softplus(torch.randn(B, L, H, device=dev) - 2).requires_grad_()
    A = -torch.rand(H, device=dev) - 0.5
    Bm = torch.randn(B, L, N, device=dev, dtype=amp, requires_grad=True)
    Cm = torch.randn(B, L, N, device=dev, dtype=amp, requires_grad=True)

    def f_ssd():
        return ssd_chunked(xs, dt, A, Bm, Cm, cfg.ssm_chunk)[0].float().square().mean()

    W = torch.randn(cfg.vocab_size, d, device=dev, requires_grad=True)
    y = torch.randint(0, cfg.vocab_size, (B, L), device=dev)

    def f_ce():
        with ac():
            return chunked_cross_entropy(x, W, y, cfg.ce_chunk)[0]

    def f_ce_plain():
        with ac():
            return F.cross_entropy((x.half() @ W.half().T).float().view(-1, cfg.vocab_size), y.view(-1))

    norm = torch.nn.RMSNorm(d).to(dev)

    def f_norm():
        with ac():
            return norm(x).float().square().mean()

    def f_matmul():
        with ac():
            return (x @ mamba.mixer.in_proj.weight.T).float().square().mean()

    n_m = cfg.n_layers - len(cfg.attn_layers)
    res = {
        "mamba block": bench("mamba block (fwd+bwd)", f_mamba, None),
        "ssd": bench("  ssd_chunked alone", f_ssd, None),
        "in_proj matmul": bench("  in_proj matmul alone", f_matmul, None),
        "rmsnorm": bench("rmsnorm fp32 input", f_norm, None),
        "mlp": bench("swiglu mlp", f_mlp, None),
        "attn": bench("mla attention", f_attn, None),
        "ce_chunked": bench("chunked CE", f_ce, None),
    }
    try:
        res["ce_plain"] = bench("plain CE (full logits)", f_ce_plain, None)
    except torch.OutOfMemoryError:
        print("plain CE: OOM", flush=True)
    est = n_m * res["mamba block"] + cfg.n_layers * res["mlp"] + len(cfg.attn_layers) * res["attn"] + res["ce_chunked"]
    print(f"estimated model step ≈ {est:.0f} ms  ({B * L / est:.1f}k tok/s); "
          f"mamba share {n_m * res['mamba block'] / est:.0%}, mlp {cfg.n_layers * res['mlp'] / est:.0%}, "
          f"attn {len(cfg.attn_layers) * res['attn'] / est:.0%}, ce {res['ce_chunked'] / est:.0%}", flush=True)


if __name__ == "__main__":
    main()
