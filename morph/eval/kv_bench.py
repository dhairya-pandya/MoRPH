"""Inference-memory accounting: MORPH cache vs an equal-depth multi-head-attention model.

MORPH caches, per token, one MLA latent (kv_lora_rank values) per CLA group — NoPE, so no
separate RoPE key. SSM layers carry a constant state (heads x headdim x d_state) plus the
conv tail, independent of context length; it is reported, not hidden. The MHA control
caches K and V (n_heads x head_dim each) in every layer.

    PYTHONPATH=. python -m morph.eval.kv_bench --config configs/model/s.json
"""
from __future__ import annotations

import argparse

from morph.model import MorphConfig
from morph.model.cla import kv_bytes_per_token


def ssm_state_bytes(cfg: MorphConfig, dtype_bytes: int = 2) -> int:
    n_ssm = cfg.n_layers - len(cfg.attn_layers)
    state = cfg.ssm_nheads * cfg.ssm_headdim * cfg.ssm_d_state
    conv = (cfg.ssm_d_conv - 1) * (cfg.ssm_d_inner + 2 * cfg.ssm_d_state)
    return n_ssm * (state + conv) * dtype_bytes


def mha_bytes_per_token(cfg: MorphConfig, dtype_bytes: int = 2) -> int:
    return cfg.n_layers * 2 * cfg.n_heads * cfg.head_dim * dtype_bytes


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/model/s.json")
    ap.add_argument("--ctx", type=int, nargs="+", default=[4096, 32768, 131072])
    args = ap.parse_args(argv)
    cfg = MorphConfig.from_json(args.config)
    kv, mha, const = kv_bytes_per_token(cfg), mha_bytes_per_token(cfg), ssm_state_bytes(cfg)
    print(f"{args.config}: layers={cfg.n_layers} attn={cfg.attn_layers} cla_groups={cfg.groups()} rank={cfg.kv_lora_rank}")
    print(f"per-token cache: MORPH {kv} B vs MHA {mha} B ({mha / kv:.0f}x); constant SSM state {const / 1e6:.2f} MB/seq")
    print(f"{'ctx':>8} {'MHA MB':>10} {'MORPH MB':>10} {'ratio':>7}")
    for L in args.ctx:
        m, o = mha * L / 1e6, (kv * L + const) / 1e6
        print(f"{L:>8} {m:>10.1f} {o:>10.1f} {m / o:>6.0f}x")


if __name__ == "__main__":
    main()
