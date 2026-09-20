"""KV-cache benchmark: the headline frontier table.

Computes, per context length, the bytes an autoregressive decode must cache for
MORPH vs an equal-config full-attention (MHA) control, and the reduction factor.

MORPH caches, per token: only the CLA-shared latent (kv_lora_rank) + one decoupled
RoPE key (qk_rope_head_dim), and ONLY for attention layers (SSM layers cache a
constant-size state independent of sequence length -> ~0 growth). An MHA baseline
caches n_heads * head_dim * 2 (K and V) for EVERY layer.

Run: PYTHONPATH=. python -m morph.eval.kv_bench --config configs/300m_hybrid.json
"""
from __future__ import annotations

import argparse

from morph.model import MorphConfig
from morph.model.cla import assign_cla_groups


def kv_bytes_per_token(cfg: MorphConfig, dtype_bytes: int = 2):
    block_types = cfg.block_types()
    n_attn = sum(1 for t in block_types if t == "a")
    n_groups = len(set(assign_cla_groups(block_types, cfg.cla_group_size).values())) or 0

    head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
    # --- MHA control: cache K and V for every layer ---
    mha = cfg.n_layers * cfg.n_heads * head_dim * 2 * dtype_bytes
    # --- MORPH: latent + rope key, cached once per CLA group (not per attn layer) ---
    morph = n_groups * (cfg.kv_lora_rank + cfg.qk_rope_head_dim) * dtype_bytes
    return mha, morph, n_attn, n_groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/300m_hybrid.json")
    ap.add_argument("--ctx", type=int, nargs="+", default=[4096, 16384, 32768])
    ap.add_argument("--batch", type=int, default=1)
    args = ap.parse_args()

    cfg = MorphConfig.from_json(args.config)
    mha_pt, morph_pt, n_attn, n_groups = kv_bytes_per_token(cfg)

    print(f"config: d_model={cfg.d_model} layers={cfg.n_layers} "
          f"attn_layers={n_attn} cla_groups={n_groups} kv_lora_rank={cfg.kv_lora_rank}")
    print(f"per-token KV: MHA={mha_pt} B  MORPH={morph_pt} B  "
          f"reduction={mha_pt / max(1, morph_pt):.1f}x\n")
    print(f"{'ctx':>8} {'MHA (MB)':>12} {'MORPH (MB)':>12} {'reduction':>10}")
    for L in args.ctx:
        mha = mha_pt * L * args.batch / 1e6
        mo = morph_pt * L * args.batch / 1e6
        print(f"{L:>8} {mha:>12.1f} {mo:>12.1f} {mha / max(1e-9, mo):>9.1f}x")


if __name__ == "__main__":
    main()
