# MORPH v2 — Layerwise Architecture

Shapes for MORPH-S (`configs/model/s.json`); the proxy XS (`configs/model/xs.json`) is the
same design at d=384, 12 layers, attention at [3, 8], latent 64. Batch `B`, length `L`.

```
  vocab 49,152 (SmolLM2)   d_model 640   24 layers   169.7M params (138.3M non-embedding)
  attention layers [4, 9, 14, 19]   CLA groups [[4, 9], [14, 19]]   tied embeddings
```

## Stack

```
 input_ids (B, L) ─▶ Embedding 49152×640 ─▶ x (B, L, 640), fp32 residual stream
   L0–L3   Mamba2 + MLP
   L4      MLA producer  (latent A)       ← self-model target: state entering L4
   L5–L8   Mamba2 + MLP
   L9      MLA consumer  (uses latent A)  ← target
   L10–L13 Mamba2 + MLP
   L14     MLA producer  (latent B)       ← target
   L15–L18 Mamba2 + MLP
   L19     MLA consumer  (uses latent B)  ← target
   L20–L23 Mamba2 + MLP
 ─▶ RMSNorm ─▶ h_f ─┬▶ LM head (tied) ─▶ logits (chunked CE + z-loss)
                    └▶ self-model head 640 → 4×640 (training only)
```

## Mamba2 layer (20 of 24) — parameters match `mamba_ssm.Mamba2`

```
 x ─▶ RMSNorm ─▶ in_proj 640 → [z 1280 | x 1280 | B 128 | C 128 | dt 20]
      xBC ─▶ depthwise causal conv1d (k=4) ─▶ SiLU
      dt  ─▶ softplus(dt + dt_bias);  A = −exp(A_log)   (20 heads × 64)
      y = SSD(x, dt, A, B, C) + D·x        state per layer: 20 × 64 × 128 (constant)
      y = RMSNorm(y · SiLU(z)) ─▶ out_proj 1280 → 640 ─▶ + residual
```
SSD runs as the chunked algorithm (chunk 64) in plain PyTorch: dense matmuls inside a
chunk, a short recurrence across chunks, fp32 decays/states.

## MLA layer (4 of 24), NoPE

```
 x ─▶ RMSNorm ─▶ h
 producer only:  c = RMSNorm(W_down h)       640 → 128     ← the only cached tensor
 all:            q = W_q h                   640 → 10 heads × 64
                 [k | v] = W_up c            128 → 10 × (64 + 64)   (per-layer W_up)
                 o = causal SDPA(q, k, v) ─▶ o_proj 640 → 640 ─▶ + residual
```
No positional encoding: Mamba2 layers carry order/recency. `k` is linear in `c`, so at
inference `W_up` can be absorbed into the query and only `c` is stored.

## MLP (every layer)

`x → RMSNorm → SiLU(W_gate x) ⊙ W_up x (640→1728) → W_down (1728→640) → + residual`

## Objective

```
 loss = CE(next token) + 1e-4 · logsumexp² + λ(t) · mean_k mean((W_k h_f + b_k − rmsnorm(x_k))²)
 x_k = residual state entering attention layer k;  no stop-gradient;  λ ramps 0→λ over warmup
```

## Memory at inference (fp16)

| | per token | fixed per sequence | 32k context |
|---|---|---|---|
| MHA, same depth/heads | 61,440 B | — | 2,013 MB |
| MORPH-S | 512 B (2 latents × 128) | 6.74 MB SSM state + conv tail | 23.5 MB |

`PYTHONPATH=. python -m morph.eval.kv_bench --config configs/model/s.json` reproduces the table.
