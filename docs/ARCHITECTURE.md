# MORPH-300M — Layerwise Architecture

Exact wiring of the 300M config (`configs/300m_hybrid.json`). All shapes are real,
pulled from the config. Batch `B`, sequence `L`, `d_model = 768`.

```
  vocab=32000                                             params: 324.2M total
  d_model=768   n_layers=28   n_heads=12                          299.6M non-embed
  pattern "mmmmmma" ×4  →  24 SSM + 4 MLA-attention               tie_embeddings=on
```

## Full layer stack (top = input)

```
             input_ids  (B, L)
                 │
        ┌────────▼────────┐
        │ Embedding 32000×768   (tied to LM head)
        └────────┬────────┘
                 │  x: (B, L, 768)
   ┌─────────────▼──────────────────────────────────────────────┐
   │ L0   Mamba2 SSM ─┐                                           │
   │ L1   Mamba2 SSM  │                                           │
   │ L2   Mamba2 SSM  │  6 SSM layers                             │
   │ L3   Mamba2 SSM  │  (no KV cache)                            │
   │ L4   Mamba2 SSM  │                                           │
   │ L5   Mamba2 SSM ─┘                                           │
   │ L6   MLA attention ●────────────┐ CLA group 0               │
   │ L7   Mamba2 SSM                 │                            │
   │ ...  (L8–L12 Mamba2 SSM)        │                            │
   │ L13  MLA attention ●────────────┘ shares latent w/ L6        │
   │ L14  Mamba2 SSM                                              │
   │ ...  (L15–L19 Mamba2 SSM)                                    │
   │ L20  MLA attention ●────────────┐ CLA group 1               │
   │ L21  Mamba2 SSM                 │                            │
   │ ...  (L22–L26 Mamba2 SSM)       │                            │
   │ L27  MLA attention ●────────────┘ shares latent w/ L20       │
   └─────────────┬──────────────────────────────────────────────┘
                 │  every layer: mixer(x) then SwiGLU MLP, both residual
        ┌────────▼────────┐
        │ RMSNorm (final) │
        └────────┬────────┘
                 │
        ┌────────▼────────┐
        │ LM head 768×32000
        └────────┬────────┘
                 ▼
             logits (B, L, 32000)

  aux (training only): L26 hidden ──▶ SelfModelHead ──▶ predicts L26 activation subset
  attention layers with cache: {6, 13, 20, 27}  →  CLA: {6,13}=grp0  {20,27}=grp1
```

## One SSM layer (Mamba2 block)  — L0–L5, L7–L12, L14–L19, L21–L26

```
  x (B,L,768)
    │
  RMSNorm
    │
  in_proj  768 → 3072            split → xin(1536) , z(1536 gate)
    │
  depthwise causal Conv1d(k=4)   over time, per channel
    │  SiLU
  x_proj  1536 → (24 + 128 + 128)     → dt(24 heads) , B(128) , C(128)
    │
  selective scan   state h: (B, 24 heads, 64 headdim, 128 d_state)   ← FIXED size
    │  A=-exp(A_log)  discretize dt   h = dA·h + dt·(x⊗B)   y = h·C + D·x
    │
  y * SiLU(z)                    gate
    │
  out_proj 1536 → 768
    │
  + residual ──▶ (B,L,768)

  KV cache: NONE. State is (24×64×128) regardless of L.  ← the memory win
```

<details><summary>Why the state is constant-size</summary>

The recurrence `h_t = dA·h_{t-1} + input_t` keeps one `(heads, headdim, d_state)` tensor
and overwrites it each step. Sequence length never enters the memory footprint —
generation at token 1 and token 100,000 use the same state. Contrast attention, whose
cache is `(L, ...)`.
</details>

## One MLA attention layer (+ CLA)  — L6, L13, L20, L27

```
  x (B,L,768)
    │
  RMSNorm ── h
    │
    ├─────────────▶ SharedLatentKV  (ONE per CLA group, reused by 2 layers)
    │                 kv_a: 768 → 128 (latent)  + RMSNorm
    │                 k_rope: 768 → 32   (decoupled RoPE key, shared across heads)
    │                 └─▶ c_kv (B,L,128) , k_rope (B,L,32)      ← ONLY THIS IS CACHED
    │
  q_proj 768 → 12×64                    per-head query
    │  split → q_nope(12×32) , q_rope(12×32)
    │
  kv_b_proj  128 → 12×(32+64)           up-project latent → per-head k_nope(32), v(64)
    │
  RoPE:  q_rope ⟲   k_rope ⟲ (broadcast to 12 heads)
    │
  q = [q_nope | q_rope] (B,12,L,64)     k = [k_nope | k_rope] (B,12,L,64)
    │
  scaled_dot_product_attention(q,k,v, causal)      v:(B,12,L,64)
    │
  o_proj  12×64 → 768
    │
  + residual ──▶ (B,L,768)

  cache/token = latent 128 + rope 32 = 160 vals; shared over 2 layers per group
  →  4 attn layers collapse to 2 cached groups  →  640 B/token  (vs 86,016 B MHA)
```

<details><summary>MLA vs MHA vs GQA — what gets cached</summary>

```
  MHA  : cache K(12×64) + V(12×64) per layer × 28   = big
  GQA  : cache K/V for a few grouped heads          = medium
  MLA  : cache one 128-d latent + 32-d rope key     = small
  +CLA : cache once per 2 layers, not per layer     = smaller
```
The full per-head K/V are *reconstructed* from the latent for the attention math, so
quality tracks MHA; only what's *stored between steps* shrinks.
</details>

## SwiGLU MLP (after every mixer, all 28 layers)

```
  x → RMSNorm → [ w_gate 768→3072 , w_up 768→3072 ] → SiLU(gate)*up → w_down 3072→768 → +res
```

## Self-remodeling paths

```
 STRUCTURAL — Mixture-of-Depths (Stage 2, per-token depth)
   scores = router(x)           top-k tokens (capacity 0.5) run the block;
   the rest bypass via residual → effective depth varies per token at inference

 WEIGHT-SPACE — Transformer²/SVF (Stage 3, per-task)
   W = U·diag(s)·Vᵀ    expert vector z rescales s:  W' = U·diag(s·softplus(z))·Vᵀ
   two-pass inference: pass1 detect task → pass2 mix experts → adapted weights
```

## Training objective

```
  loss = CrossEntropy(next-token)  +  λ · MSE( selfmodel(hidden) , hidden_L26.detach() )
                                        λ = 0.1     (self-modeling → simpler internals)
```

## Numbers at a glance

| thing | value |
|---|---|
| layers | 28 (24 SSM + 4 MLA) |
| d_model / heads | 768 / 12 |
| SSM state | 24 heads × 64 × 128 (fixed) |
| MLA latent rank / rope | 128 / 32 |
| CLA groups | 2 (from 4 attn layers) |
| MLP hidden | 3072 (SwiGLU) |
| KV/token | 640 B (134× < MHA's 86,016 B) |
| params | 324.2M total / 299.6M non-embed |

See `README.md` for the high-level view and `MODEL_CARD.md` for citations.
