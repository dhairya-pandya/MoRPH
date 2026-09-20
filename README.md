# MORPH

A ~300M LLM built from scratch to kill the **KV-cache tax** and **remodel itself** at
inference. Trains on one 16GB GPU (Kaggle/Colab).

```
             KV cache @ 32k context
   MHA  ████████████████████████████  2,819 MB
 MORPH  ▏                                  21 MB   (134x smaller)
```

## The stack (28 layers)

```
  tokens
    │
    ▼
 ┌──────────────────────────────────────────────┐
 │  6×  Mamba2 SSM     ← constant state, NO cache │   ┐
 │  1×  MLA attention  ← tiny latent cache        │   │ pattern
 │      · · · repeated 4× · · ·                   │   │ "mmmmmma"
 └──────────────────────────────────────────────┘   ┘  ×4 = 28
    │                                             ↑
    │                          self-modeling head─┘ (aux loss, training only)
    ▼
  RMSNorm → LM head
```

24 SSM layers carry the sequence cheaply; 4 attention layers keep long-range recall.
Only the attention layers cache — and barely.

<details><summary><b>Why SSM-majority? (click)</b></summary>

```
 attention layer          SSM layer
 ────────────────         ─────────────
 cache grows per token    fixed-size state
   t1 t2 t3 t4 ...           [====]  ← same size forever
   ▓  ▓▓ ▓▓▓ ▓▓▓▓
 O(n) memory              O(1) memory
```
SSMs never grow a cache — that's the whole memory win. But pure-SSM models forget
long-range detail, so we keep a few real attention layers. ~6:1 is the Zamba2/Jamba ratio.
</details>

## Why the cache is 134x smaller

```
  MHA per token          MLA + CLA per token
  ─────────────          ───────────────────
  K:  ████████ (12h)     latent:  ██  (rank 128)
  V:  ████████ (12h)     rope key: ▏  (32)
  × 28 layers            × 2 shared groups (not 28!)
  = 86,016 B             = 640 B
```

<details><summary><b>MLA + CLA, in ascii (click)</b></summary>

```
 MLA: don't cache big K/V — cache a small latent, rebuild on read
   x ──▶ down-proj ──▶ [latent 128] ──▶ up-proj ──▶ K,V
                          ▲ this is all that's cached

 CLA: neighboring attention layers SHARE one latent
   layer A ┐
   layer B ┴─▶ [one shared latent]     4 attn layers → 2 caches
```
Latent compression (MLA) + cross-layer sharing (CLA) compound. Recall stays intact
because full K/V are reconstructed for the math; only storage shrinks.
</details>

## The "self-remodel" part

```
 Mixture-of-Depths — each token picks its own compute
   easy token  ──────────────skip──────────────▶   (cheap)
   hard token  ──▶[ layer ]──▶[ layer ]──▶          (full)
                     router decides, per token, at inference
```

<details><summary><b>Self-modeling: the model predicts itself (click)</b></summary>

```
   hidden state ──▶ aux head ──▶ guess its OWN future activations
                                        │
                          loss = LM + λ·(guess − actual)²
```
From Premakumar et al. 2024: forcing a network to predict its own internals makes those
internals *simpler and lower-rank* — which is exactly what makes MLA's low-rank cache work
without losing quality. It's the glue, not a gimmick.
</details>

Plus a Transformer²/SVF adapter (Stage 3) that re-tunes weights per task at inference.

## Run it

```bash
# CPU smoke — no GPU needed
pip install torch numpy && bash scripts/run_tests.sh

# Real training on Kaggle/Colab (resumes across capped sessions via HF Hub)
pip install -r requirements.txt mamba-ssm causal-conv1d
export HF_TOKEN=...
PYTHONPATH=. python -m morph.train.pretrain --config configs/300m_hybrid.json \
  --hub_repo <you>/morph-300m --batch 8 --grad_accum 16 --seq 2048 --session_minutes 540
```

Notebooks for both platforms in `notebooks/`.

## Status

`[x]` Stage 0 arch + tests + KV bench + resumable trainer · `[ ]` Stage 1 pretrain (~50–100B tok)
· `[ ]` Stage 2 MoD · `[ ]` Stage 3 SVF · `[ ]` Stage 4 SFT · `[ ]` Stage 5 publish

<details><summary>Credits</summary>

Combination is the contribution; the parts aren't mine: DeepSeek MLA · CLA (Brandon et al.)
· Mamba2 · Zamba2/Hymba · Mixture-of-Depths (Raposo et al.) · Transformer²/SVF (Sakana) ·
Self-Modeling (Premakumar et al.). Full list in `MODEL_CARD.md`. MIT licensed.
</details>
