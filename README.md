# MORPH

A small LLM built from scratch to kill the **KV-cache tax**, trained with a **self-modeling**
objective that pushes the model to make its own internals simple. Trains on free Kaggle
T4×2 sessions (fp16) or any Ampere+ GPU (bf16), resuming across capped sessions.

```
             inference memory @ 32k context (MORPH-S, 170M)
   MHA  ████████████████████████████  2,013 MB
 MORPH  ▍                                 24 MB   (86x; 512 B/token + 6.7 MB fixed SSM state)
```

## The stack (MORPH-S: 24 layers, d=640)

```
  tokens
    │
    ▼
  L0–3    Mamba2 SSM ×4          ← constant state, no cache
  L4      MLA attention (NoPE)   ● produces latent A  ──┐ cached (128 values/token)
  L5–8    Mamba2 SSM ×4                                 │
  L9      MLA attention (NoPE)   ○ reuses latent A  ◀───┘
  L10–13  Mamba2 SSM ×4
  L14     MLA attention (NoPE)   ● produces latent B  ──┐ cached
  L15–18  Mamba2 SSM ×4                                 │
  L19     MLA attention (NoPE)   ○ reuses latent B  ◀───┘
  L20–23  Mamba2 SSM ×4
    │                     every layer = mixer + SwiGLU MLP (pre-norm residual)
    ▼
  RMSNorm ──▶ LM head
          └──▶ self-model head: predicts the states entering L4/L9/L14/L19 (training only)
```

Design choices, each from published evidence:
- **1 attention : 5 SSM, attention mid-stack, SSM at both ends** — hybrid ablations
  (Meta, arXiv 2510.04800) find front-loaded attention hurts.
- **NoPE MLA** — Mamba2 already supplies position; the latent is the only cached tensor
  (Kimi Linear, arXiv 2510.26692).
- **True cross-layer sharing** — the second attention layer of a pair reuses the first one's
  latent with its own up-projection: 2 cache entries per token instead of 4.
- **Mamba2 in plain PyTorch** (chunked SSD) with `mamba_ssm`-compatible parameters: no CUDA
  build, runs on T4; fast kernels are optional.

<details><summary><b>Self-modeling (Premakumar et al. 2024, arXiv 2407.10188)</b></summary>

```
  final hidden ──▶ linear "extra output units" ──▶ â  ≈  a = rmsnorm(state entering an attention layer)
  loss = CE + z·lse² + λ · mean((â − a)²)        gradient flows into BOTH â and a
```
The paper found that when a network must predict its own activations it becomes simpler
(lower RLCT, narrower weights) — "learning to self-model is learning to make oneself
modelable". MORPH aims that pressure at the states that get compressed into the cached
latent. Targets are RMS-normalized so the model can't cheat by just shrinking activations.
The proxy ablation tests it: λ ∈ {0, 0.1, 1.0}, a detached-target control, and a half-size
latent (does self-modeling protect a smaller cache?).
</details>

## Run it

```bash
# CPU gate (20 tests: SSD parity, causality, CLA, self-model grads, exact resume, DDP)
bash scripts/run_tests.sh

# data: FineWeb-Edu -> uint16 shards, SmolLM2 tokenizer (CPU, ~5.2B tokens)
PYTHONPATH=. python -m morph.data.prepare --out data/fineweb_edu_smollm2

# train (1 GPU, or torchrun --nproc_per_node=2); resumes automatically
PYTHONPATH=. python -m morph.train.pretrain --model_config configs/model/s.json \
  --train_config configs/train/main_s.json --set data_dir=data/fineweb_edu_smollm2
```

On Kaggle, `scripts/launch_kaggle.py` pushes private kernels via the Kaggle CLI:
`prep` (CPU), `gate` (throughput check), `proxy R0 R1` (two ablation runs, one per T4),
`main` (a 12 h session of the main run). Checkpoints mirror to a private HF repo when a
Kaggle secret `HF_TOKEN` is attached.

## Status

`[x]` v2 architecture + pipeline · `[~]` data prep + GPU gate (Kaggle) · `[ ]` proxy ablation
(R0–R5) · `[ ]` main run MORPH-S, 5B tokens · `[ ]` MoD · `[ ]` SVF · `[ ]` SFT · `[ ]` publish

<details><summary>Credits</summary>

Mamba2 / SSD (Dao & Gu) · DeepSeek MLA · CLA (Brandon et al.) · Kimi Linear · Muon /
Moonlight · Mixture-of-Depths (Raposo et al.) · Transformer²/SVF (Sakana) · Self-Modeling
(Premakumar et al.). MIT licensed.
</details>
