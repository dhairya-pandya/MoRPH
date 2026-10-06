# MORPH v2 — Architecture Revision + Self-Modeling + Kaggle/Colab Training Pipeline

Date: 2026-09-26 · Status: DRAFT for review · Scope: Stage 1 (pretraining) only

## 1. Intent

Train MORPH from scratch on free/cheap GPUs (Kaggle T4×2, Colab Pro) with the
self-modeling objective of Premakumar et al. (arXiv 2407.10188; AF post "Self-prediction
acts as an emergent regularizer") implemented faithfully, and test whether it makes the
model's cached internals simpler — the property MORPH's KV-minimal design relies on.

Decisions confirmed by the user (2026-09-26):
- Main model **MORPH-S (~170M)**, ~**5B tokens**.
- **Proxy ablation first** (small model) to choose self-modeling settings from evidence.
- Checkpoints on **HF Hub** (private), so Kaggle and Colab sessions resume each other.
- Claude launches Kaggle kernels through the user's local `kaggle` CLI (confirming each
  push); Colab via the official `colab` CLI (installed 2026-09-26).

Success criteria:
1. Pipeline trains MORPH-S across ≥2 capped sessions with exact resume (same data order,
   optimizer state, schedule), on T4 (fp16) and on Ampere+ (bf16), 1 or 2 GPUs.
2. Proxy ablation reports: val loss, self-model diagnostics (effective rank, rank-r energy,
   weight width, LLC) for λ=0 vs self-modeling variants, and the rank-halving interaction.
3. Main run reaches 5B tokens with a usable checkpoint at every stage (floor checkpoint).

Non-goals (this round): MoD (Stage 2), SVF (Stage 3), SFT, cached generation/inference
engine, long-context extension, publishing weights.

## 2. Findings that drive the changes

| # | Finding | Consequence |
|---|---------|-------------|
| F1 | Paper: self-model = **extra linear output units** predicting chosen hidden activations; loss `w_c·L_c + (w_s/n)·‖â−a‖²`; **no stop-gradient** — "learning to self-model is learning to make oneself modelable". Current code detaches the target and uses a 2-layer MLP head. | Rewrite head: linear, gradient into targets. |
| F2 | Current CLA is not sharing: each attention layer recomputes the latent from *its own* input with shared weights ⇒ a real cache would hold 4 latents; the 640 B/token claim is wrong. | True CLA: producer computes latent, consumer reuses it. |
| F3 | Default tokenizer = Llama-3 (128k vocab, gated) vs `vocab_size=32000` ⇒ index errors. | SmolLM2 tokenizer (49,152, ungated, FineWeb-Edu-trained). |
| F4 | Kaggle retired P100 on 2026-09-15 ⇒ Kaggle GPU = **T4×2 only** (sm_75, **no bf16**), 30 GPU-h/week, 12 h sessions. | fp16 autocast + GradScaler on T4; bf16 on Ampere+. |
| F5 | `mamba-ssm` has no wheels for current Colab/Kaggle (Py3.12, torch 2.11, cu128); source build hangs (state-spaces/mamba#983). Fallback = Python loop over timesteps (untrainable); fused vs fallback params differ. | Own Mamba2 with chunked SSD in pure PyTorch, param-compatible with `mamba_ssm.Mamba2`. |
| F6 | Hybrid design studies: ~1:5 attention:SSM, **attention mid-stack, SSM at the extremes**, **NoPE** in hybrids (Meta 2510.04800; Kimi Linear 2510.26692 uses NoPE-MLA). Current: attention at last layer, decoupled RoPE. | Explicit mid-stack attention list; NoPE MLA ⇒ cache = latent only. |
| F7 | 50–100B tokens is ~1 year of Kaggle quota for ~300M. | Right-size: MORPH-S, 5B tokens (~2–2.5 weeks Kaggle, less with Colab). |
| F8 | Streaming `ds.skip(n)` re-reads skipped docs; no DDP; cosine needs fixed total; step-count-only checkpoints. | Pre-tokenized shards, deterministic sampler, DDP, WSD, time-based checkpoints. |

Considered and rejected: Gated DeltaNet/KDA mixers (better recall, but Triton kernels risky on
T4); NextLat next-latent prediction (arXiv 2511.05963: stop-grad targets; *worse* LM perplexity
10.88 vs 10.52 at 1.3B/100B tokens).

## 3. Model v2

### 3.1 Presets

| preset | d | layers | attention layers | CLA groups | heads×dim | latent r | Mamba2 (di / heads / N) | MLP | total / non-emb |
|---|---|---|---|---|---|---|---|---|---|
| XS (proxy) | 384 | 12 | [3, 8] | [[3, 8]] | 6×64 | 64 | 768 / 12 / 128 | 1024 | ~44M / ~25M |
| **S (main)** | 640 | 24 | [4, 9, 14, 19] | [[4, 9], [14, 19]] | 10×64 | 128 | 1280 / 20 / 128 | 1728 | ~168M / ~137M |

Common: vocab 49,152 (SmolLM2), tied embeddings, pre-norm RMSNorm (eps 1e-5), every layer =
mixer + SwiGLU MLP (both residual), final RMSNorm, no dropout, no biases (except conv1d).

### 3.2 Mamba2 mixer (`morph/model/mamba2_block.py`, `morph/model/ssd.py`)
Parameter layout identical to `mamba_ssm.modules.mamba2.Mamba2` (ngroups=1):
`in_proj` d→[z(di), x(di), B(N), C(N), dt(H)], depthwise causal `conv1d` (k=4, bias) over xBC,
SiLU, `dt = softplus(dt + dt_bias)`, `A = −exp(A_log)`, SSD scan, `+ D·x`,
`norm` = gated RMSNorm `rmsnorm(y·silu(z))`, `out_proj` di→d. Mamba2 inits for `dt_bias`
(dt∈[1e-3, 0.1] log-uniform, floor 1e-4), `A_log` (A∈U[1,16]), `D=1`.

SSD backend, chosen per device at runtime:
1. `torch` (default, always available): chunked SSD ("ssd_minimal" algorithm; chunk Q=64,
   configurable). Pads L to a multiple of Q (padded dt=0 ⇒ no effect). Decays/cumsums/exp and
   inter-chunk state recurrence in fp32. `torch.compile`-friendly.
2. Optional fast paths if importable and validated by the GPU gate: `mamba_ssm`
   `mamba_chunk_scan_combined`; `fla` `chunk_simple_gla` (SSD ≡ simple-GLA with q=C, k=B,
   v=x·dt, g=A·dt, scale=1). Same parameters ⇒ checkpoints portable across backends.

Sequential reference scan kept in `ssd.py` for tests only.

### 3.3 MLA with NoPE + true CLA (`morph/model/mla_attention.py`)
- Producer layer: `c = RMSNorm(W_dkv · h)` (r dims) — **the only cached tensor**.
- Every attention layer (producer and consumer): `q = W_q h` (H×64), `[k, v] = W_ukv · c`
  (per-layer up-projection, H×(64+64)), causal SDPA (scale 1/√64), `o_proj`.
- Consumer layers take `c` from their group's producer (computed from the producer's input),
  own no `W_dkv`. No RoPE anywhere; Mamba2 supplies position/recency.
- Absorb-compatible for later inference (k linear in c, no norm on k).
- Backbone passes `{group_id: c}` between layers; checkpoint-safe.

KV accounting (S, fp16, per token): 2 groups × 128 × 2 B = **512 B** vs equal-depth MHA
(24 × 2 × 640 × 2 B) = 61,440 B ⇒ **120×** (60× without CLA). Plus constant SSM state
20 layers × 20 × 64 × 128 × 2 B ≈ 6.6 MB/sequence (+0.2 MB conv state), reported honestly by
`kv_bench`. At 32k: ~23.5 MB total vs ~2.0 GB MHA.

### 3.4 Self-modeling head (`morph/model/self_model_head.py`)
Faithful to the paper, adapted to a pre-norm residual LM:
- **Source**: final representation `h_f = norm_f(x_L)` (the input to the LM head) — the paper
  "augmented the output layer".
- **Targets**: residual-stream states entering each attention layer (default
  `self_model_targets = attn_layers`) — the states compressed into the cached latent.
- **Head**: one linear map `W ∈ R^{(T·d)×d}` + bias (T targets), i.e. extra output units.
- **Target normalization**: `a_k = x_k / rms(x_k)` (parameter-free, per token). Reason: in a
  pre-norm residual net a global rescale of the residual stream is functionally free, so an
  unnormalized MSE can be driven down by shrinking activations without simplifying anything.
  Normalizing removes that shortcut; only *structure/direction* can be made predictable.
- **Loss**: `L_sm = mean_k mean_{tokens,units} (W_k h_f + b_k − a_k)²` (the paper's `/n`).
  **No stop-gradient** by default (`self_model_detach=false`); `true` is the ablation control.
- **Total loss**: `L = CE + z_loss·logsumexp² + λ(t)·L_sm`, `λ(t)` ramps 0→λ linearly over the
  LR warmup, then constant. Default λ=0.1 until the proxy ablation picks λ*.
- **Diagnostics** (eval-time, fixed eval batch): per-target self-model R², effective rank
  (Roy–Vetterli, `exp(H(σ/Σσ))`) of `a_k`, rank-r energy `Σ_{i≤r}σ_i²/Σσ_i²` (how much of the
  state a rank-r latent can hold), weight std per module type (paper's width metric), and
  offline LLC via SGLD (paper's RLCT) for proxy checkpoints.

### 3.5 Losses (`morph/model/losses.py`)
Chunked cross-entropy over token chunks (default 4096 tokens) with per-chunk activation
recompute ⇒ never materializes full fp32 logits (49k vocab). z-loss 1e-4. Inputs/targets
from windows of `seq_len+1` tokens: `x=w[:-1]`, `y=w[1:]` (no wasted token).

### 3.6 Init
Linear/embedding N(0, 0.02); residual output projections (`out_proj`, `o_proj`, `w_down`)
scaled by 1/√(2·n_layers); Mamba2-specific inits as in 3.2; self-model head N(0, 0.02), bias 0.

## 4. Data pipeline

### 4.1 Preparation (`morph/data/prepare.py`, runs on Kaggle CPU — no GPU quota)
- Source `HuggingFaceFW/fineweb-edu`, config `sample-10BT`, streamed.
- Tokenizer `HuggingFaceTB/SmolLM2-135M` (49,152; EOS=`<|endoftext|>` id 0 appended per doc).
- Validation = every 1000th document (deterministic, disjoint) until ≥5M tokens.
- Train tokens until target (default 5.2B), written as raw little-endian **uint16** shards of
  100M tokens (`train_00000.bin`…) + `val_00000.bin` + `meta.json` (tokenizer, vocab, eos,
  per-shard counts, source, doc counts, code version).
- Parallel tokenization with `tokenizers` batch encode; resumable by shard (skips finished
  shards on restart).
- Output: Kaggle kernel output (~10.4 GB, used by Kaggle training kernels as input) and a
  mirror in a private HF dataset repo (used by Colab).

### 4.2 Loading (`morph/data/shards.py`)
- `np.memmap` shards; sample = window of `seq_len+1` tokens at stride `seq_len`.
- Global permutation of all window ids from `(seed, epoch)`; optimizer step `s` consumes
  permutation entries `[s·G, (s+1)·G)` where G = global batch in sequences. Rank `r` of `W`
  takes entries `r, r+W, …`, then splits into micro-batches.
- ⇒ data order depends only on `(seed, G, seq_len)`, **not** on world size or micro-batch:
  a run may resume on 1 GPU (Colab) after 2 GPUs (Kaggle) and see identical data.
- Cursor = optimizer step (checkpointed). Background prefetch thread + pinned memory.
- `SyntheticSource` (random tokens) for CPU smoke tests.

## 5. Trainer (`morph/train/`)

### 5.1 Configuration
`configs/model/{xs,s}.json` (MorphConfig) + `configs/train/{proxy_xs,main_s}.json`
(TrainConfig); every field overridable from CLI (`--set key=value`). Run name + output dir
per run; the config + git SHA are stored in every checkpoint.

| setting | proxy XS | main S |
|---|---|---|
| seq_len | 2048 | 2048 |
| global batch | 64 seq (131k tok) | 128 seq (262k tok) |
| tokens | 0.5B (~3.8k steps) | 5.0B (~19.1k steps) |
| peak LR | 3e-3 | 2e-3 |
| warmup | 200 steps | 500 steps |
| schedule | WSD, decay last 20% (1−sqrt) to 0 | same |

### 5.2 Hardware adaptation (`morph/train/distributed.py`)
- `torchrun` DDP when `WORLD_SIZE>1` (NCCL; gloo on CPU for tests), `no_sync` during
  accumulation, rank-0-only I/O.
- Precision: sm ≥ 80 ⇒ bf16 autocast, TF32 on; sm 75 (T4) ⇒ fp16 autocast + dynamic
  GradScaler; CPU ⇒ fp32. Master weights fp32 always; residual stream fp32.
- `grad_accum = G / (micro_batch × world)`; `micro_batch` from config or `auto` (gate result
  file, else OOM-backoff probe).
- Activation checkpointing per layer (default on for ≤24 GB GPUs); optional `torch.compile`
  per layer (enabled when the gate shows a speedup).

### 5.3 Optimization (`morph/train/optim.py`, `schedule.py`)
- **Muon** for 2-D weight matrices inside blocks (in_proj split into its z/x/B/C/dt row
  blocks, out_proj, q/kv/o projections, MLP, self-model head): Nesterov momentum 0.95,
  Newton–Schulz 5 steps (fp32 on T4, bf16 on Ampere+), update scaled `0.2·√max(m,n)` to match
  AdamW RMS (Moonlight), decoupled WD 0.1. Own implementation (portable, no torch ≥2.9 need).
- **AdamW** (β 0.9/0.95, eps 1e-8) for embeddings, norms, conv1d, `A_log`, `D`, `dt_bias`,
  biases; WD 0 for these.
- `--optimizer adamw` switches everything to AdamW (WD 0.1 on matrices).
- WSD: warmup → constant → decay over the final `decay_frac` of the *planned* token budget.
  The budget may be raised on resume while still in the stable phase (extends the run).
- Grad clip 1.0 (after unscale); non-finite loss ⇒ skip step; 20 consecutive skips ⇒ abort
  with a clear message.

### 5.4 Checkpointing & resume (`morph/train/checkpoint.py`)
- Full checkpoint = model + both optimizers + scaler + step/tokens + schedule plan + RNG +
  configs + git SHA + recent metrics. Written atomically (`tmp` → rename).
- Triggers: every `ckpt_minutes` (default 45), at the session deadline, at the end. Keep the
  last 2 locally.
- Session deadline: `--time_limit_hours` (Kaggle default 11.5); stop at a step boundary with
  ≥10 min reserve; SIGTERM/SIGINT ⇒ checkpoint at next boundary. All ranks agree via
  broadcast.
- HF Hub: push each checkpoint to a private model repo in the background (`run_as_future`),
  keep the latest 2 there, wait for pending uploads before exit; failures logged, training
  continues. Milestone **weights-only** safetensors every 500M tokens, kept permanently.
- Resume discovery: newest step among local `ckpt_dir`, `/kaggle/input/*/**` (Kaggle output
  chaining fallback), and the HF repo. Model config must match (hash); train-config changes
  are allowed with a logged diff (e.g. micro-batch, world size, budget).

### 5.5 Logging & in-run eval
- Every `log_every` steps: loss, CE, self-model loss, z-loss, λ, LR, grad norm, loss scale,
  tokens, tokens/s, MFU estimate, peak memory → stdout + `metrics.jsonl` (pushed with
  checkpoints); W&B if `WANDB_API_KEY` present.
- Every `eval_every` steps: val loss on a fixed 1M-token val slice + self-model diagnostics
  (3.4) on a fixed 64k-token batch.

## 6. Evaluation tools (`morph/eval/`)
- `kv_bench.py` — corrected accounting (NoPE, true CLA, SSM state).
- `diagnostics.py` — effective rank, rank-r energy, R², weight width (shared with trainer).
- `llc.py` — SGLD local learning coefficient (nβ, ε, γ, steps, chains; `--calibrate` sweeps
  ε for the plateau). Used on proxy checkpoints.
- `hellaswag.py` — 10,042-example validation accuracy (normalized-length log-likelihood);
  run at main-run milestones.
- `compare_runs.py` — table of final metrics across runs from their `metrics.jsonl`.
- `quality_bench.py` — updated to shards + v2 model (val perplexity).

## 7. GPU gate (`morph/tools/gpu_gate.py`)
Run first on each new hardware type: device/capability, chosen precision, backend parity
(torch SSD vs any fast path, fwd+grad tolerance), tokens/s and peak memory over micro-batch ∈
{2,4,8,16} × {ckpt on/off} × {compile on/off}, one fp16 GradScaler step sanity check, a DDP
all-reduce check when 2 GPUs. Writes `gate_<gpu>.json` (recommended micro-batch, compile,
backend) consumed by `--micro_batch auto`; prints projected hours for the configured budget.

## 8. Launch automation
- `scripts/kaggle/` — kernel templates: `prep` (CPU, internet) and `train` (GPU
  `NvidiaTeslaT4`, internet; sources: prep kernel output). Notebook = clone repo at a pinned
  commit → install → `torchrun --nproc_per_node=2 -m morph.train.pretrain …`.
  `scripts/launch_kaggle.py` fills metadata, `kaggle kernels push`, polls `status`, pulls
  logs/output. HF token via Kaggle secret `HF_TOKEN` (one-time manual attach in the Kaggle UI;
  if unavailable, resume falls back to Kaggle output chaining).
- `scripts/launch_colab.py` — proxy/main sessions via `colab new --gpu … / exec / download /
  stop`; code via `git clone` at the pinned commit on the VM; accelerator chosen from
  `colab usage` balance.
- Manual path: updated `notebooks/kaggle_train.ipynb`, `notebooks/colab_train.ipynb`.
- Code reaches Kaggle/Colab by `git clone` of the public GitHub repo, so implementation lands
  on a branch that must be pushed (user approval required before any push/launch).

## 9. Experiment plan

**Phase 0 — build (local, CPU).** All of §3–8; tests (§10) green; CPU smoke + 2-process gloo
DDP run.

**Phase 1 — data (Kaggle CPU).** 5.2B train + ≥5M val tokens.

**Phase 2 — gate + proxy ablation (Kaggle T4×2; user choice 2026-09-26 — the Colab
account has 0 compute units).** XS, 0.5B tokens each (~2–3 h per run, ~12–18 h total):

| run | λ | target grad | latent r |
|---|---|---|---|
| R0 | 0 | — | 64 |
| R1 | 0.1 | flows (paper) | 64 |
| R2 | 1.0 | flows (paper) | 64 |
| R3 | λ* of R1/R2 | detached | 64 |
| R4 | 0 | — | 32 |
| R5 | λ* | flows | 32 |

Decision rule: λ* = the larger λ whose final val loss ≤ R0 + 0.5%; else 0.1 if R1 within
1%; else self-modeling off for the main run (and report). Claims: mechanism (R1/R2 vs R3),
complexity reduction (rank/energy/LLC vs R0), cache interaction ((R5−R4) vs (R1−R0)).

**Phase 3 — main run (Kaggle T4×2 sessions + optional Colab bursts).** MORPH-S, 5B tokens,
λ*; milestone HellaSwag every 1B tokens; decay phase starts at 4B tokens.

Compute estimate (to be replaced by gate measurements): XS run ≈ 1.3e17 FLOPs (~0.5 h A100,
~1 h L4, ~2–3 h T4×2); S main ≈ 5e18 FLOPs (~70 h T4×2 ≈ 2.3 Kaggle weeks at an assumed 20
effective TFLOPS; 1 A100-hour ≈ 3.5 T4×2-hours).

## 10. Testing (CPU, `scripts/run_tests.sh`)
1. SSD chunked == sequential reference (outputs + grads; several Q, L not multiple of Q).
2. Mamba2 parameter names/shapes match `mamba_ssm.Mamba2` (hard-coded expectation).
3. Whole-model causality (perturbing token t+k never changes logits ≤ t).
4. CLA: consumers own no `W_dkv`; consumer uses producer latent; KV accounting numbers.
5. Self-model: grads reach target-producing layers iff not detached; loss invariant to
   residual-stream rescaling; λ ramp.
6. Chunked CE == `F.cross_entropy` (value + grad); z-loss value.
7. Data: `prepare` round-trip on tiny text; sampler identical across world sizes/micro-batch;
   resume from cursor reproduces the stream; val/train disjoint.
8. WSD shape; budget extension while stable; Muon update ~orthogonal; every parameter in
   exactly one optimizer group.
9. Trainer: N steps continuous == N/2 + checkpoint + resume + N/2 (fp32 CPU, exact params);
   atomic save + retention; deadline stop.
10. DDP (gloo, 2 procs) == 1 proc at equal global batch (tolerance).
GPU-only behaviour (fp16 scaler, NCCL, compile, fast paths) is verified by the gate (§7).

## 11. Risks
| risk | mitigation |
|---|---|
| T4 throughput below estimate (pure-PyTorch SSD, 70 W T4) | gate measures first; fast-path backends; shift more work to Colab; budget adjusts (WSD) |
| fp16 overflow/instability on T4 | GradScaler, z-loss, fp32 SSD internals + residual stream, clip; bf16 hosts |
| self-model over-regularizes / collapses rank | λ ramp, diagnostics each eval, proxy decides λ |
| Kaggle secret not attached to CLI-pushed kernel | one-time UI attach; Kaggle output-chaining fallback |
| HF upload failure/limits | background retries, local retention, weights-only milestones small |
| 12 h cap mid-step | time-based checkpoints + deadline stop with reserve + signal handler |

## 12. Existing code
Kept unchanged: `mod_router.py`, `svf.py` (Stage 2/3; MoD remains SSM-only and is known
non-causal — noted in PROGRESS). Replaced: `fineweb.py` streaming loader (superseded by
shards), old `pretrain.py` loop. Docs (`README.md`, `docs/ARCHITECTURE.md`, `MODEL_CARD.md`)
updated to v2 numbers after Phase 0.
