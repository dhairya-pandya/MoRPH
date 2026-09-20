# MORPH — Modular self-remORPHing LLM

An open-source **~300M** language model on a **novel hybrid architecture** that
**minimizes the KV-cache memory/latency vs. accuracy tradeoff** and can **self-remodel**
— reconfigure its own structure for efficiency. Trainable **from scratch on a single
Kaggle/Colab GPU (16GB)**.

## Why

Full-attention KV cache grows linearly with context and dominates inference memory.
MORPH attacks this from four proven angles at once:

| # | Mechanism | Role |
|---|-----------|------|
| 1 | **SSM-majority hybrid** — ~6:1 Mamba2 : MLA-attention | Most layers carry a *constant-size* recurrent state (no KV cache); a few MLA layers keep long-range recall |
| 2 | **MLA + CLA** (latent KV + cross-layer sharing) | The few attention layers cache only a small shared latent + one RoPE key |
| 3 | **Mixture-of-Depths** router | Per-token adaptive depth — the model *structurally* remodels its compute at inference |
| 4 | **Transformer²/SVF** expert vectors | Inference-time *weight-space* self-adaptation per task |
| + | **Self-modeling aux loss** (Premakumar et al. 2024) | Network predicts its own hidden state → emergent simplification → more compressible internals (makes 1–2 hold quality) |

### Measured KV-cache reduction (300M config, vs equal-depth MHA)

```
per-token KV: MHA=86016 B  MORPH=640 B  reduction=134.4x
   ctx     MHA (MB)   MORPH (MB)  reduction
  4096        352.3          2.6     134.4x
 16384       1409.3         10.5     134.4x
 32768       2818.6         21.0     134.4x
```

## Layout

```
morph/model/     backbone, mamba2 (pytorch fallback + fused), mla+cla, mod, svf, self-model head
morph/data/      resumable streaming FineWeb-Edu loader (+ synthetic for CPU tests)
morph/train/     resumable session-capped pretrainer, HF-Hub checkpointing
morph/eval/      kv_bench (headline table), quality_bench (perplexity + hooks)
configs/         300m_hybrid.json
notebooks/       kaggle_train.ipynb, colab_train.ipynb
tests/           Stage-0 gate (CPU, no GPU needed)
```

## Quickstart (CPU dev / smoke)

```bash
pip install torch numpy
PYTHONPATH=. python -c "import tests.test_model as T; [getattr(T,n)() for n in dir(T) if n.startswith('test_')]"
PYTHONPATH=. python -m morph.eval.kv_bench --config configs/300m_hybrid.json
PYTHONPATH=. python -m morph.train.pretrain --config configs/300m_hybrid.json --smoke --steps 20 --batch 2 --seq 128
```

## Training on Kaggle/Colab (16GB, multi-session)

```bash
pip install -r requirements.txt mamba-ssm causal-conv1d
export HF_TOKEN=...   # for off-box checkpoints
PYTHONPATH=. python -m morph.train.pretrain \
  --config configs/300m_hybrid.json --hub_repo <you>/morph-300m \
  --batch 8 --grad_accum 16 --seq 2048 --session_minutes 500
```

Sessions are time-capped; the trainer resumes from the latest HF-Hub checkpoint
(model + optimizer + LR + data offset) each run. Target: **~50–100B tokens**.

## Roadmap (stages)

- **Stage 0** ✅ skeleton, CPU tests, 300M sizing, KV-bench, resumable trainer
- **Stage 1** pretrain hybrid + self-modeling (~50–100B tokens, multi-session)
- **Stage 2** enable Mixture-of-Depths, anneal for adaptive depth
- **Stage 3** Transformer²/SVF expert-vector self-adaptation
- **Stage 4** light instruction SFT → **Stage 5** publish weights + model card

## Provenance

Design backed by a fan-out deep-research pass (23/25 claims adversarially verified).
See `MODEL_CARD.md` and the plan for citations (DeepSeek MLA, CLA, Mamba2, Zamba2,
MoD, Transformer²/SVF, self-modeling).
