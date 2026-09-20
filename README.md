# MORPH

A small language model I'm building from scratch to chip away at one specific,
annoying problem: the KV cache.

If you've ever tried to run a transformer over a long context, you know the pain.
The KV cache grows with every token, and pretty soon it — not the weights — is what's
eating your VRAM. MORPH is a ~300M-parameter model designed so that most of its layers
simply don't have a KV cache to grow, and the few that do keep it tiny. The goal is a
model you can actually pretrain from scratch on a single free-tier GPU (Kaggle or Colab,
16GB) without the memory falling over on long sequences.

The name stands for **M**odular self-rem**ORPH**ing — because the second thing it does is
reconfigure its own structure at inference to save compute. More on that below.

## The idea

There's no single trick here. There are four, each borrowed from work that already
proved it out, stacked so they reinforce each other:

1. **Most layers are Mamba2 (an SSM), not attention.** State-space layers carry a
   fixed-size recurrent state instead of a cache that grows with the sequence. About
   6 out of every 7 layers are SSM. They do the heavy lifting cheaply.

2. **The few attention layers use MLA + CLA.** Where we *do* keep real attention — to
   preserve the long-range recall that pure SSMs are weak at — we use Multi-head Latent
   Attention, which compresses keys/values into a small latent vector, and Cross-Layer
   Attention, which lets neighboring attention layers share that latent. So even the
   caching layers barely cache anything.

3. **Mixture-of-Depths lets tokens skip layers.** Easy tokens take a shortcut; hard ones
   get the full stack. The model decides per token, at inference. That's the
   "self-remodeling" part — it reshapes how much compute it spends on the fly.

4. **It learns to model itself.** During training there's an auxiliary loss where the
   network predicts its own hidden activations (from Premakumar et al. 2024). This sounds
   odd, but the paper shows it pushes the network toward simpler, lower-rank internal
   representations — and simpler internals are exactly what makes the low-rank KV
   compression in step 2 work without hurting quality. It's the glue that holds the rest
   together.

There's also a Transformer²/SVF adapter planned (step 3 of the roadmap) that re-tunes the
weights per task at inference, but that comes after the base model trains.

## Does it actually shrink the cache?

Here's the config-level number, comparing MORPH-300M against a same-depth vanilla
multi-head-attention model:

```
per-token KV: MHA=86,016 B  ->  MORPH=640 B   (134x smaller)

   context     MHA        MORPH     reduction
     4,096   352 MB       2.6 MB      134x
    16,384   1,409 MB    10.5 MB      134x
    32,768   2,819 MB    21.0 MB      134x
```

At a 32k context the cache goes from ~2.8 GB to ~21 MB. That's the whole point.

## What's actually here

This is Stage 0 — the architecture and the training pipeline, fully wired and tested,
but not yet trained to convergence. What works today:

- The full model (`morph/model/`) — config, the Mamba2 block (with a pure-PyTorch
  fallback so it runs on a Mac/CPU when the CUDA kernels aren't available), MLA+CLA
  attention, the self-modeling head, and scaffolds for Mixture-of-Depths and SVF.
- A resumable, session-capped trainer (`morph/train/`). Kaggle and Colab kill your
  session after a few hours, so the trainer checkpoints everything — model, optimizer,
  LR schedule, and its place in the data stream — pushes it to the Hugging Face Hub, and
  picks up exactly where it left off next session.
- A streaming FineWeb-Edu data loader, plus a synthetic stream so the smoke tests run
  anywhere.
- The KV benchmark that produced the table above, and a perplexity eval.
- Tests that pass on plain CPU, no GPU required.

## Try it (CPU, no GPU)

```bash
pip install torch numpy
bash scripts/run_tests.sh
PYTHONPATH=. python -m morph.train.pretrain --config configs/300m_hybrid.json \
    --smoke --steps 20 --batch 2 --seq 128
```

## Train it for real (Kaggle / Colab)

```bash
pip install -r requirements.txt mamba-ssm causal-conv1d
export HF_TOKEN=...        # so checkpoints survive the session dying
PYTHONPATH=. python -m morph.train.pretrain \
  --config configs/300m_hybrid.json --hub_repo <you>/morph-300m \
  --batch 8 --grad_accum 16 --seq 2048 --session_minutes 540
```

Run it, let the session expire, run it again — it resumes from the Hub. The plan is
~50–100B tokens, which is many sessions, so a valid checkpoint always exists along the way.
There are ready-to-run notebooks in `notebooks/` for both platforms.

## Where it's going

- [x] **Stage 0** — architecture, tests, KV benchmark, resumable trainer
- [ ] **Stage 1** — pretrain the hybrid backbone with self-modeling (~50–100B tokens)
- [ ] **Stage 2** — turn on Mixture-of-Depths and anneal it in
- [ ] **Stage 3** — add the Transformer²/SVF self-adaptation adapter
- [ ] **Stage 4** — a light instruction-tuning pass so it's usable
- [ ] **Stage 5** — publish the weights and a proper model card

## Credit where it's due

None of the individual pieces are mine — the contribution is the combination at this
scale. The design leans on DeepSeek's MLA, Cross-Layer Attention (Brandon et al.), Mamba2,
the Zamba2/Hymba hybrids, Mixture-of-Depths (Raposo et al.), Transformer²/SVF (Sakana),
and the self-modeling work of Premakumar et al. See `MODEL_CARD.md` for the full list.

MIT licensed. If you build on it, I'd love to hear about it.
