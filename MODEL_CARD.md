# Model Card: MORPH-300M

**Status:** in development (Stage 0 complete — architecture + pipeline; not yet trained to convergence).

## Summary
MORPH-300M is a ~300M-parameter (299.6M non-embedding) decoder LM built to minimize
the KV-cache/accuracy tradeoff while self-remodeling for efficiency. It targets
from-scratch training on a single 16GB GPU (Kaggle/Colab).

## Architecture
- 28 layers, d_model=768, 12 heads. Layer schedule `mmmmmma` tiled → 24 SSM + 4 attention.
- **SSM layers:** Mamba2 (constant-size recurrent state, no KV cache).
- **Attention layers:** Multi-head Latent Attention (MLA) — low-rank latent KV
  (rank 128) + decoupled RoPE (32 dim).
- **CLA:** the 4 attention layers share KV in groups of 2 → 2 latent-KV modules.
- **Self-modeling head:** predicts a 256-unit subset of a mid-layer's hidden state
  (λ=0.1) for emergent simplification.
- **Mixture-of-Depths** (Stage 2) and **Transformer²/SVF** (Stage 3): off at init.

## Measured (config-level)
- KV cache per token: **640 B** vs 86,016 B for equal-depth MHA → **134x** smaller.
- At 32k ctx: **21 MB** vs 2,819 MB.

## Intended use / limitations
- Research artifact demonstrating KV-minimal + self-remodeling design at small scale.
- Not yet trained → no quality guarantees until Stage 1+ completes.
- Pure-SSM recall is weaker than full attention; mitigated by the MLA layers + CLA.

## Training (planned)
- Data: FineWeb-Edu (streamed), ~50–100B tokens. Tokenizer: reused 32k BPE.
- Precision: bf16 + grad checkpointing. Resumable across capped sessions via HF Hub.

## Citations
DeepSeek-V2 MLA; CLA (arXiv 2405.12981); Mamba2 (state-spaces); Zamba2/Hymba hybrids;
Mixture-of-Depths (2404.02258); Transformer²/SVF (2501.06252); Self-Modeling in Neural
Networks (Premakumar et al. 2024). Research task `wia4ra29q`, 23/25 claims verified.
