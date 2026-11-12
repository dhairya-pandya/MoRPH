# Model Card: MORPH-S

**Status:** pretrained (3B tokens, validation perplexity 17.4 on FineWeb-Edu, HellaSwag 32.7% zero-shot); instruction-tuned variant (smol-smoltalk, 0.15B tokens, HellaSwag 32.4%) in `sft/`.

## Summary
MORPH-S is a ~170M-parameter (138M non-embedding) decoder LM: a Mamba2-majority hybrid with
four NoPE Multi-head Latent Attention layers sharing two cached latents, trained with a
self-modeling auxiliary objective. It targets from-scratch training on free Kaggle T4×2.

## Architecture
- 24 layers, d_model 640; Mamba2 (20 layers, 20 heads × 64, d_state 128) + MLA at layers
  4, 9, 14, 19 (10 heads × 64, latent 128, no positional encoding); SwiGLU MLP 1728.
- Cross-layer latent sharing: (4 → 9), (14 → 19).
- Self-modeling head predicts the RMS-normalized residual states entering the attention
  layers from the final hidden state; gradients flow into the targets (Premakumar et al.).

## Inference memory (fp16)
512 B/token of cache + 6.74 MB constant SSM state per sequence; 23.5 MB at 32k tokens vs
2,013 MB for equal-depth multi-head attention.

## Training
FineWeb-Edu (sample-10BT), SmolLM2 tokenizer, 3B tokens (12 Kaggle sessions), seq 2048, 262k tokens/step,
Muon + AdamW, WSD schedule, fp16 + loss scaling on T4 (bf16 on Ampere+).

## Limitations
Small and lightly trained (3B tokens): expect weak factual recall and reasoning; HellaSwag 32.7% zero-shot (chance 25%). Four attention layers limit exact long-range recall
compared with full attention. The self-modeling effect has only been shown on small
classifiers; the proxy ablation tests whether it transfers to language modeling.

## Citations
Mamba2 (Dao & Gu 2024) · DeepSeek-V2 MLA · CLA (Brandon et al. 2024) · Kimi Linear (2510.26692)
· Hybrid design study (2510.04800) · Muon / Moonlight (2502.16982) · Self-modeling
(Premakumar et al., arXiv 2407.10188) · SmolLM2 tokenizer · FineWeb-Edu.
