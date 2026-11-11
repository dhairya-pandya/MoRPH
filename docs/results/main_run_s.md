# Main run: MORPH-S (170M, λ=1.0) on FineWeb-Edu

Status: in progress — 2.62B of 3B tokens after 10 Kaggle sessions (T4×2, fp16, DDP).
Final numbers land when the last session finishes.

Setup: `configs/model/s.json` + `configs/train/main_s.json`. Global batch 128 × 2048 tokens,
Muon (matrices) + AdamW (embeddings, norms, heads), peak LR 2e-3, WSD schedule (decay over the
last 20% of tokens), z-loss 1e-4, self-model λ = 1.0 ramped over warmup. ~9.0k tokens/s on two
T4s, so a 12-hour session covers ~0.37B tokens. Sessions resume from the previous session's
output (`require_resume` guard), checkpointing every 45 minutes and at the deadline.

Validation (FineWeb-Edu held-out shard, 2048-token windows) and diagnostics on the states
entering the four attention layers:

| step | tokens | val ppl | erank L4 | erank L19 | self-model R² L4 | self-model R² L19 |
|---|---|---|---|---|---|---|
| 250 | 0.07B | 124.6 | 406 | 364 | 0.71 | 0.88 |
| 1750 | 0.46B | 28.0 | 458 | 553 | 0.74 | 0.95 |
| 3250 | 0.85B | 24.1 | 492 | 563 | 0.65 | 0.92 |
| 4750 | 1.25B | 22.5 | 506 | 563 | 0.62 | 0.91 |
| 6250 | 1.64B | 21.6 | 514 | 564 | 0.59 | 0.91 |
| 7750 | 2.03B | 21.0 | 519 | 564 | 0.57 | 0.90 |
| 9250 | 2.43B | 20.1 | 523 | 564 | 0.56 | 0.89 |
| 9750 | 2.56B | 19.0 | 523 | 563 | 0.56 | 0.89 |

erank = effective rank of the RMS-normalized state (out of 640). R² = variance of that state
explained by the self-model head's prediction from the final hidden state.

Observations so far:
1. **The decay phase pays off immediately.** Validation perplexity fell 20.1 → 19.0 in the
   first 0.16B tokens of LR decay (from 2.4B), more than over the previous 0.4B tokens at peak LR.
2. **Self-modeling does not stop the representation from growing.** Effective rank grows and then
   plateaus: L19 reaches 564 by 0.5B tokens, L4 climbs slowly to 523 by 2.4B. The proxy
   ablation showed λ=1.0 lowers erank relative to λ=0 (−8% at XS), not that it shrinks it
   over training.
3. **Late states are easy to self-model, early ones get harder.** R² for the state entering L19
   stays near 0.9 (it is a few layers below the final hidden state the head reads from); for
   L4 it falls from 0.74 to 0.56 as early layers specialise. The penalty is therefore mostly
   felt at the first attention layer.
