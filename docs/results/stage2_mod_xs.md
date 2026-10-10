# Stage 2: Mixture-of-Depths on the XS proxy

Setup: both runs continue the proxy R2 checkpoint (MORPH-XS 44M, λ=1.0, 0.5B tokens) for
another 0.2B tokens (1,526 steps × 131k tokens, new data order, seed 2024), Muon+AdamW,
peak LR 1e-3, WSD, one Kaggle T4 each, single seed.
MoD run: layers 1, 5, 7, 10 (4 of 10 SSM layers) routed; capacity annealed 1.0 → 0.5 over the
first 600 steps; router aux BCE weight 0.01.

| run | val CE (top-k) | Δ vs control | val CE (causal routing) | tokens running routed layers | tok/s |
|---|---|---|---|---|---|
| control (no MoD) | 3.3266 | — | — | 100% | 14.2k |
| MoD | 3.4025 | +2.28% | 3.4507 (+3.73%) | 47.8% (causal) | 16.8k (+19%) |

Validation CE over the run (every 250 steps):
- control: 3.397 3.390 3.382 3.375 3.363 3.327 3.327
- MoD top-k: 3.690 3.530 3.481 3.462 3.442 3.403 3.403
- MoD causal: 3.717 3.595 3.549 3.526 3.495 3.451 3.451

Findings:
1. **Routing works as built.** Causal routing executes 47.8% of tokens against a 50% training
   capacity, so the router learned to predict its own top-k choice without future tokens.
2. **The trade is poor at this scale and budget.** Skipping half the tokens in 4 of 12 layers
   gave +19% throughput but cost 2.3% loss (3.7% under generation-time routing). Spending the
   same compute on ~19% more tokens for the control would be worth far less than 2.3%
   (control improved 2.1% over the whole 0.2B tokens), so MoD loses here.
3. **The causal gap (+1.4%) is router error**: the router's per-token guess disagrees with
   the top-k it was trained to imitate.
4. The capacity drop costs a large immediate loss jump (3.69 at the first eval vs 3.40) that
   0.2B tokens did not fully recover; the curve was still falling.
5. Both runs hit torch.compile's recompile limit (fixed afterwards), so absolute tok/s is
   below what the fixed code reaches; the ratio between the runs is still fair.

## Gentler settings (v2)

| run | routed layers | capacity | aux weight | val CE | Δ vs control | causal val CE | executed (causal) |
|---|---|---|---|---|---|---|---|
| v2b | 5, 7 | 0.5 | 0.05 | 3.3635 | +1.11% | 3.3969 (+2.11%) | 50.1% |
| v2a | 1, 5, 7, 10 | 0.75 | 0.05 | — | — | — | killed at 0.15B tokens (host out of memory) |

Routing only two layers halves the loss penalty, but it also skips only ~8% of layer compute;
the cost per unit of compute saved is no better. The causal-routing gap stays ~1%. Throughput
from this session is not comparable with the original control (that run hit the compile bug),
so a control rerun on the fixed code is queued together with v2a.

Options before trying MoD on MORPH-S: route fewer layers or use capacity 0.75, anneal more
slowly over more tokens, raise the aux weight to shrink the causal gap, and compare against a
compute-matched (not token-matched) control.
