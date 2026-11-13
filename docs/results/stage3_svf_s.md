# Stage 3: SVF experts and two-pass adaptation on MORPH-S

Base: the finished main run (MORPH-S, 170M, 3B tokens), frozen. Same recipe as the
[XS run](stage3_svf_xs.md): every block linear (126 matrices) is SVD-decomposed and an expert
is one singular-value scale vector per matrix, **77.6k trainable numbers (0.05% of the model)**.

Experts: 300 steps × 16 × 1024 tokens on ~8M tokens per domain, AdamW lr 2e-3, micro-batch 4,
about 26–28 minutes each on one T4 (math, code: open-web-math, codeparrot-clean; stories:
TinyStories). Router: linear classifier on the mean-pooled final hidden state of the first 128
tokens, FineWeb-Edu as "general" (identity expert). Evaluation: 64 held-out 1024-token windows
per domain, scored after the 128-token prompt. One Kaggle T4×2 session.

| domain | base | math expert | code expert | stories expert | two-pass | Δ two-pass vs base | routing acc |
|---|---|---|---|---|---|---|---|
| general | 2.813 | 2.846 | 2.920 | 2.988 | 2.817 | +0.1% | 91% |
| math | 3.206 | **2.990** | 3.135 | 3.309 | 3.005 | −6.3% | 84% |
| code | 3.353 | 3.160 | **2.761** | 3.556 | 2.765 | −17.5% | 98% |
| stories | 2.435 | 2.315 | 2.411 | **1.955** | 1.955 | −19.7% | 100% |

Averaged over the four domains: base 2.952 → two-pass 2.635 (**−10.7%**; XS: −8.1%).

Findings:
1. **The method scales up.** Domain gains are larger at 170M than at 44M (math −6.3% vs
   −3.9%, code −17.5% vs −12.9%, stories −19.7% vs −15.6%) with the same steps and data. A likely
   reason: a better-trained base has more in its singular directions for a rescaling to select.
2. **Two-pass routing is nearly free.** It lands within 0.6% of the oracle (own expert) on
   every domain and costs general text only +0.1%, while a single always-on expert costs
   general text +1.1% (math) to +6.2% (stories).
3. **Math is the hardest domain to route** (84%), and it is also where two-pass trails the oracle most (3.005 vs 2.990).

Experts and router: `svf/` in [dhairyapandya/morph-s](https://huggingface.co/dhairyapandya/morph-s).
Reproduce: `python scripts/launch_kaggle.py exp svf-s`.
