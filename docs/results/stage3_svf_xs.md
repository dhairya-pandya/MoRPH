# Stage 3: SVF experts and two-pass adaptation on the XS proxy

Base: proxy R2 (MORPH-XS, 44M, λ=1.0, 0.5B FineWeb-Edu tokens), frozen. Every block linear
(63 matrices) is SVD-decomposed; an expert is one singular-value scale vector per matrix,
**23.2k trainable numbers (0.05% of the model)**.

Experts: supervised next-token training on ~8M tokens per domain, 300 steps × 16 × 1024 tokens,
AdamW lr 2e-3, about 7 minutes each on one T4.
- math: open-web-math
- code: codeparrot-clean
- stories: TinyStories

Router: linear classifier on the mean-pooled final hidden state of the first 128 tokens (base
model), trained on 512 prompts per domain plus FineWeb-Edu as "general" (identity expert).
Evaluation: 64 held-out windows of 1024 tokens per domain. All methods are scored on the tokens
after the 128-token prompt.

| domain | base | math expert | code expert | stories expert | own expert | two-pass | Δ two-pass vs base | routing acc |
|---|---|---|---|---|---|---|---|---|
| general | 3.326 | 3.366 | 3.484 | 3.536 | 3.326 (identity) | 3.332 | +0.2% | 89% |
| math | 3.736 | **3.579** | 3.729 | 3.907 | 3.579 | 3.591 | −3.9% | 84% |
| code | 4.276 | 4.049 | **3.717** | 4.648 | 3.717 | 3.725 | −12.9% | 95% |
| stories | 2.976 | 2.910 | 3.040 | **2.512** | 2.512 | 2.512 | −15.6% | 100% |

Averaged over the four domains: base 3.579 → two-pass 3.290 (−8.1%).

Findings:
1. **Tiny experts carry real domain skill.** Rescaling singular values alone cuts loss 4–16% on
   the target domain; the diagonal of the expert matrix is the best entry in every row.
2. **Experts are specialists and hurt elsewhere.** Each expert raises general-domain loss by
   1.2–6.3%, so a single always-on expert would be a bad trade.
3. **Two-pass routing keeps the gains and avoids the damage.** The router picks the right
   domain 84–100% of the time, the mixed model lands within 0.3% of the oracle expert on each
   specialist domain, and on general text it stays within 0.2% of the base model by falling
   back to the identity expert.
4. Math is the hardest to separate and to improve: open-web-math overlaps FineWeb-Edu, which
   the base model was trained on.

Next: repeat on MORPH-S once the main run has a converged checkpoint, and try larger / more
domains and an RL objective once the model follows instructions.
