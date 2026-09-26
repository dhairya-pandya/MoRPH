# Proxy ablation: self-modeling in a hybrid LM

Setup: MORPH-XS (44.0M params: 10 Mamba2 + 2 NoPE-MLA layers sharing one latent, d=384),
FineWeb-Edu, SmolLM2 tokenizer, 0.5B tokens (3,815 steps × 131k tokens), Muon+AdamW, WSD,
fp16 on one Kaggle T4 per run (~8–9 h, 15.7–17.2k tok/s), one seed per run.
Self-model targets: RMS-normalized residual states entering the attention layers (L3, L8),
predicted by linear extra outputs on the final hidden state (Premakumar et al. 2024).

Metrics on a fixed 64k-token validation batch:
- **erank** — effective rank (Roy–Vetterli) of the target states (lower = simpler);
- **energy64** — fraction of variance in the top-64 principal directions, 64 = the cached
  latent size (higher = a rank-64 latent can hold more of the state);
- **sm_r2** — R² of the self-model head (R0 uses a detached readout probe).

| run | setting | val CE | Δ vs R0 | erank L3 / L8 | energy64 L3 / L8 | sm_r2 L3 / L8 |
|---|---|---|---|---|---|---|
| R0 | baseline (probe) | 3.3689 | — | 328.3 / 338.3 | 0.612 / 0.574 | 0.48 / 0.69 |
| R1 | λ=0.1, targets get gradient | 3.3675 | −0.04% | 322.9 / 329.9 | 0.632 / 0.612 | 0.57 / 0.79 |
| R2 | λ=1.0, targets get gradient | 3.3750 | +0.18% | 300.5 / 319.0 | 0.691 / 0.628 | 0.68 / 0.86 |
| R3 | λ=1.0, targets detached | 3.4271 | +1.73% | 334.4 / 346.2 | 0.542 / 0.526 | 0.86 / 0.93 |

Findings:
1. **Dose-dependent simplification, as in the paper.** Self-modeling lowers the effective rank
   of the states that feed the cache and concentrates their variance (energy64 +13% at L3 for
   λ=1.0) at a +0.18% validation-loss cost (within the 0.5% budget) — λ* = 1.0.
2. **The mechanism is the gradient into the targets.** Detaching the targets (R3) removes the
   simplification (rank rises above baseline) and costs 1.7% loss: the network then has to
   copy its internals into the final state instead of making them simpler.
3. **Weight width did not narrow** (all-block std 0.1026 → 0.1055), unlike the paper's
   classifiers; Muon's normalized updates are a likely confound for this measure.
4. Single seed: the R0/R1 loss gap is noise; R2/R3 gaps and the rank/energy trends are larger
   and monotone in λ.

Pending: R4/R5 (latent 32 with λ=0 vs λ*=1.0) — does self-modeling protect a halved cache?
