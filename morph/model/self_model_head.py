"""Self-modeling auxiliary head (Premakumar et al. 2024, arXiv 2407.10188).

Paper recipe: augment the OUTPUT layer with extra linear units that predict the activations
of chosen internal layers; loss = task + (w_s / n) * ||a_hat - a||^2, with gradients flowing
into BOTH the prediction and the target ("learning to self-model is learning to make oneself
modelable"). The reported effect is self-regularization: lower RLCT, narrower weights.

LM adaptation used here:
- source: the final normalized hidden state (the LM head's input), as in the paper;
- targets: residual states entering the attention layers — the states compressed into the
  cached MLA latent;
- targets are RMS-normalized (parameter-free). In a pre-norm residual net a global rescale
  of the residual stream is functionally free, so a raw MSE could be lowered by shrinking
  activations without simplifying anything; normalizing leaves only structure to predict;
- `detach=True` is the ablation control that removes the "make oneself modelable" path.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn


def rms_normalize(t: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    t = t.float()
    return t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + eps)


class SelfModelHead(nn.Module):
    def __init__(self, cfg, n_targets: int):
        super().__init__()
        self.n, self.d = n_targets, cfg.d_model
        self.detach = cfg.self_model_detach
        self.proj = nn.Linear(cfg.d_model, n_targets * cfg.d_model, bias=True)
        self.proj.weight.muon_splits = [cfg.d_model] * n_targets

    def forward(self, source: torch.Tensor, targets: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (mean loss over targets, per-target losses detached)."""
        B, L, _ = source.shape
        pred = self.proj(source).float().view(B, L, self.n, self.d)
        losses = []
        for k, t in enumerate(targets):
            a = rms_normalize(t.detach() if self.detach else t)
            losses.append((pred[:, :, k] - a).pow(2).mean())
        per = torch.stack(losses)
        return per.mean(), per.detach()
