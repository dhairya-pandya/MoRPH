"""Self-modeling auxiliary head (Premakumar et al. 2024).

An auxiliary task in which the network predicts a subset of its OWN hidden
activations. Training on LM + lambda * self-prediction is reported to induce
*emergent simplification*: the internal representation becomes simpler, lower
effective rank, and more robust.

In MORPH this is load-bearing: simpler / lower-rank internals compress better,
which is exactly what MLA's low-rank latent KV and the MoD router want. The head
predicts the (detached) activation of a chosen layer from an *earlier* layer's
state, so the network is pressured to make its own future state predictable.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfModelHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.enabled = cfg.self_model_enabled
        self.target_units = min(cfg.self_model_dim, cfg.d_model)
        # small MLP predictor: current state -> predicted target-layer state (subset of units)
        hidden = max(self.target_units, cfg.d_model // 2)
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, self.target_units, bias=False),
        )

    def forward(self, source_state: torch.Tensor, target_state: torch.Tensor) -> torch.Tensor:
        """Return the self-modeling MSE loss (scalar). Target is detached."""
        pred = self.net(source_state)                       # (B, L, target_units)
        target = target_state[..., : self.target_units].detach()
        return F.mse_loss(pred, target)
