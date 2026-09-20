"""Mixture-of-Depths router (structural self-remodeling).

Per-token top-k routing: only `capacity` fraction of tokens flow through the wrapped
block; the rest skip it via the residual. At inference this lets the model reconfigure
its own effective depth per token — the "remodel itself into a more efficient system"
behavior. Wrapped around any block (SSM or attention).

Uses the token-choice / expert-choice style top-k selection from the MoD paper
(Raposo et al. 2024). Off during Stage 1 pretrain; enabled and annealed in Stage 2.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MoDWrapper(nn.Module):
    def __init__(self, cfg, block: nn.Module):
        super().__init__()
        self.block = block
        self.capacity = cfg.mod_capacity
        self.router = nn.Linear(cfg.d_model, 1, bias=False)
        self.last_exec_frac = None  # populated on forward, for logging/eval

    def forward(self, x, **kw):
        B, L, D = x.shape
        scores = self.router(x).squeeze(-1)           # (B, L)
        k = max(1, int(self.capacity * L))
        # top-k tokens per sequence get full compute
        topv, topi = torch.topk(scores, k, dim=1)     # (B, k)
        gate = torch.sigmoid(topv)                    # (B, k) straight-through weight

        # gather selected tokens
        idx = topi.unsqueeze(-1).expand(B, k, D)
        sel = torch.gather(x, 1, idx)                 # (B, k, D)
        out_sel = self.block(sel, **kw)               # block sees only chosen tokens
        # residual + gated update, scattered back
        delta = (out_sel - sel) * gate.unsqueeze(-1)
        out = x.clone()
        out.scatter_add_(1, idx, delta)
        self.last_exec_frac = k / L
        return out
