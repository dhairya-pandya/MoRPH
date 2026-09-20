"""SwiGLU MLP block (pre-norm residual), shared by both SSM and attention stacks."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden = int(cfg.mlp_ratio * cfg.d_model)
        self.norm = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.w_gate = nn.Linear(cfg.d_model, hidden, bias=cfg.mlp_bias)
        self.w_up = nn.Linear(cfg.d_model, hidden, bias=cfg.mlp_bias)
        self.w_down = nn.Linear(hidden, cfg.d_model, bias=cfg.mlp_bias)

    def forward(self, x, **kw):
        h = self.norm(x)
        return x + self.w_down(F.silu(self.w_gate(h)) * self.w_up(h))
