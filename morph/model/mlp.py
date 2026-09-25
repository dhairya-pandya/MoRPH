"""SwiGLU MLP block (pre-norm residual), used after every mixer."""
from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.w_gate = nn.Linear(cfg.d_model, cfg.mlp_hidden, bias=False)
        self.w_up = nn.Linear(cfg.d_model, cfg.mlp_hidden, bias=False)
        self.w_down = nn.Linear(cfg.mlp_hidden, cfg.d_model, bias=False)
        self.w_down.weight.residual_out = True

    def forward(self, x, **kw):
        h = self.norm(x)
        return x + self.w_down(F.silu(self.w_gate(h)) * self.w_up(h))
