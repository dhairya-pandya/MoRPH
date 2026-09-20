"""Transformer-Squared / SVF — weight-space self-adaptation (Stage 3).

Decompose a linear weight W = U diag(s) V^T (SVD). An "expert" is a learned vector
z that rescales the singular values: W' = U diag(s * softplus(z)) V^T. At inference a
lightweight dispatch pass picks/mixes experts per prompt, re-tuning the model's own
weights with no hot-path parameter growth. RL-trained per the paper; supervised
fitting is the fallback if RL is unstable at 300M.

Stage-3 scaffold: the SVFLinear wrapper and a no-op expert are provided; expert
training (RL/supervised) lands in morph/train/svf_train.py.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SVFLinear(nn.Module):
    """Wrap a trained nn.Linear; adapt only its singular values via expert vector z."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        W = linear.weight.data.float()             # (out, in)
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        self.register_buffer("U", U)               # (out, r)
        self.register_buffer("S", S)               # (r,)
        self.register_buffer("Vh", Vh)             # (r, in)
        self.bias = linear.bias
        self.z = nn.Parameter(torch.zeros_like(S)) # expert vector (per-prompt at inference)

    def effective_weight(self) -> torch.Tensor:
        s = self.S * F.softplus(self.z)
        return (self.U * s.unsqueeze(0)) @ self.Vh

    def forward(self, x):
        return F.linear(x, self.effective_weight().to(x.dtype), self.bias)
