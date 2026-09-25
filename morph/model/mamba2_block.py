"""Mamba2 SSM mixer — same parameters as `mamba_ssm.modules.mamba2.Mamba2` (ngroups=1).

    in_proj  d -> [z (di), x (di), B (N), C (N), dt (H)]
    conv1d   depthwise causal (k = d_conv) over xBC, then SiLU
    dt       softplus(dt + dt_bias);  A = -exp(A_log)
    y        SSD(x, dt, A, B, C) + D * x
    norm     gated RMSNorm: rmsnorm(y * silu(z)) * weight
    out_proj di -> d

Matching names/shapes keeps checkpoints loadable by the fused mamba_ssm module. The scan
backend (pure PyTorch by default) is chosen by `cfg.ssm_backend`. A constant-size state
(heads x headdim x d_state) is all an SSM layer carries at inference: no KV cache.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ssd import ssd


class GatedRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, y, z):
        y = (y * F.silu(z)).float()
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + self.eps)
        return (y * self.weight.float()).to(z.dtype)


class Mamba2Mixer(nn.Module):
    def __init__(self, cfg, dt_min: float = 1e-3, dt_max: float = 0.1, dt_floor: float = 1e-4):
        super().__init__()
        d, di, N, H = cfg.d_model, cfg.ssm_d_inner, cfg.ssm_d_state, cfg.ssm_nheads
        self.d_inner, self.d_state, self.nheads, self.headdim = di, N, H, cfg.ssm_headdim
        self.chunk, self.backend = cfg.ssm_chunk, cfg.ssm_backend

        self.in_proj = nn.Linear(d, 2 * di + 2 * N + H, bias=False)
        self.in_proj.weight.muon_splits = [di, di, N, N, H]   # orthogonalize each part separately
        conv_dim = di + 2 * N
        self.conv1d = nn.Conv1d(conv_dim, conv_dim, cfg.ssm_d_conv, groups=conv_dim,
                                padding=cfg.ssm_d_conv - 1, bias=True)
        # Mamba2 inits: dt log-uniform in [dt_min, dt_max], A in U[1, 16], D = 1
        dt = torch.exp(torch.rand(H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = dt.clamp(min=dt_floor)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))   # inverse softplus
        self.A_log = nn.Parameter(torch.log(torch.empty(H).uniform_(1, 16)))
        self.D = nn.Parameter(torch.ones(H))
        self.norm = GatedRMSNorm(di, cfg.norm_eps)
        self.out_proj = nn.Linear(di, d, bias=False)
        self.out_proj.weight.residual_out = True

    def forward(self, u):
        b, l, _ = u.shape
        di, N, H, P = self.d_inner, self.d_state, self.nheads, self.headdim
        z, xBC, dt = torch.split(self.in_proj(u), [di, di + 2 * N, H], dim=-1)
        xBC = F.silu(self.conv1d(xBC.transpose(1, 2))[..., :l].transpose(1, 2))
        x, B, C = torch.split(xBC, [di, N, N], dim=-1)
        dt = F.softplus(dt.float() + self.dt_bias.float())
        A = -torch.exp(self.A_log.float())
        xh = x.reshape(b, l, H, P)
        y = ssd(xh, dt, A, B, C, chunk=self.chunk, backend=self.backend)
        y = y + xh * self.D.view(1, 1, H, 1).to(xh.dtype)
        y = self.norm(y.reshape(b, l, di), z)
        return self.out_proj(y)


class Mamba2Block(nn.Module):
    """Pre-norm residual SSM block."""

    def __init__(self, cfg):
        super().__init__()
        self.norm = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.mixer = Mamba2Mixer(cfg)

    def forward(self, x, **kw):
        return x + self.mixer(self.norm(x))
