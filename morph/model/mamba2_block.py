"""Mamba2 SSM block with a portable pure-PyTorch fallback.

On a CUDA box with `mamba-ssm` installed we use the fast fused kernel. Everywhere
else (Mac/CPU CI, kernel build failures) we fall back to a correct-but-slower
selective-scan written in plain PyTorch, so shape/grad tests and smoke runs work.

Both paths expose the same contract: (B, L, d_model) -> (B, L, d_model), and carry a
*constant-size* recurrent state at inference (no growing KV cache — the whole point).
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # fast path (GPU only)
    from mamba_ssm import Mamba2 as _FusedMamba2  # type: ignore
    _HAS_MAMBA = True
except Exception:  # pragma: no cover - depends on env
    _FusedMamba2 = None
    _HAS_MAMBA = False


class PyTorchSSM(nn.Module):
    """Minimal selective SSM (Mamba-style) in pure PyTorch.

    Sequential scan over time — O(L) memory-independent recurrence, correct on CPU.
    Not speed-competitive with the fused kernel; used only as a fallback.
    """

    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int, headdim: int):
        super().__init__()
        self.d_model = d_model
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.headdim = headdim
        self.nheads = self.d_inner // headdim
        assert self.d_inner % headdim == 0, "d_inner must be divisible by headdim"

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv,
            groups=self.d_inner, padding=d_conv - 1, bias=True,
        )
        # selective params: dt, B, C projected from x
        self.x_proj = nn.Linear(self.d_inner, self.nheads + 2 * d_state, bias=False)
        self.dt_bias = nn.Parameter(torch.zeros(self.nheads))
        # A is per-head, parameterized in log space and negated for stability
        self.A_log = nn.Parameter(torch.log(torch.arange(1, self.nheads + 1, dtype=torch.float32)))
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, D)
        B, L, _ = x.shape
        xz = self.in_proj(x)                      # (B, L, 2*d_inner)
        xin, z = xz.chunk(2, dim=-1)              # gate z
        # depthwise causal conv over time
        xc = self.conv1d(xin.transpose(1, 2))[..., :L].transpose(1, 2)
        xc = F.silu(xc)                           # (B, L, d_inner)

        params = self.x_proj(xc)                  # (B, L, nheads + 2*d_state)
        dt, Bp, Cp = torch.split(params, [self.nheads, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(dt + self.dt_bias)        # (B, L, nheads), > 0

        A = -torch.exp(self.A_log.float())        # (nheads,)  negative real
        # reshape into heads
        xc_h = xc.view(B, L, self.nheads, self.headdim)   # (B, L, H, P)

        # discretize: dA = exp(dt * A), per (B,L,H)
        dA = torch.exp(dt * A)                    # (B, L, H)
        # recurrent scan; state h: (B, H, P, N)
        h = xc.new_zeros(B, self.nheads, self.headdim, self.d_state)
        ys = []
        for t in range(L):
            dA_t = dA[:, t].unsqueeze(-1).unsqueeze(-1)          # (B,H,1,1)
            # x_t: (B,H,P); B_t: (B,N); outer product -> (B,H,P,N)
            x_t = xc_h[:, t]                                     # (B,H,P)
            B_t = Bp[:, t]                                       # (B,N)
            dt_t = dt[:, t]                                      # (B,H)
            inp = (dt_t.unsqueeze(-1).unsqueeze(-1)              # (B,H,1,1)
                   * x_t.unsqueeze(-1)                            # (B,H,P,1)
                   * B_t.unsqueeze(1).unsqueeze(1))               # (B,1,1,N)
            h = dA_t * h + inp                                    # (B,H,P,N)
            C_t = Cp[:, t]                                        # (B,N)
            y_t = torch.einsum("bhpn,bn->bhp", h, C_t)            # (B,H,P)
            y_t = y_t + self.D.view(1, -1, 1) * x_t
            ys.append(y_t)
        y = torch.stack(ys, dim=1).reshape(B, L, self.d_inner)   # (B,L,d_inner)
        y = y * F.silu(z)                                        # gate
        return self.out_proj(y)


class Mamba2Block(nn.Module):
    """Pre-norm residual SSM block; uses fused Mamba2 when available."""

    def __init__(self, cfg):
        super().__init__()
        self.norm = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        if _HAS_MAMBA:
            self.mixer = _FusedMamba2(
                d_model=cfg.d_model, d_state=cfg.ssm_d_state,
                d_conv=cfg.ssm_d_conv, expand=cfg.ssm_expand, headdim=cfg.ssm_headdim,
            )
            self.backend = "fused"
        else:
            self.mixer = PyTorchSSM(
                cfg.d_model, cfg.ssm_d_state, cfg.ssm_d_conv, cfg.ssm_expand, cfg.ssm_headdim,
            )
            self.backend = "pytorch"

    def forward(self, x, **kw):
        return x + self.mixer(self.norm(x))
