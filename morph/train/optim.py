"""Optimizers: Muon for hidden weight matrices, AdamW for everything else.

Muon (Jordan et al. 2024; Moonlight 2502.16982) orthogonalizes the Nesterov-momentum
update with a quintic Newton-Schulz iteration and rescales it by 0.2*sqrt(max(m, n)) so its
RMS matches AdamW's — one LR serves both optimizers. Fused projections tagged with a
`muon_splits` attribute (e.g. Mamba2 in_proj = z|x|B|C|dt) are orthogonalized per block.
"""
from __future__ import annotations

from typing import List

import torch


def newton_schulz(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * (A @ A)) @ X
    return X.T if transposed else X


def _orth_update(g: torch.Tensor, splits, steps: int) -> torch.Tensor:
    parts = g.split(splits, dim=0) if splits else (g,)
    outs = [newton_schulz(p, steps) * (0.2 * max(p.shape) ** 0.5) for p in parts]
    return torch.cat(outs, dim=0) if len(outs) > 1 else outs[0]


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=2e-3, momentum=0.95, nesterov=True, weight_decay=0.0, ns_steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                      weight_decay=weight_decay, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.float()
                st = self.state[p]
                if "momentum_buffer" not in st:
                    st["momentum_buffer"] = torch.zeros_like(g)
                buf = st["momentum_buffer"]
                buf.mul_(group["momentum"]).add_(g)
                g = g.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf
                u = _orth_update(g, getattr(p, "muon_splits", None), group["ns_steps"])
                p.mul_(1.0 - group["lr"] * group["weight_decay"])
                p.add_(u.to(p.dtype), alpha=-group["lr"])


def split_params(model):
    """(muon matrices, adamw params) — every trainable parameter exactly once."""
    embed = model.model.embed.weight
    muon, adamw = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and p is not embed:
            muon.append(p)
        else:
            adamw.append(p)
    return muon, adamw


def build_optimizers(model, tcfg) -> List[torch.optim.Optimizer]:
    betas = tuple(tcfg.adam_betas)
    muon, rest = split_params(model)
    if tcfg.optimizer == "muon":
        return [
            Muon(muon, lr=tcfg.lr, momentum=tcfg.muon_momentum, weight_decay=tcfg.weight_decay),
            torch.optim.AdamW(rest, lr=tcfg.lr, betas=betas, eps=1e-8, weight_decay=0.0),
        ]
    if tcfg.optimizer == "adamw":
        return [torch.optim.AdamW([
            {"params": muon, "weight_decay": tcfg.weight_decay},
            {"params": rest, "weight_decay": 0.0},
        ], lr=tcfg.lr, betas=betas, eps=1e-8)]
    raise ValueError(f"unknown optimizer {tcfg.optimizer!r}")


def set_lr(optimizers, lr: float):
    for opt in optimizers:
        for g in opt.param_groups:
            g["lr"] = lr
