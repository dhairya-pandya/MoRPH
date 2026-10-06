"""Mixture-of-Depths routing (Raposo et al. 2024, arXiv 2404.02258) — Stage 2 structural self-remodel.

A router scores every token; only routed tokens run the wrapped layer (Mamba2 mixer + MLP),
the rest pass through on the residual. The model thereby chooses its own depth per token.

Training ("topk"): per sequence the top-k tokens (k = capacity * L) are gathered IN ORIGINAL
ORDER, run through the layer as a shorter sequence, and their update is scaled by
2*sigmoid(router logit) (so a zero-initialized router leaves the pretrained layer unchanged
and the router still gets a gradient) before being scattered back. Static shapes, fast.

Top-k over a sequence looks at future tokens, so it cannot be used for generation. An
auxiliary BCE loss trains the router logit to predict top-k membership; at inference
("causal") a token runs the layer iff its logit > 0 — a per-token decision that uses no
future information. For an SSM, a skipped token simply does not update the recurrent/conv
state, exactly as in the gathered training sequence.
"""
from __future__ import annotations

from typing import Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoDRouter(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, 1, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.proj.weight.no_muon = True      # 1 x d vector: plain AdamW
        self.proj.weight.zero_init = True    # keep zero init: scale 2*sigmoid(0) = 1 leaves the layer unchanged

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).squeeze(-1).float()                     # (B, L)


def route_topk(x: torch.Tensor, logits: torch.Tensor, capacity: float,
               inner: Callable[[torch.Tensor], torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Returns (output, aux BCE loss, executed fraction). inner maps (B, k, D) -> (B, k, D) residual output."""
    B, L, D = x.shape
    k = max(1, min(L, int(round(capacity * L))))
    idx = logits.topk(k, dim=1).indices.sort(dim=1).values          # keep temporal order
    target = torch.zeros_like(logits).scatter_(1, idx, 1.0)
    aux = F.binary_cross_entropy_with_logits(logits, target)
    gidx = idx.unsqueeze(-1).expand(B, k, D)
    xs = torch.gather(x, 1, gidx)
    scale = 2.0 * torch.sigmoid(torch.gather(logits, 1, idx)).unsqueeze(-1)
    delta = (inner(xs) - xs) * scale.to(xs.dtype)
    return x.scatter_add(1, gidx, delta.to(x.dtype)), aux, k / L


def route_causal(x: torch.Tensor, logits: torch.Tensor,
                 inner: Callable[[torch.Tensor], torch.Tensor]) -> Tuple[torch.Tensor, float]:
    """Inference routing: token t runs the layer iff logits[t] > 0. Returns (output, executed fraction)."""
    out = x.clone()
    mask = logits > 0
    for b in range(x.size(0)):
        sel = mask[b].nonzero(as_tuple=True)[0]
        if sel.numel() == 0:
            continue
        xs = x[b, sel].unsqueeze(0)
        scale = 2.0 * torch.sigmoid(logits[b, sel]).view(1, -1, 1)
        out[b, sel] = x[b, sel] + ((inner(xs) - xs) * scale.to(xs.dtype)).squeeze(0).to(x.dtype)
    return out, float(mask.float().mean())
