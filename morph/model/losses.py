"""Chunked cross-entropy + z-loss without materializing full-vocab fp32 logits.

Tokens are processed in chunks; each chunk's logits are recomputed in backward
(activation checkpointing), so peak memory is one chunk of logits instead of
batch x seq x vocab. Targets equal to -100 are ignored.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _chunk_terms(h, weight, y):
    logits = F.linear(h, weight).float()
    lse = torch.logsumexp(logits, dim=-1)
    valid = y != -100
    tgt = logits.gather(-1, y.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    nll = ((lse - tgt) * valid).sum()
    zsq = ((lse * lse) * valid).sum()
    return nll, zsq


def chunked_cross_entropy(hidden, weight, targets, chunk: int = 1024):
    """Returns (mean CE, mean logsumexp^2) over non-ignored targets."""
    h = hidden.reshape(-1, hidden.size(-1))
    y = targets.reshape(-1)
    nll = hidden.new_zeros((), dtype=torch.float32)
    zsq = hidden.new_zeros((), dtype=torch.float32)
    use_ckpt = torch.is_grad_enabled()
    for s in range(0, h.size(0), chunk):
        args = (h[s:s + chunk], weight, y[s:s + chunk])
        a, b = checkpoint(_chunk_terms, *args, use_reentrant=False) if use_ckpt else _chunk_terms(*args)
        nll = nll + a
        zsq = zsq + b
    n = (y != -100).sum().clamp_min(1)
    return nll / n, zsq / n
