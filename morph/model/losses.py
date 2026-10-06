"""Chunked cross-entropy + z-loss without materializing full-vocab fp32 logits.

Tokens are processed in chunks; each chunk's logits are recomputed in backward
(activation checkpointing), so peak memory is one chunk of logits instead of
batch x seq x vocab. Targets equal to -100 are ignored.
"""
from __future__ import annotations

import os

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


def _chunk_grad(logits: torch.Tensor, y: torch.Tensor, valid: torch.Tensor, z_coef: float):
    """Per-chunk CE/z-loss sums and the O(1)-scaled logit gradient (not yet divided by n)."""
    lf = logits.float()
    lse = torch.logsumexp(lf, dim=-1)
    tgt = lf.gather(-1, y.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    vf = valid.float()
    nll = ((lse - tgt) * vf).sum()
    zsq = (lse * lse * vf).sum()
    p = torch.exp(lf - lse.unsqueeze(-1))
    g = p * ((1.0 + 2.0 * z_coef * lse) * vf).unsqueeze(-1)
    g = g.scatter_add(-1, y.clamp_min(0).unsqueeze(-1), -vf.unsqueeze(-1))
    return g.to(logits.dtype), nll, zsq


_COMPILED = {}


def _chunk_grad_fn(device_type: str):
    """torch.compile'd chunk math on CUDA (fuses the softmax passes); eager fallback on failure/CPU."""
    if device_type != "cuda" or os.environ.get("MORPH_COMPILE_CE", "1") == "0":
        return _chunk_grad
    if "fn" not in _COMPILED:
        try:
            _COMPILED["fn"] = torch.compile(_chunk_grad, dynamic=False)
        except Exception as e:  # pragma: no cover - env dependent
            print(f"[losses] compile unavailable ({e}); eager CE", flush=True)
            _COMPILED["fn"] = _chunk_grad
    return _COMPILED["fn"]


class _FusedLinearCE(torch.autograd.Function):
    """loss = mean CE + z * mean lse^2, gradients computed in forward chunk by chunk.

    Never stores logits; backward just rescales the stored input/weight gradients.
    The logit gradient is kept O(1) (division by the token count happens after the
    matmuls, in fp32) so fp16 does not underflow it."""

    @staticmethod
    def forward(ctx, h, weight, y, chunk: int, z_coef: float, compute_dtype):
        valid = y != -100
        n = valid.sum().clamp_min(1).float()
        W = weight.to(compute_dtype)
        grad_h = torch.empty(h.shape, dtype=torch.float32, device=h.device)
        grad_w = torch.zeros(weight.shape, dtype=torch.float32, device=h.device)
        nll = torch.zeros((), dtype=torch.float32, device=h.device)
        zsq = torch.zeros((), dtype=torch.float32, device=h.device)
        fn = _chunk_grad_fn(h.device.type)
        for s in range(0, h.size(0), chunk):
            hc = h[s:s + chunk].to(compute_dtype)
            g, a, b = fn(hc @ W.T, y[s:s + chunk], valid[s:s + chunk], z_coef)
            nll += a
            zsq += b
            grad_h[s:s + chunk] = (g @ W).float()
            grad_w += (g.T @ hc).float()
        ctx.save_for_backward(grad_h / n, grad_w / n)
        ctx.h_dtype, ctx.w_dtype = h.dtype, weight.dtype
        ce, zm = nll / n, zsq / n
        ctx.mark_non_differentiable(ce, zm)
        return ce + z_coef * zm, ce, zm

    @staticmethod
    def backward(ctx, go, _gce, _gz):
        grad_h, grad_w = ctx.saved_tensors
        return (grad_h * go).to(ctx.h_dtype), (grad_w * go).to(ctx.w_dtype), None, None, None, None


def fused_linear_cross_entropy(hidden, weight, targets, chunk: int = 1024, z_coef: float = 0.0):
    """Returns (loss = CE + z * lse^2, CE, mean lse^2) with a memory- and bandwidth-lean backward."""
    dev = hidden.device.type
    dtype = torch.get_autocast_dtype(dev) if torch.is_autocast_enabled(dev) else hidden.dtype
    with torch.autocast(device_type=dev, enabled=False):
        return _FusedLinearCE.apply(hidden.reshape(-1, hidden.size(-1)), weight, targets.reshape(-1),
                                    chunk, z_coef, dtype)
