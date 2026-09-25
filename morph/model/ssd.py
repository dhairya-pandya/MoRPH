"""Mamba2 state-space-duality (SSD) scan.

`ssd_chunked` is the chunked SSD algorithm from the Mamba2 paper ("ssd_minimal") written
in plain PyTorch: intra-chunk work is dense matmuls, only a short recurrence runs across
chunks. No custom CUDA, so it runs on T4 / CPU and under torch.compile. Decays, cumsums
and the inter-chunk state recurrence are computed in fp32 (autocast disabled).

Shapes (ngroups = 1):
    x  (b, l, h, p)   inputs per head
    dt (b, l, h)      step sizes after softplus (> 0)
    A  (h,)           negative decay rates
    B  (b, l, n)      input projection (shared by heads)
    C  (b, l, n)      output projection (shared by heads)
Returns y (b, l, h, p) (without the D skip term) and the final state (b, h, p, n).

Optional fast paths (`mamba_ssm`, `fla`) compute the same function; the GPU gate checks
parity before they are used.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def segsum(x: torch.Tensor) -> torch.Tensor:
    """(..., T) -> (..., T, T) with out[..., i, j] = sum(x[..., j+1:i+1]) for i >= j, else -inf."""
    T = x.size(-1)
    x = x.unsqueeze(-1).expand(*x.shape, T)                   # [..., i, j] = x[..., i]
    below = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=-1)
    x = x.masked_fill(~below, 0.0)
    out = torch.cumsum(x, dim=-2)
    keep = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=0)
    return out.masked_fill(~keep, float("-inf"))


def ssd_chunked(x, dt, A, B, C, chunk: int = 64):
    b, l, h, p = x.shape
    n = B.shape[-1]
    out_dtype = x.dtype
    with torch.autocast(device_type=x.device.type, enabled=False):
        x, dt, B, C = x.float(), dt.float(), B.float(), C.float()
        pad = (-l) % chunk
        if pad:  # padded steps have dt = 0 -> no decay, no input: they change nothing
            x = F.pad(x, (0, 0, 0, 0, 0, pad))
            dt = F.pad(dt, (0, 0, 0, pad))
            B = F.pad(B, (0, 0, 0, pad))
            C = F.pad(C, (0, 0, 0, pad))
        L = l + pad
        c = L // chunk

        xd = (x * dt.unsqueeze(-1)).view(b, c, chunk, h, p)              # discretized input
        a = (dt * A.float().view(1, 1, h)).view(b, c, chunk, h).permute(0, 3, 1, 2)  # (b,h,c,l)
        Bc = B.view(b, c, chunk, n)
        Cc = C.view(b, c, chunk, n)
        a_cum = torch.cumsum(a, dim=-1)                                  # (b,h,c,l)

        # 1. intra-chunk (diagonal blocks)
        decay = torch.exp(segsum(a))                                     # (b,h,c,l,s)
        CB = torch.einsum("bcln,bcsn->bcls", Cc, Bc)                     # (b,c,l,s)
        scores = decay * CB.unsqueeze(1)                                 # (b,h,c,l,s)
        y_diag = torch.einsum("bhcls,bcshp->bclhp", scores, xd)

        # 2. state at the end of every chunk
        decay_to_end = torch.exp(a_cum[..., -1:] - a_cum)                # (b,h,c,l)
        xw = xd * decay_to_end.permute(0, 2, 3, 1).unsqueeze(-1)         # (b,c,l,h,p)
        states = torch.einsum("bcln,bclhp->bchpn", Bc, xw)               # (b,c,h,p,n)

        # 3. recurrence across chunks
        states = torch.cat([torch.zeros_like(states[:, :1]), states], dim=1)
        chunk_decay = torch.exp(segsum(F.pad(a_cum[..., -1], (1, 0))))   # (b,h,c+1,c+1)
        new_states = torch.einsum("bhzc,bchpn->bzhpn", chunk_decay, states)
        states, final_state = new_states[:, :-1], new_states[:, -1]

        # 4. contribution of the carried-in state to each position
        y_off = torch.einsum("bcln,bchpn->bclhp", Cc, states)
        y_off = y_off * torch.exp(a_cum).permute(0, 2, 3, 1).unsqueeze(-1)

        y = (y_diag + y_off).reshape(b, L, h, p)[:, :l]
    return y.to(out_dtype), final_state


def ssd_reference(x, dt, A, B, C):
    """Step-by-step recurrence (tests only): h_t = exp(dt_t A) h_{t-1} + dt_t x_t B_t^T, y_t = h_t C_t."""
    b, l, h, p = x.shape
    n = B.shape[-1]
    x, dt, B, C = x.float(), dt.float(), B.float(), C.float()
    state = x.new_zeros(b, h, p, n)
    ys = []
    for t in range(l):
        dA = torch.exp(dt[:, t] * A.float())                             # (b,h)
        inp = dt[:, t, :, None, None] * x[:, t, :, :, None] * B[:, t, None, None, :]
        state = state * dA[:, :, None, None] + inp
        ys.append(torch.einsum("bhpn,bn->bhp", state, C[:, t]))
    return torch.stack(ys, dim=1), state


def ssd(x, dt, A, B, C, chunk: int = 64, backend: str = "torch"):
    """Dispatch to the requested backend; returns y (b,l,h,p)."""
    if backend == "torch" or x.device.type != "cuda":
        return ssd_chunked(x, dt, A, B, C, chunk)[0]
    if backend == "mamba_ssm":
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined  # type: ignore
        return mamba_chunk_scan_combined(x, dt, A, B.unsqueeze(2), C.unsqueeze(2), chunk_size=chunk)
    if backend == "fla":
        from fla.ops.simple_gla import chunk_simple_gla  # type: ignore
        h = x.shape[2]
        q = C.unsqueeze(2).expand(-1, -1, h, -1).contiguous()
        k = B.unsqueeze(2).expand(-1, -1, h, -1).contiguous()
        v = (x * dt.unsqueeze(-1)).contiguous()
        g = (dt * A.view(1, 1, -1)).float().contiguous()
        y, _ = chunk_simple_gla(q, k, v, g, scale=1.0)
        return y.to(x.dtype)
    raise ValueError(f"unknown ssd backend {backend!r}")
