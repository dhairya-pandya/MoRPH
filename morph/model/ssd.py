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


def _decay(a_cum: torch.Tensor) -> torch.Tensor:
    """exp(a_cum[..., i] - a_cum[..., j]) for i >= j else 0, shape (..., T, T). fp32."""
    T = a_cum.size(-1)
    diff = a_cum.unsqueeze(-1) - a_cum.unsqueeze(-2)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=a_cum.device))
    return torch.exp(diff.masked_fill(~mask, float("-inf")))


def ssd_chunked(x, dt, A, B, C, chunk: int = 64, mm_dtype=None):
    """mm_dtype: dtype of the large matmuls (default: fp16/bf16 inputs keep their dtype, else fp32).
    Exponentials, cumsums and the inter-chunk state recurrence always run in fp32."""
    b, l, h, p = x.shape
    n = B.shape[-1]
    out_dtype = x.dtype
    if mm_dtype is None:
        mm_dtype = x.dtype if x.dtype in (torch.float16, torch.bfloat16) else torch.float32
    with torch.autocast(device_type=x.device.type, enabled=False):
        dt = dt.float()
        pad = (-l) % chunk
        if pad:  # padded steps have dt = 0 -> no decay, no input: they change nothing
            x = F.pad(x, (0, 0, 0, 0, 0, pad))
            dt = F.pad(dt, (0, 0, 0, pad))
            B = F.pad(B, (0, 0, 0, pad))
            C = F.pad(C, (0, 0, 0, pad))
        L = l + pad
        c = L // chunk

        xd = (x.float() * dt.unsqueeze(-1)).to(mm_dtype).view(b, c, chunk, h, p)   # discretized input
        a = (dt * A.float().view(1, 1, h)).view(b, c, chunk, h).permute(0, 3, 1, 2)  # (b,h,c,l) log-decay
        Bc = B.to(mm_dtype).view(b, c, chunk, n)
        Cc = C.to(mm_dtype).view(b, c, chunk, n)
        a_cum = torch.cumsum(a, dim=-1)                                  # (b,h,c,l) fp32

        # 1. intra-chunk (diagonal blocks)
        CB = torch.einsum("bcln,bcsn->bcls", Cc, Bc).float()             # (b,c,l,s)
        scores = (_decay(a_cum) * CB.unsqueeze(1)).to(mm_dtype)          # (b,h,c,l,s)
        y_diag = torch.einsum("bhcls,bcshp->bclhp", scores, xd)

        # 2. state at the end of every chunk
        decay_to_end = torch.exp(a_cum[..., -1:] - a_cum)                # (b,h,c,l)
        xw = xd * decay_to_end.permute(0, 2, 3, 1).unsqueeze(-1).to(mm_dtype)
        states = torch.einsum("bcln,bclhp->bchpn", Bc, xw).float()       # (b,c,h,p,n)

        # 3. recurrence across chunks (fp32)
        states = torch.cat([torch.zeros_like(states[:, :1]), states], dim=1)
        chunk_decay = _decay(F.pad(a_cum[..., -1], (1, 0)).cumsum(-1))   # (b,h,c+1,c+1)
        new_states = torch.einsum("bhzc,bchpn->bzhpn", chunk_decay, states)
        states, final_state = new_states[:, :-1], new_states[:, -1]

        # 4. contribution of the carried-in state to each position
        y_off = torch.einsum("bcln,bchpn->bclhp", Cc, states.to(mm_dtype)).float()
        y_off = y_off * torch.exp(a_cum).permute(0, 2, 3, 1).unsqueeze(-1)

        y = (y_diag.float() + y_off).reshape(b, L, h, p)[:, :l]
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
