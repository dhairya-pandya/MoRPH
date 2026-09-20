"""Multi-head Latent Attention (MLA) with cross-layer KV sharing (CLA).

MLA compresses K/V into a low-rank latent vector `c_kv` (dim = kv_lora_rank) plus a
single decoupled-RoPE key `k_rope`. Only these two small tensors are cached at
inference -> the KV cache shrinks ~an order of magnitude vs MHA.

CLA: a group of `cla_group_size` attention layers share ONE `SharedLatentKV` module,
so the compressed latent is computed and cached once per group (another ~Gx cut).

This is a from-scratch, training-time-correct implementation (dense attention, no
cache object). The cache path is a straightforward extension for generation.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_rope(seq_len: int, dim: int, theta: float, device, dtype):
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)              # (L, dim/2)
    cos = torch.cos(freqs).to(dtype)
    sin = torch.sin(freqs).to(dtype)
    return cos, sin                                # (L, dim/2)


def apply_rope(x, cos, sin):
    # x: (..., L, dim); rotate-half
    x1, x2 = x[..., ::2], x[..., 1::2]
    # broadcast cos/sin: (L, dim/2) -> (..., L, dim/2)
    while cos.dim() < x1.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    ro1 = x1 * cos - x2 * sin
    ro2 = x1 * sin + x2 * cos
    out = torch.stack((ro1, ro2), dim=-1).flatten(-2)
    return out


class SharedLatentKV(nn.Module):
    """Computes the shared compressed latent + decoupled RoPE key for a CLA group.

    Output tensors are exactly what an inference cache would store: `c_kv`
    (B, L, kv_lora_rank) and `k_rope` (B, L, qk_rope_head_dim) — both tiny.
    """

    def __init__(self, cfg):
        super().__init__()
        self.kv_a_proj = nn.Linear(cfg.d_model, cfg.kv_lora_rank, bias=False)
        self.kv_a_norm = nn.RMSNorm(cfg.kv_lora_rank, eps=cfg.norm_eps)
        self.k_rope_proj = nn.Linear(cfg.d_model, cfg.qk_rope_head_dim, bias=False)

    def forward(self, x):
        c_kv = self.kv_a_norm(self.kv_a_proj(x))   # (B, L, rank)
        k_rope = self.k_rope_proj(x)               # (B, L, rope_dim)  shared across heads
        return c_kv, k_rope


class MLAAttention(nn.Module):
    def __init__(self, cfg, shared: SharedLatentKV):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.nope = cfg.qk_nope_head_dim
        self.rope = cfg.qk_rope_head_dim
        self.v_head_dim = cfg.v_head_dim
        self.qk_head_dim = self.nope + self.rope
        self.shared = shared                       # shared across the CLA group

        self.norm = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        # query projection (per-head nope + rope)
        self.q_proj = nn.Linear(cfg.d_model, self.n_heads * self.qk_head_dim, bias=False)
        # up-project the shared latent to this layer's per-head k_nope and v
        self.kv_b_proj = nn.Linear(cfg.kv_lora_rank,
                                   self.n_heads * (self.nope + self.v_head_dim), bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.v_head_dim, cfg.d_model, bias=False)
        self.scale = 1.0 / math.sqrt(self.qk_head_dim)

    def forward(self, x, rope_cache=None, **kw):
        B, L, _ = x.shape
        h = self.norm(x)
        c_kv, k_rope = self.shared(h)              # (B,L,rank), (B,L,rope)

        q = self.q_proj(h).view(B, L, self.n_heads, self.qk_head_dim)
        q_nope, q_rope = q.split([self.nope, self.rope], dim=-1)   # (B,L,H,nope),(B,L,H,rope)

        kv = self.kv_b_proj(c_kv).view(B, L, self.n_heads, self.nope + self.v_head_dim)
        k_nope, v = kv.split([self.nope, self.v_head_dim], dim=-1)  # (B,L,H,nope),(B,L,H,vd)

        # RoPE on the decoupled parts. q_rope per-head; k_rope shared -> broadcast to heads.
        if rope_cache is not None:
            cos, sin = rope_cache
            cos, sin = cos[:L], sin[:L]
            # q_rope: (B,L,H,rope) -> move to (B,H,L,rope) for rope, then back
            q_rope = apply_rope(q_rope.transpose(1, 2), cos, sin).transpose(1, 2)
            k_rope = apply_rope(k_rope, cos, sin)                  # (B,L,rope)
        k_rope_h = k_rope.unsqueeze(2).expand(B, L, self.n_heads, self.rope)

        q_full = torch.cat([q_nope, q_rope], dim=-1)              # (B,L,H,qk)
        k_full = torch.cat([k_nope, k_rope_h], dim=-1)            # (B,L,H,qk)

        # (B,H,L,d)
        q_full = q_full.transpose(1, 2)
        k_full = k_full.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = F.scaled_dot_product_attention(q_full, k_full, v, is_causal=True)
        attn = attn.transpose(1, 2).reshape(B, L, self.n_heads * self.v_head_dim)
        return x + self.o_proj(attn)
