"""Multi-head Latent Attention (MLA) without positional encoding + true cross-layer sharing.

MLA caches a low-rank latent `c = RMSNorm(W_down h)` instead of per-head K/V; every layer
rebuilds its own K and V from `c` with a per-layer up-projection. In a hybrid the Mamba2
layers carry position/recency, so the attention layers use NoPE (Kimi Linear; Meta hybrid
study) — the latent is then the ONLY thing cached, and K stays linear in `c` (absorbable
into the query at inference).

CLA: within a group the first attention layer (producer) computes `c` from its own input;
later layers (consumers) reuse that same tensor. One cache entry per group per token.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentProducer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.down = nn.Linear(cfg.d_model, cfg.kv_lora_rank, bias=False)
        self.norm = nn.RMSNorm(cfg.kv_lora_rank, eps=cfg.norm_eps)

    def forward(self, h):
        return self.norm(self.down(h))      # (B, L, r)  <- the cached tensor


class MLAAttention(nn.Module):
    def __init__(self, cfg, is_producer: bool):
        super().__init__()
        self.n_heads, self.head_dim = cfg.n_heads, cfg.head_dim
        inner = cfg.n_heads * cfg.head_dim
        self.norm = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.latent = LatentProducer(cfg) if is_producer else None
        self.q_proj = nn.Linear(cfg.d_model, inner, bias=False)
        self.kv_up = nn.Linear(cfg.kv_lora_rank, 2 * inner, bias=False)
        self.kv_up.weight.muon_splits = [inner, inner]
        self.o_proj = nn.Linear(inner, cfg.d_model, bias=False)
        self.o_proj.weight.residual_out = True

    def forward(self, x, latent=None):
        B, L, _ = x.shape
        H, D = self.n_heads, self.head_dim
        h = self.norm(x)
        if self.latent is not None:
            latent = self.latent(h)
        elif latent is None:
            raise ValueError("consumer attention layer needs its group's latent")
        q = self.q_proj(h).view(B, L, H, D).transpose(1, 2)
        k, v = self.kv_up(latent).view(B, L, H, 2 * D).split(D, dim=-1)
        o = F.scaled_dot_product_attention(q, k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        return x + self.o_proj(o.transpose(1, 2).reshape(B, L, H * D)), latent
