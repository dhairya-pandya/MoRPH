"""MORPH backbone: SSM-majority hybrid + LM head + self-modeling head.

Layer i = mixer (Mamba2, or NoPE-MLA at cfg.attn_layers) + SwiGLU MLP, both pre-norm
residual. MLA producers hand their latent to the consumers of the same CLA group.
Optional per-layer activation checkpointing (`grad_ckpt`). The loss is the chunked CE +
z-loss + lambda * self-modeling loss (see self_model_head.py).
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional
import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .config import MorphConfig
from .losses import chunked_cross_entropy
from .mamba2_block import Mamba2Block
from .mla_attention import MLAAttention
from .mlp import SwiGLU
from .mod_router import MoDWrapper
from .self_model_head import SelfModelHead


class MorphLayer(nn.Module):
    def __init__(self, cfg: MorphConfig, idx: int):
        super().__init__()
        self.is_attn = cfg.is_attn(idx)
        if self.is_attn:
            self.mixer = MLAAttention(cfg, is_producer=cfg.is_producer(idx))
        else:
            mixer = Mamba2Block(cfg)
            if cfg.mod_enabled and idx in cfg.mod_layers:
                mixer = MoDWrapper(cfg, mixer)
            self.mixer = mixer
        self.mlp = SwiGLU(cfg)

    def forward(self, x, latent=None):
        if self.is_attn:
            x, latent = self.mixer(x, latent)
        else:
            x = self.mixer(x)
        return self.mlp(x), latent


class MorphModel(nn.Module):
    def __init__(self, cfg: MorphConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([MorphLayer(cfg, i) for i in range(cfg.n_layers)])
        self.norm_f = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.group_of = cfg.group_of()
        self.grad_ckpt = False

    def forward(self, input_ids, collect: Iterable[int] = ()):
        """Returns (final normed hidden, {layer_idx: residual state entering that layer})."""
        collect = set(collect)
        x = self.embed(input_ids)
        latents: Dict[int, torch.Tensor] = {}
        collected: Dict[int, torch.Tensor] = {}
        for i, layer in enumerate(self.layers):
            if i in collect:
                collected[i] = x
            g = self.group_of.get(i)
            lat = latents.get(g) if g is not None else None
            if self.grad_ckpt and self.training and torch.is_grad_enabled():
                x, lat = checkpoint(layer, x, lat, use_reentrant=False)
            else:
                x, lat = layer(x, lat)
            if g is not None and g not in latents:
                latents[g] = lat
        return self.norm_f(x), collected


class MorphForCausalLM(nn.Module):
    def __init__(self, cfg: MorphConfig):
        super().__init__()
        self.cfg = cfg
        self.model = MorphModel(cfg)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.model.embed.weight
        self.targets = cfg.resolved_targets() if cfg.self_model_enabled else []
        self.self_model = SelfModelHead(cfg, len(self.targets)) if self.targets else None
        self._init_weights()

    def _init_weights(self):
        std = self.cfg.init_std
        out_std = std / math.sqrt(2 * self.cfg.n_layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                s = out_std if getattr(m.weight, "residual_out", False) else std
                nn.init.normal_(m.weight, mean=0.0, std=s)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=std)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.model.embed.weight.numel()
        return n

    def set_grad_checkpointing(self, on: bool = True):
        self.model.grad_ckpt = on

    def forward(self, input_ids, targets=None, sm_lambda: Optional[float] = None, collect: Iterable[int] = ()):
        """targets: already shifted next tokens (same shape as input_ids), -100 = ignore."""
        want = set(collect) | (set(self.targets) if (targets is not None and self.self_model is not None) else set())
        hidden, collected = self.model(input_ids, collect=want)
        out = {"hidden": hidden, "collected": collected}
        if targets is None:
            out["logits"] = self.lm_head(hidden)
            return out
        ce, zsq = chunked_cross_entropy(hidden, self.lm_head.weight, targets, self.cfg.ce_chunk)
        loss = ce + self.cfg.z_loss * zsq
        out.update(ce=ce.detach(), z_loss=zsq.detach())
        if self.self_model is not None:
            if self.cfg.self_model_probe:   # measures self-modelability without shaping the backbone
                sm, per = self.self_model(hidden.detach(), [collected[t].detach() for t in self.targets])
                loss = loss + sm
            else:
                lam = self.cfg.self_model_lambda if sm_lambda is None else sm_lambda
                sm, per = self.self_model(hidden, [collected[t] for t in self.targets])
                loss = loss + lam * sm      # computed even at lam = 0 so DDP sees every parameter used
            out.update(sm_loss=sm.detach(), sm_per_target=per)
        out["loss"] = loss
        return out
