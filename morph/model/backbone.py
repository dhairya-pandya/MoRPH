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
from .losses import chunked_cross_entropy, fused_linear_cross_entropy
from .mamba2_block import Mamba2Block
from .mla_attention import MLAAttention
from .mlp import SwiGLU
from .mod_router import MoDRouter, route_causal, route_topk
from .self_model_head import SelfModelHead


class MorphLayer(nn.Module):
    def __init__(self, cfg: MorphConfig, idx: int):
        super().__init__()
        self.is_attn = cfg.is_attn(idx)
        if self.is_attn:
            self.mixer = MLAAttention(cfg, is_producer=cfg.is_producer(idx))
        else:
            self.mixer = Mamba2Block(cfg)
        self.mlp = SwiGLU(cfg)
        self.router = MoDRouter(cfg.d_model) if (cfg.mod_enabled and idx in cfg.mod_layers) else None

    def _inner(self, x):
        return self.mlp(self.mixer(x))

    def forward(self, x, latent=None, capacity: Optional[float] = None, mod_mode: str = "topk"):
        """Returns (x, latent, mod_aux, executed_fraction)."""
        zero = x.new_zeros((), dtype=torch.float32)
        if self.is_attn:
            x, latent = self.mixer(x, latent)
            return self.mlp(x), latent, zero, 1.0
        if self.router is None:
            return self._inner(x), latent, zero, 1.0
        logits = self.router.logits(x)
        if mod_mode == "causal":
            out, frac = route_causal(x, logits, self._inner)
            return out, latent, zero, frac
        out, aux, frac = route_topk(x, logits, capacity, self._inner)
        return out, latent, aux, frac


class MorphModel(nn.Module):
    def __init__(self, cfg: MorphConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([MorphLayer(cfg, i) for i in range(cfg.n_layers)])
        self.norm_f = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.group_of = cfg.group_of()
        self.grad_ckpt = False

    def forward(self, input_ids, collect: Iterable[int] = (), capacity: Optional[float] = None,
                mod_mode: str = "topk"):
        """Returns (final normed hidden, {layer_idx: residual state entering that layer}, mod stats)."""
        collect = set(collect)
        capacity = self.cfg.mod_capacity if capacity is None else capacity
        x = self.embed(input_ids)
        latents: Dict[int, torch.Tensor] = {}
        collected: Dict[int, torch.Tensor] = {}
        aux_terms, fracs = [], {}
        for i, layer in enumerate(self.layers):
            if i in collect:
                collected[i] = x
            g = self.group_of.get(i)
            lat = latents.get(g) if g is not None else None
            if self.grad_ckpt and self.training and torch.is_grad_enabled():
                x, lat, aux, frac = checkpoint(layer, x, lat, capacity, mod_mode, use_reentrant=False)
            else:
                x, lat, aux, frac = layer(x, lat, capacity, mod_mode)
            if g is not None and g not in latents:
                latents[g] = lat
            if layer.router is not None:
                aux_terms.append(aux)
                fracs[i] = frac
        stats = {"mod_aux": torch.stack(aux_terms).mean() if aux_terms else None, "mod_frac": fracs}
        return self.norm_f(x), collected, stats


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
            if isinstance(m, nn.Linear) and getattr(m.weight, "zero_init", False):
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
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

    def forward(self, input_ids, targets=None, sm_lambda: Optional[float] = None, collect: Iterable[int] = (),
                mod_capacity: Optional[float] = None, mod_mode: str = "topk"):
        """targets: already shifted next tokens (same shape as input_ids), -100 = ignore.
        mod_mode: "topk" (training/static shapes) or "causal" (per-token router threshold, generation-safe)."""
        want = set(collect) | (set(self.targets) if (targets is not None and self.self_model is not None) else set())
        hidden, collected, mod = self.model(input_ids, collect=want, capacity=mod_capacity, mod_mode=mod_mode)
        out = {"hidden": hidden, "collected": collected, "mod_frac": mod["mod_frac"]}
        if targets is None:
            out["logits"] = self.lm_head(hidden)
            return out
        if torch.is_grad_enabled():
            loss, ce, zsq = fused_linear_cross_entropy(hidden, self.lm_head.weight, targets,
                                                       self.cfg.ce_chunk, self.cfg.z_loss)
        else:
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
        if mod["mod_aux"] is not None:
            loss = loss + self.cfg.mod_aux_weight * mod["mod_aux"]
            out["mod_aux"] = mod["mod_aux"].detach()
        out["loss"] = loss
        return out
