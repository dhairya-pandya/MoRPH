"""MORPH backbone: assembles the SSM-majority hybrid + heads.

Layer i is either a Mamba2 SSM block ('m') or an MLA attention block ('a') per
`cfg.layer_pattern`, each followed by a SwiGLU MLP. Attention layers share a
`SharedLatentKV` within their CLA group. Optional MoD wrappers add adaptive depth.
The self-modeling aux head (Premakumar 2024) is computed in MorphForCausalLM.
"""
from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MorphConfig
from .mamba2_block import Mamba2Block
from .mla_attention import MLAAttention, SharedLatentKV, build_rope
from .mlp import SwiGLU
from .mod_router import MoDWrapper
from .self_model_head import SelfModelHead
from .cla import assign_cla_groups


class MorphModel(nn.Module):
    def __init__(self, cfg: MorphConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)

        block_types = cfg.block_types()
        cla_map = assign_cla_groups(block_types, cfg.cla_group_size)
        n_groups = (max(cla_map.values()) + 1) if cla_map else 0
        # one shared latent-KV module per CLA group
        self.shared_kv = nn.ModuleList([SharedLatentKV(cfg) for _ in range(n_groups)])

        self.layers = nn.ModuleList()
        self.is_attn = []
        for i, t in enumerate(block_types):
            if t == "a":
                mixer = MLAAttention(cfg, self.shared_kv[cla_map[i]])
                self.is_attn.append(True)
            else:
                mixer = Mamba2Block(cfg)
                self.is_attn.append(False)
            if cfg.mod_enabled and i in cfg.mod_layers:
                mixer = MoDWrapper(cfg, mixer)
            self.layers.append(nn.ModuleDict({"mixer": mixer, "mlp": SwiGLU(cfg)}))

        self.norm_f = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        cos, sin = build_rope(cfg.max_seq_len, cfg.qk_rope_head_dim, cfg.rope_theta,
                              torch.device("cpu"), torch.float32)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, input_ids: torch.Tensor, collect_layer: Optional[int] = None):
        x = self.embed(input_ids)
        rope = (self.rope_cos.to(x.dtype), self.rope_sin.to(x.dtype))
        collected = None
        n = len(self.layers)
        tgt = collect_layer % n if collect_layer is not None else None
        for i, layer in enumerate(self.layers):
            x = layer["mixer"](x, rope_cache=rope)
            x = layer["mlp"](x)
            if tgt is not None and i == tgt:
                collected = x
        return self.norm_f(x), collected


class MorphForCausalLM(nn.Module):
    def __init__(self, cfg: MorphConfig):
        super().__init__()
        self.cfg = cfg
        self.model = MorphModel(cfg)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.model.embed.weight
        self.self_model = SelfModelHead(cfg) if cfg.self_model_enabled else None
        self.apply(self._init)

    def _init(self, m):
        std = self.cfg.init_std
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=std)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding and self.cfg.tie_embeddings:
            n -= self.model.embed.weight.numel()
        return n

    def forward(self, input_ids, labels=None):
        collect = self.cfg.self_model_layer if self.self_model is not None else None
        hidden, collected = self.model(input_ids, collect_layer=collect)
        logits = self.lm_head(hidden)

        out = {"logits": logits}
        if labels is not None:
            lm_loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
            loss = lm_loss
            out["lm_loss"] = lm_loss.detach()
            if self.self_model is not None and collected is not None:
                sm_loss = self.self_model(hidden, collected)
                loss = loss + self.cfg.self_model_lambda * sm_loss
                out["self_model_loss"] = sm_loss.detach()
            out["loss"] = loss
        return out
