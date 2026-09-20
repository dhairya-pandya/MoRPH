"""MorphConfig — single source of truth for architecture + self-remodeling knobs.

The layer schedule is expressed as a string pattern of block types repeated to
`n_layers`, so the SSM:attention ratio and where MLA/CLA sit is fully declarative.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List
import json


@dataclass
class MorphConfig:
    # ---- vocab / dims ----
    vocab_size: int = 32000          # reuse Llama-3 / GPT-NeoX 32k BPE
    d_model: int = 768
    n_layers: int = 28
    max_seq_len: int = 2048
    tie_embeddings: bool = True

    # ---- layer schedule ----
    # Pattern of block types tiled to n_layers. "m" = Mamba2 SSM, "a" = MLA attention.
    # Default ~6:1 SSM:attention (Zamba2/Jamba-proven). 28 layers -> 4 attention layers.
    layer_pattern: str = "mmmmmma"   # 6 SSM then 1 attention, tiled

    # ---- Mamba2 SSM block ----
    ssm_d_state: int = 128
    ssm_d_conv: int = 4
    ssm_expand: int = 2
    ssm_headdim: int = 64

    # ---- MLA attention block ----
    n_heads: int = 12
    kv_lora_rank: int = 128          # latent KV dimension (the KV-cache compression)
    q_lora_rank: int = 0             # 0 = no low-rank query proj (small model)
    qk_rope_head_dim: int = 32       # decoupled-RoPE per-head dim
    qk_nope_head_dim: int = 32       # non-positional per-head dim
    v_head_dim: int = 64
    rope_theta: float = 10000.0

    # ---- CLA: cross-layer KV sharing across MLA layers ----
    # Group size G: every G attention layers share one latent-KV projection.
    cla_group_size: int = 2

    # ---- MLP ----
    mlp_ratio: float = 4.0
    mlp_bias: bool = False

    # ---- self-modeling aux head (Premakumar et al. 2024) ----
    self_model_enabled: bool = True
    self_model_layer: int = -2       # which hidden layer's state to self-predict (index)
    self_model_dim: int = 256        # #activation units predicted (a subset, for cheapness)
    self_model_lambda: float = 0.1   # aux loss weight

    # ---- Mixture-of-Depths (Stage 2 structural self-remodel) ----
    mod_enabled: bool = False        # off during Stage 1 pretrain; on in Stage 2
    mod_capacity: float = 0.5        # fraction of tokens routed through full compute
    mod_layers: List[int] = field(default_factory=list)  # which layers gain a MoD router

    # ---- SVF (Stage 3 weight-space self-adaptation) ----
    svf_enabled: bool = False

    # ---- misc ----
    norm_eps: float = 1e-5
    dropout: float = 0.0
    init_std: float = 0.02

    def block_types(self) -> List[str]:
        pat = self.layer_pattern
        return [pat[i % len(pat)] for i in range(self.n_layers)]

    @property
    def head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "MorphConfig":
        with open(path) as f:
            return cls(**json.load(f))
