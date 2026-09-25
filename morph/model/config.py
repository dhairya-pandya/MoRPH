"""MorphConfig — single source of truth for architecture + training-objective knobs.

v2 layout: every layer is (mixer + SwiGLU MLP). The mixer is a Mamba2 SSM except at
`attn_layers`, which are NoPE Multi-head Latent Attention. `cla_groups` lists groups of
attention layers that share one cached latent: the first layer of a group (producer)
computes it, later layers (consumers) reuse it with their own up-projection.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields
from typing import Dict, List, Optional
import hashlib
import json


@dataclass
class MorphConfig:
    # ---- vocab / dims ----
    vocab_size: int = 49152          # SmolLM2 tokenizer (ungated, FineWeb-Edu trained)
    d_model: int = 640
    n_layers: int = 24
    max_seq_len: int = 2048
    tie_embeddings: bool = True

    # ---- layer schedule: attention mid-stack, SSM at the extremes (~1:5) ----
    attn_layers: List[int] = field(default_factory=lambda: [4, 9, 14, 19])
    cla_groups: List[List[int]] = field(default_factory=lambda: [[4, 9], [14, 19]])

    # ---- Mamba2 SSM (parameter layout of mamba_ssm.Mamba2, ngroups=1) ----
    ssm_d_state: int = 128
    ssm_d_conv: int = 4
    ssm_expand: int = 2
    ssm_headdim: int = 64
    ssm_chunk: int = 64
    ssm_backend: str = "torch"       # torch | mamba_ssm | fla

    # ---- MLA attention (NoPE) ----
    n_heads: int = 10
    head_dim: int = 64
    kv_lora_rank: int = 128          # cached latent size per CLA group

    # ---- MLP ----
    mlp_hidden: int = 1728

    # ---- objective ----
    z_loss: float = 1e-4
    ce_chunk: int = 1024             # tokens per chunk in the chunked cross-entropy

    # ---- self-modeling (Premakumar et al. 2024) ----
    self_model_enabled: bool = True
    self_model_targets: Optional[List[int]] = None   # layer indices whose INPUT state is predicted; None -> attn_layers
    self_model_lambda: float = 0.1
    self_model_detach: bool = False  # paper: gradients flow into the targets
    self_model_probe: bool = False   # baseline readout: head trains on fully detached states, backbone unaffected

    # ---- Mixture-of-Depths (Stage 2; SSM layers only) ----
    mod_enabled: bool = False
    mod_capacity: float = 0.5
    mod_layers: List[int] = field(default_factory=list)

    # ---- SVF (Stage 3) ----
    svf_enabled: bool = False

    # ---- misc ----
    norm_eps: float = 1e-5
    init_std: float = 0.02

    def __post_init__(self):
        attn = set(self.attn_layers)
        if any(i < 0 or i >= self.n_layers for i in attn):
            raise ValueError(f"attn_layers {self.attn_layers} out of range for n_layers={self.n_layers}")
        seen = set()
        for g in self.cla_groups:
            if not g or any(i not in attn for i in g):
                raise ValueError(f"cla group {g} must be a non-empty subset of attn_layers")
            if sorted(g) != list(g) or seen & set(g):
                raise ValueError(f"cla groups must be sorted and disjoint: {self.cla_groups}")
            seen |= set(g)
        if any(i in attn for i in self.mod_layers):
            raise ValueError("MoD is only supported on SSM layers (attention gather is non-causal)")
        if (self.ssm_expand * self.d_model) % self.ssm_headdim:
            raise ValueError("ssm_expand * d_model must be divisible by ssm_headdim")
        for t in self.resolved_targets():
            if t < 0 or t >= self.n_layers:
                raise ValueError(f"self_model target {t} out of range")

    # ---- derived structure ----
    def is_attn(self, i: int) -> bool:
        return i in self.attn_layers

    def groups(self) -> List[List[int]]:
        """CLA groups including singleton groups for ungrouped attention layers."""
        grouped = {i for g in self.cla_groups for i in g}
        out = [list(g) for g in self.cla_groups] + [[i] for i in self.attn_layers if i not in grouped]
        return sorted(out, key=lambda g: g[0])

    def group_of(self) -> Dict[int, int]:
        return {i: gi for gi, g in enumerate(self.groups()) for i in g}

    def is_producer(self, i: int) -> bool:
        return any(g[0] == i for g in self.groups())

    def resolved_targets(self) -> List[int]:
        return list(self.self_model_targets) if self.self_model_targets is not None else list(self.attn_layers)

    @property
    def ssm_d_inner(self) -> int:
        return self.ssm_expand * self.d_model

    @property
    def ssm_nheads(self) -> int:
        return self.ssm_d_inner // self.ssm_headdim

    # ---- io ----
    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "MorphConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown MorphConfig keys: {sorted(unknown)}")
        return cls(**d)

    @classmethod
    def from_json(cls, path: str) -> "MorphConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def arch_hash(self) -> str:
        """Hash of fields that change parameter shapes (resume compatibility check)."""
        keys = ["vocab_size", "d_model", "n_layers", "tie_embeddings", "attn_layers", "cla_groups",
                "ssm_d_state", "ssm_d_conv", "ssm_expand", "ssm_headdim", "n_heads", "head_dim",
                "kv_lora_rank", "mlp_hidden", "self_model_enabled", "self_model_targets",
                "mod_enabled", "mod_layers"]
        blob = json.dumps({k: getattr(self, k) for k in keys}, sort_keys=True)
        return hashlib.sha1(blob.encode()).hexdigest()[:12]
