"""CLA accounting helpers: which attention layers cache a latent.

Grouping itself lives in `MorphConfig.groups()`; only the producer (first layer) of each
group computes and caches the latent.
"""
from __future__ import annotations

from typing import List


def cached_layers(cfg) -> List[int]:
    return [g[0] for g in cfg.groups()]


def kv_bytes_per_token(cfg, dtype_bytes: int = 2) -> int:
    return len(cfg.groups()) * cfg.kv_lora_rank * dtype_bytes
