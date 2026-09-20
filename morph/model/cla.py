"""CLA grouping: assign attention layers to cross-layer KV-sharing groups.

Given the layer schedule, every `cla_group_size` consecutive *attention* layers share
one `SharedLatentKV`. Returns, for each attention layer index, the id of the shared
module it should use. The backbone builds one SharedLatentKV per distinct group id.
"""
from __future__ import annotations

from typing import Dict, List


def assign_cla_groups(block_types: List[str], group_size: int) -> Dict[int, int]:
    """Map layer_index -> shared_group_id, for attention ('a') layers only."""
    mapping: Dict[int, int] = {}
    attn_count = 0
    for i, t in enumerate(block_types):
        if t == "a":
            mapping[i] = attn_count // max(1, group_size)
            attn_count += 1
    return mapping
