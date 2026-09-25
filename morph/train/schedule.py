"""Warmup-Stable-Decay LR schedule and the self-modeling lambda ramp.

WSD keeps the LR constant until the final `decay_frac` of the planned steps, then decays
with a (1 - sqrt) shape. Because nothing depends on the total until the decay starts, the
token budget can be raised on resume as long as the run is still in the stable phase.
"""
from __future__ import annotations

import math


def wsd_lr(step: int, total_steps: int, warmup: int, decay_frac: float, peak: float, min_frac: float = 0.0) -> float:
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    decay_start = decay_start_step(total_steps, decay_frac)
    if step < decay_start:
        return peak
    prog = min(1.0, (step - decay_start) / max(1, total_steps - decay_start))
    return peak * (min_frac + (1.0 - min_frac) * (1.0 - math.sqrt(prog)))


def decay_start_step(total_steps: int, decay_frac: float) -> int:
    return int(round(total_steps * (1.0 - decay_frac)))


def sm_lambda(step: int, warmup: int, lam: float) -> float:
    """Self-modeling weight ramps 0 -> lam over the LR warmup."""
    return lam * min(1.0, (step + 1) / max(1, warmup))
