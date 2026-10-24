"""Deterministic, resumable loading of uint16 token shards.

A sample is a window of seq_len+1 tokens taken at stride seq_len inside one shard. All
train windows are permuted with a seeded RNG per epoch. Optimizer step s consumes
permutation entries [s*G, (s+1)*G) (G = global batch in sequences); rank r of W takes
entries r, r+W, ... and cuts them into micro-batches. The data order therefore depends only
on (seed, G, seq_len): a run can resume on a different number of GPUs or micro-batch size
and see exactly the same data. The only resume state is the step counter.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np
import torch

from .prepare import META


class ShardSet:
    """Windows of `seq_len + 1` tokens over the shards of one split."""

    def __init__(self, data_dir: str, split: str, seq_len: int):
        with open(os.path.join(data_dir, META)) as f:
            self.meta = json.load(f)
        info = self.meta[split]
        self.seq_len = seq_len
        self.arrays = [np.memmap(os.path.join(data_dir, fn), dtype=np.uint16, mode="r") for fn in info["files"]]
        # SFT data: a per-token loss mask (1 = learn this target) stored next to the tokens
        self.masks = ([np.memmap(os.path.join(data_dir, fn), dtype=np.uint8, mode="r") for fn in info["mask_files"]]
                      if self.meta.get("masked") else None)
        per = np.array([max(0, (len(a) - 1) // seq_len) for a in self.arrays], dtype=np.int64)
        self.starts = np.concatenate([[0], np.cumsum(per)])
        self.n_windows = int(self.starts[-1])
        if self.n_windows == 0:
            raise ValueError(f"no {split} windows of length {seq_len + 1} in {data_dir}")

    def _locate(self, wid: int):
        s = int(np.searchsorted(self.starts, wid, side="right") - 1)
        return s, (wid - int(self.starts[s])) * self.seq_len

    def window(self, wid: int) -> np.ndarray:
        s, off = self._locate(wid)
        return np.asarray(self.arrays[s][off: off + self.seq_len + 1], dtype=np.int64)

    def window_mask(self, wid: int) -> np.ndarray:
        s, off = self._locate(wid)
        return np.asarray(self.masks[s][off: off + self.seq_len + 1], dtype=np.bool_)


class SyntheticSet:
    """Random tokens, deterministic per window id (CPU smoke tests)."""

    def __init__(self, vocab_size: int, seq_len: int, n_windows: int = 1_000_000, seed: int = 0):
        self.vocab_size, self.seq_len, self.n_windows, self.seed = vocab_size, seq_len, n_windows, seed

    def window(self, wid: int) -> np.ndarray:
        return np.random.default_rng([self.seed, wid]).integers(0, self.vocab_size, self.seq_len + 1)


class WindowSampler:
    def __init__(self, n_windows: int, global_batch: int, seed: int):
        self.n, self.G, self.seed = n_windows, global_batch, seed
        self._perms: Dict[int, np.ndarray] = {}

    def _perm(self, epoch: int) -> np.ndarray:
        if epoch not in self._perms:
            self._perms = {epoch: np.random.default_rng([self.seed, epoch]).permutation(self.n)}
        return self._perms[epoch]

    def step_ids(self, step: int) -> List[int]:
        out, pos, end = [], step * self.G, (step + 1) * self.G
        while pos < end:
            epoch, i = divmod(pos, self.n)
            take = min(end - pos, self.n - i)
            out.extend(self._perm(epoch)[i: i + take].tolist())
            pos += take
        return out

    def epoch_of(self, step: int) -> float:
        return step * self.G / self.n


def _stack(ds, ids: List[int]):
    """Tensor of windows, or (windows, masks) when the dataset carries a loss mask."""
    tok = torch.from_numpy(np.stack([ds.window(w) for w in ids]))
    if getattr(ds, "masks", None) is None:
        return tok
    return tok, torch.from_numpy(np.stack([ds.window_mask(w) for w in ids]))


def micro_batches(ds, sampler: WindowSampler, step: int, rank: int, world: int, micro: int) -> List:
    ids = sampler.step_ids(step)[rank::world]
    if len(ids) % micro:
        raise ValueError(f"per-rank batch {len(ids)} not divisible by micro-batch {micro}")
    return [_stack(ds, ids[i: i + micro]) for i in range(0, len(ids), micro)]


def val_batches(ds, n_windows: int, micro: int) -> List:
    """First `n_windows` validation windows in order (fixed across evals and runs)."""
    n = min(n_windows, ds.n_windows)
    return [_stack(ds, list(range(i, min(n, i + micro)))) for i in range(0, n, micro)]


def inputs_targets(mb, device):
    """Model inputs and next-token targets; masked-out targets become -100 (ignored)."""
    tok, mask = (mb, None) if torch.is_tensor(mb) else mb
    tok = tok.to(device, non_blocking=True)
    x, y = tok[:, :-1], tok[:, 1:]
    if mask is not None:
        y = y.masked_fill(~mask[:, 1:].to(device), -100)
    return x, y
