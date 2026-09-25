"""Data gate: shard writing round-trip, world-size-independent sampling, resume, val split."""
import json
import os
import tempfile

import numpy as np

from morph.data.prepare import write_shards
from morph.data.shards import ShardSet, WindowSampler, micro_batches


def _docs(n=400, seed=0):
    rng = np.random.default_rng(seed)
    for i in range(n):
        yield rng.integers(1, 1000, size=rng.integers(5, 60)).tolist()


def test_write_shards_roundtrip_and_val_split():
    with tempfile.TemporaryDirectory() as d:
        docs = list(_docs())
        meta = write_shards(iter(docs), d, eos_id=0, train_tokens=5000, val_every=10,
                            val_tokens=300, shard_tokens=1000)
        assert meta["train_tokens"] >= 5000 and len(meta["train"]["files"]) >= 5
        assert all(c == 1000 for c in meta["train"]["tokens"][:-1])
        train = np.concatenate([np.fromfile(os.path.join(d, f), dtype=np.uint16) for f in meta["train"]["files"]])
        val = np.fromfile(os.path.join(d, meta["val"]["files"][0]), dtype=np.uint16)
        # rebuild the expected streams: every 10th doc goes to val until val_tokens is reached
        exp_train, exp_val, vt = [], [], 0
        for i, doc in enumerate(docs[: meta["docs"]]):
            seq = doc + [0]
            if i % 10 == 0 and vt < 300:
                exp_val += seq
                vt += len(seq)
            else:
                exp_train += seq
        assert train.tolist() == exp_train and val.tolist() == exp_val
        with open(os.path.join(d, "meta.json")) as f:
            assert json.load(f)["eos_id"] == 0


def test_sampler_is_independent_of_world_size_and_micro_batch():
    s = WindowSampler(n_windows=1000, global_batch=16, seed=3)

    class DS:
        def window(self, w):
            return np.array([w, w])

    for step in (0, 5, 62, 63, 200):     # includes epoch wrap (1000 / 16 = 62.5 steps/epoch)
        ref = sorted(s.step_ids(step))
        for world, micro in [(1, 16), (2, 4), (4, 2), (8, 1)]:
            got = []
            for r in range(world):
                for mb in micro_batches(DS(), s, step, r, world, micro):
                    got += mb[:, 0].tolist()
            assert sorted(got) == ref


def test_sampler_epochs_cover_all_windows_once():
    s = WindowSampler(n_windows=100, global_batch=10, seed=1)
    ids = [w for st in range(10) for w in s.step_ids(st)]
    assert sorted(ids) == list(range(100))
    assert s.step_ids(3) == WindowSampler(100, 10, 1).step_ids(3)   # resume = same stream


def test_shardset_windows_stay_inside_shards():
    with tempfile.TemporaryDirectory() as d:
        write_shards(_docs(), d, eos_id=0, train_tokens=3000, val_every=10 ** 9, val_tokens=0, shard_tokens=1000)
        ds = ShardSet(d, "train", seq_len=64)
        assert ds.n_windows == sum((1000 - 1) // 64 for _ in range(3))
        w = ds.window(ds.n_windows - 1)
        assert len(w) == 65
