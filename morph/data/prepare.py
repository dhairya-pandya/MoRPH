"""Tokenize FineWeb-Edu into uint16 token shards (run once, CPU only).

Output layout (`--out`):
    train_00000.bin ...   raw little-endian uint16 tokens, docs separated by EOS
    val_00000.bin         every `val_every`-th document until `val_tokens` is reached
    meta.json             tokenizer, vocab, eos id, per-shard token counts, provenance

Parquet files of the source dataset are downloaded one at a time, read in record batches
and deleted after use, so peak disk is ~one parquet file + shards.

    python -m morph.data.prepare --out data/fineweb_edu_smollm2 --train_tokens 5.2e9
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Iterable, Iterator, List

import numpy as np

META = "meta.json"


class ShardWriter:
    """Accumulates token ids and writes fixed-size uint16 shards."""

    def __init__(self, out_dir: str, prefix: str, shard_tokens: int):
        self.out_dir, self.prefix, self.shard_tokens = out_dir, prefix, int(shard_tokens)
        self.buf: List[np.ndarray] = []
        self.buf_len = 0
        self.counts: List[int] = []
        self.total = 0

    def add(self, ids: np.ndarray):
        self.buf.append(ids)
        self.buf_len += len(ids)
        self.total += len(ids)
        while self.buf_len >= self.shard_tokens:
            flat = np.concatenate(self.buf)
            self._write(flat[: self.shard_tokens])
            rest = flat[self.shard_tokens:]
            self.buf, self.buf_len = ([rest] if len(rest) else []), len(rest)

    def _write(self, arr: np.ndarray):
        name = f"{self.prefix}_{len(self.counts):05d}.bin"
        tmp = os.path.join(self.out_dir, name + ".tmp")
        arr.astype("<u2").tofile(tmp)
        os.replace(tmp, os.path.join(self.out_dir, name))
        self.counts.append(int(len(arr)))

    def close(self):
        if self.buf_len:
            self._write(np.concatenate(self.buf))
            self.buf, self.buf_len = [], 0

    def files(self) -> List[str]:
        return [f"{self.prefix}_{i:05d}.bin" for i in range(len(self.counts))]


def write_shards(docs: Iterable[List[int]], out_dir: str, eos_id: int, train_tokens: int,
                 val_every: int = 1000, val_tokens: int = 5_000_000,
                 shard_tokens: int = 100_000_000, meta_extra: dict | None = None) -> dict:
    """Consume tokenized docs (lists of ids, no EOS) until `train_tokens` train tokens exist."""
    os.makedirs(out_dir, exist_ok=True)
    train = ShardWriter(out_dir, "train", shard_tokens)
    val = ShardWriter(out_dir, "val", 10 ** 12)          # single val shard
    n_docs = n_val_docs = 0
    for ids in docs:
        arr = np.fromiter(ids, dtype=np.int64, count=len(ids))
        if arr.size and arr.max() >= 65536:
            raise ValueError("token id >= 65536 does not fit uint16")
        arr = np.append(arr, eos_id).astype(np.uint16)
        if n_docs % val_every == 0 and val.total < val_tokens:
            val.add(arr)
            n_val_docs += 1
        else:
            train.add(arr)
        n_docs += 1
        if train.total >= train_tokens:
            break
    train.close()
    val.close()
    meta = {
        "eos_id": int(eos_id),
        "dtype": "uint16",
        "train": {"files": train.files(), "tokens": train.counts},
        "val": {"files": val.files(), "tokens": val.counts},
        "docs": n_docs, "val_docs": n_val_docs,
        "train_tokens": int(sum(train.counts)), "val_tokens": int(sum(val.counts)),
    }
    meta.update(meta_extra or {})
    with open(os.path.join(out_dir, META), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def fineweb_docs(tokenizer, repo: str, subdir: str, batch_docs: int, cache_dir: str) -> Iterator[List[int]]:
    """Stream parquet files of `repo/subdir` in sorted order, yield tokenized docs."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    files = sorted(f.path for f in HfApi().list_repo_tree(repo, path_in_repo=subdir, repo_type="dataset")
                   if f.path.endswith(".parquet"))
    for fi, path in enumerate(files):
        t0 = time.time()
        local = hf_hub_download(repo, path, repo_type="dataset", cache_dir=cache_dir)
        pf = pq.ParquetFile(local)
        for batch in pf.iter_batches(batch_size=batch_docs, columns=["text"]):
            texts = batch.column(0).to_pylist()
            for enc in tokenizer.encode_batch(texts, add_special_tokens=False):
                yield enc.ids
        print(f"[prepare] finished {path} ({fi + 1}/{len(files)}) in {time.time() - t0:.0f}s", flush=True)
        try:
            os.remove(os.path.realpath(local))
        except OSError:
            pass


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--subdir", default="sample/10BT")
    ap.add_argument("--train_tokens", type=float, default=5.2e9)
    ap.add_argument("--val_tokens", type=float, default=5e6)
    ap.add_argument("--val_every", type=int, default=1000)
    ap.add_argument("--shard_tokens", type=float, default=1e8)
    ap.add_argument("--batch_docs", type=int, default=2000)
    ap.add_argument("--cache_dir", default=None)
    args = ap.parse_args(argv)

    from tokenizers import Tokenizer
    tok = Tokenizer.from_pretrained(args.tokenizer)
    eos = tok.token_to_id("<|endoftext|>")
    if eos is None:
        raise ValueError("tokenizer has no <|endoftext|> token")
    t0 = time.time()
    meta = write_shards(
        fineweb_docs(tok, args.dataset, args.subdir, args.batch_docs, args.cache_dir),
        args.out, eos_id=eos, train_tokens=int(args.train_tokens), val_every=args.val_every,
        val_tokens=int(args.val_tokens), shard_tokens=int(args.shard_tokens),
        meta_extra={"tokenizer": args.tokenizer, "vocab_size": tok.get_vocab_size(),
                    "source": f"{args.dataset}/{args.subdir}"},
    )
    print(f"[prepare] done: {meta['train_tokens'] / 1e9:.3f}B train, {meta['val_tokens'] / 1e6:.1f}M val "
          f"tokens from {meta['docs']} docs in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
