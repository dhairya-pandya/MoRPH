"""Resumable streaming loader for FineWeb-Edu (or any HF text dataset).

Kaggle/Colab sessions are time-capped, so the loader must resume mid-corpus. We
stream with HF `datasets`, skip to a saved document offset, tokenize, pack into
fixed-length blocks, and yield batches. The consumed-document count is checkpointed
by the trainer so the next session continues where this one stopped.

Falls back to a synthetic random-token stream when `datasets`/tokenizer are absent,
so smoke tests run anywhere (Mac/CPU).
"""
from __future__ import annotations

from typing import Iterator, Optional
import torch


class PackedTextStream:
    def __init__(self, tokenizer, seq_len: int, dataset_name: str = "HuggingFaceFW/fineweb-edu",
                 split: str = "train", subset: Optional[str] = "sample-10BT",
                 text_key: str = "text", start_doc: int = 0):
        self.tok = tokenizer
        self.seq_len = seq_len
        self.text_key = text_key
        self.start_doc = start_doc
        self.docs_consumed = start_doc
        self._ds = None
        self.dataset_name = dataset_name
        self.split = split
        self.subset = subset

    def _ensure_ds(self):
        if self._ds is not None:
            return
        from datasets import load_dataset  # lazy import
        kw = {"streaming": True, "split": self.split}
        if self.subset:
            kw["name"] = self.subset
        ds = load_dataset(self.dataset_name, **kw)
        if self.start_doc:
            ds = ds.skip(self.start_doc)
        self._ds = ds

    def token_iter(self) -> Iterator[int]:
        self._ensure_ds()
        eos = getattr(self.tok, "eos_token_id", None) or 0
        for row in self._ds:
            self.docs_consumed += 1
            ids = self.tok.encode(row[self.text_key])
            yield from ids
            yield eos

    def blocks(self) -> Iterator[torch.Tensor]:
        buf = []
        for t in self.token_iter():
            buf.append(t)
            if len(buf) >= self.seq_len + 1:
                yield torch.tensor(buf[: self.seq_len + 1], dtype=torch.long)
                buf = buf[self.seq_len:]

    def batches(self, batch_size: int) -> Iterator[torch.Tensor]:
        batch = []
        for blk in self.blocks():
            batch.append(blk)
            if len(batch) == batch_size:
                yield torch.stack(batch)          # (B, seq_len+1)
                batch = []


class SyntheticStream:
    """Dependency-free random-token batches for smoke tests."""

    def __init__(self, vocab_size: int, seq_len: int, seed: int = 0):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.g = torch.Generator().manual_seed(seed)
        self.docs_consumed = 0

    def batches(self, batch_size: int) -> Iterator[torch.Tensor]:
        while True:
            self.docs_consumed += batch_size
            yield torch.randint(0, self.vocab_size, (batch_size, self.seq_len + 1),
                                 generator=self.g)
