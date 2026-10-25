"""Stage 4 data: chat conversations -> packed token shards with an assistant-only loss mask.

Conversations are rendered in ChatML, which the SmolLM2 tokenizer already has tokens for:

    <|im_start|>system\\n...<|im_end|>\\n<|im_start|>user\\n...<|im_end|>\\n<|im_start|>assistant\\n...<|im_end|>\\n

Every token gets a mask bit: 1 for an assistant reply and its closing <|im_end|> (what the
model should learn to produce), 0 for everything else. Conversations are separated by EOS and
packed back to back; the shards sit next to uint8 mask shards (`train_00000.mask.bin`) and
`meta.json` sets `"masked": true`, which the shard loader and trainer pick up.

    python -m morph.data.sft --out data/sft_smoltalk
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, Iterable, Iterator, List, Tuple

import numpy as np

from .prepare import META, ShardWriter

IM_START, IM_END = "<|im_start|>", "<|im_end|>"


def render(tok, messages: List[Dict[str, str]]) -> Tuple[List[int], List[int]]:
    """Token ids and assistant-only mask for one conversation (no trailing EOS)."""
    start, end = tok.token_to_id(IM_START), tok.token_to_id(IM_END)
    nl = tok.encode("\n", add_special_tokens=False).ids
    ids: List[int] = []
    mask: List[int] = []
    for m in messages:
        head = [start] + tok.encode(f"{m['role']}\n", add_special_tokens=False).ids
        body = tok.encode(m["content"], add_special_tokens=False).ids + [end]
        learn = 1 if m["role"] == "assistant" else 0
        ids += head + body + nl
        mask += [0] * len(head) + [learn] * len(body) + [0] * len(nl)
    return ids, mask


def prompt_ids(tok, messages: List[Dict[str, str]]) -> List[int]:
    """Conversation so far plus an open assistant turn, for generation."""
    ids, _ = render(tok, messages)
    return ids + [tok.token_to_id(IM_START)] + tok.encode("assistant\n", add_special_tokens=False).ids


def write_sft_shards(convs: Iterable[List[Dict[str, str]]], tok, out_dir: str, eos_id: int,
                     max_tokens: int, val_convs: Iterable[List[Dict[str, str]]] = (),
                     shard_tokens: int = 50_000_000, meta_extra: dict | None = None) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    meta = {"eos_id": int(eos_id), "dtype": "uint16", "masked": True}
    for split, source, cap in (("train", convs, max_tokens), ("val", val_convs, 10 ** 12)):
        ids_w = ShardWriter(out_dir, split, shard_tokens)
        mask_w = ShardWriter(out_dir, split, shard_tokens, dtype="u1", suffix=".mask.bin")
        n = learn = 0
        for msgs in source:
            ids, mask = render(tok, msgs)
            ids_w.add(np.array(ids + [eos_id], dtype=np.uint16))
            mask_w.add(np.array(mask + [0], dtype=np.uint8))
            n += 1
            learn += sum(mask)
            if ids_w.total >= cap:
                break
        ids_w.close()
        mask_w.close()
        meta[split] = {"files": ids_w.files(), "mask_files": mask_w.files(), "tokens": ids_w.counts}
        meta[f"{split}_tokens"] = int(sum(ids_w.counts))
        meta[f"{split}_conversations"] = n
        meta[f"{split}_learn_fraction"] = learn / max(1, sum(ids_w.counts))
    meta.update(meta_extra or {})
    with open(os.path.join(out_dir, META), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def hf_conversations(repo: str, files: List[str], column: str = "messages",
                     cache_dir: str | None = None) -> Iterator[List[Dict[str, str]]]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    for path in files:
        local = hf_hub_download(repo, path, repo_type="dataset", cache_dir=cache_dir)
        for batch in pq.ParquetFile(local).iter_batches(batch_size=1000, columns=[column]):
            for msgs in batch.column(0).to_pylist():
                yield msgs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--dataset", default="HuggingFaceTB/smol-smoltalk")
    ap.add_argument("--max_tokens", type=float, default=3e8)
    ap.add_argument("--val_convs", type=int, default=2000)
    ap.add_argument("--cache_dir", default=None)
    args = ap.parse_args(argv)

    from huggingface_hub import HfApi
    from tokenizers import Tokenizer
    tok = Tokenizer.from_pretrained(args.tokenizer)
    files = sorted(f for f in HfApi().list_repo_files(args.dataset, repo_type="dataset") if f.endswith(".parquet"))
    train = [f for f in files if "/train" in f]
    test = [f for f in files if "/test" in f]

    def first(n, it):
        for i, x in enumerate(it):
            if i >= n:
                return
            yield x

    t0 = time.time()
    meta = write_sft_shards(hf_conversations(args.dataset, train, cache_dir=args.cache_dir), tok, args.out,
                            eos_id=tok.token_to_id("<|endoftext|>"), max_tokens=int(args.max_tokens),
                            val_convs=first(args.val_convs, hf_conversations(args.dataset, test or train[-1:],
                                                                             cache_dir=args.cache_dir)),
                            meta_extra={"tokenizer": args.tokenizer, "vocab_size": tok.get_vocab_size(),
                                        "source": args.dataset})
    print(f"[sft] {meta['train_conversations']} train conversations, {meta['train_tokens'] / 1e6:.1f}M tokens "
          f"({meta['train_learn_fraction']:.0%} learned), {meta['val_conversations']} val, "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
