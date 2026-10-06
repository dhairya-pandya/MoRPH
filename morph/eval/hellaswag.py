"""HellaSwag validation accuracy (10,042 items), completion-style like llm.c / lm-eval acc_norm.

For each item the 4 endings are scored by the mean log-likelihood of their tokens given the
context; accuracy = fraction where the gold ending scores highest.

    PYTHONPATH=. python -m morph.eval.hellaswag --ckpt runs/main_s/checkpoints/ckpt_*.pt [--limit 1000]
"""
from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

REPO, FILE = "Rowan/hellaswag", "data/validation-00000-of-00001.parquet"


def load_items(cache_dir: str):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(REPO, FILE, repo_type="dataset", cache_dir=cache_dir)
    return pq.read_table(path, columns=["ctx", "endings", "label"]).to_pylist()


@torch.no_grad()
def score_item(model, tok, item, device, amp_dtype) -> bool:
    ctx = tok.encode(item["ctx"], add_special_tokens=False).ids
    rows, masks = [], []
    for end in item["endings"]:
        e = tok.encode(" " + end, add_special_tokens=False).ids
        rows.append(ctx + e)
        masks.append([0] * len(ctx) + [1] * len(e))
    L = max(map(len, rows))
    ids = torch.zeros(4, L, dtype=torch.long)
    m = torch.zeros(4, L)
    for i, (r, k) in enumerate(zip(rows, masks)):
        ids[i, :len(r)] = torch.tensor(r)
        m[i, :len(k)] = torch.tensor(k, dtype=torch.float)
    ids, m = ids.to(device), m.to(device)
    with torch.autocast(device.type, dtype=amp_dtype) if amp_dtype else torch.no_grad():
        logits = model(ids)["logits"].float()
    lp = -F.cross_entropy(logits[:, :-1].transpose(1, 2), ids[:, 1:], reduction="none")
    mm = m[:, 1:]
    score = (lp * mm).sum(1) / mm.sum(1)
    return int(score.argmax()) == int(item["label"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--tokenizer", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)
    from tokenizers import Tokenizer
    from morph.eval.quality_bench import load_model
    from morph.train.distributed import pick_precision
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args, device).eval()
    tok = Tokenizer.from_pretrained(args.tokenizer)
    items = load_items(args.cache)
    items = items[: args.limit] if args.limit else items
    amp = pick_precision(device)
    correct = sum(score_item(model, tok, it, device, amp) for it in items)
    print(json.dumps({"hellaswag_acc": correct / len(items), "n": len(items)}))


if __name__ == "__main__":
    main()
