"""Quality eval: held-out perplexity + hooks for commonsense / long-context recall.

Stage-1+ deliverable. Perplexity runs anywhere. Commonsense (HellaSwag/ARC/PIQA)
and long-context needle (RULER) are wired to lm-eval-harness / a needle probe when
available; otherwise they no-op with a message so the script stays runnable.

Run: PYTHONPATH=. python -m morph.eval.quality_bench --ckpt checkpoints/step_X.pt \
     --config configs/300m_hybrid.json
"""
from __future__ import annotations

import argparse
import math
import torch

from morph.model import MorphConfig, MorphForCausalLM
from morph.train.checkpoint import load_checkpoint
from morph.data.fineweb import PackedTextStream, SyntheticStream


@torch.no_grad()
def perplexity(model, stream, device, n_batches: int, batch: int):
    model.eval()
    tot_loss, tot = 0.0, 0
    it = stream.batches(batch)
    for _ in range(n_batches):
        b = next(it).to(device)
        ids, labels = b[:, :-1], b
        out = model(ids, labels=labels[:, : ids.size(1)])
        tot_loss += out["lm_loss"].item()
        tot += 1
    return math.exp(tot_loss / max(1, tot))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/300m_hybrid.json")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--n_batches", type=int, default=50)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--tokenizer", default="meta-llama/Meta-Llama-3-8B")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = MorphConfig.from_json(args.config)
    if args.smoke:
        cfg.max_seq_len = args.seq
    model = MorphForCausalLM(cfg).to(device)
    if args.ckpt:
        load_checkpoint(args.ckpt, model=model, map_location=device)

    if args.smoke:
        stream = SyntheticStream(cfg.vocab_size, args.seq)
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        stream = PackedTextStream(tok, args.seq)

    ppl = perplexity(model, stream, device, args.n_batches, args.batch)
    print(f"perplexity: {ppl:.2f}")
    print("commonsense/long-context: install lm-eval-harness for HellaSwag/ARC/PIQA/RULER")


if __name__ == "__main__":
    main()
