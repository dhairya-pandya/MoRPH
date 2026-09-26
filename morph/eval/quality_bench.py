"""Evaluate a checkpoint: validation perplexity + self-modeling diagnostics.

    PYTHONPATH=. python -m morph.eval.quality_bench --ckpt runs/proxy_R1/checkpoints/ckpt_0003815.pt \
        --data_dir data/fineweb_edu_smollm2 [--weights milestone.safetensors --config configs/model/s.json]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from morph.data.shards import ShardSet, val_batches
from morph.model import MorphConfig, MorphForCausalLM
from morph.train.distributed import pick_precision
from morph.train.pretrain import evaluate


def load_model(args, device):
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        cfg = MorphConfig.from_dict(ck["model_config"])
        model = MorphForCausalLM(cfg)
        model.load_state_dict(ck["model"])
    else:
        from safetensors.torch import load_file
        cfg = MorphConfig.from_json(args.config)
        model = MorphForCausalLM(cfg)
        sd = {k: v.float() for k, v in load_file(args.weights).items()}
        model.load_state_dict(sd, strict=not cfg.tie_embeddings)
    return model.to(device)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--eval_tokens", type=int, default=2_000_000)
    ap.add_argument("--diag_tokens", type=int, default=65_536)
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args, device)
    ds = ShardSet(args.data_dir, "val", args.seq_len)
    mbs = val_batches(ds, max(1, args.eval_tokens // args.seq_len), args.batch)
    diag = torch.from_numpy(np.stack([ds.window(w)[:-1] for w in range(max(1, args.diag_tokens // args.seq_len))]))
    print(json.dumps(evaluate(model, mbs, diag, device, pick_precision(device)), indent=2))


if __name__ == "__main__":
    main()
