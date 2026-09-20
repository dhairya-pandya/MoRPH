"""Stage-1 pretraining: from-scratch, resumable, session-time-capped.

Designed for Kaggle/Colab: trains until either the step target OR the wall-clock cap
(minus a safety margin) is hit, then checkpoints cleanly and exits so the next
session resumes. Loss = LM + lambda * self-modeling (handled inside the model).

Smoke run (CPU, synthetic data, no deps beyond torch):
    PYTHONPATH=. python -m morph.train.pretrain --config configs/300m_hybrid.json \
        --smoke --steps 20 --batch 2 --seq 128
"""
from __future__ import annotations

import argparse
import math
import os
import time

import torch

from morph.model import MorphConfig, MorphForCausalLM
from morph.data.fineweb import PackedTextStream, SyntheticStream
from morph.train.checkpoint import (
    save_checkpoint, load_checkpoint, latest_local, push_to_hub, pull_latest_from_hub,
)


def cosine_lr(step, warmup, total, base_lr, min_lr):
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    if step >= total:
        return min_lr
    prog = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * prog))


def get_tokenizer(name: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    return tok


def build_stream(args, cfg, start_doc):
    if args.smoke:
        return SyntheticStream(cfg.vocab_size, args.seq)
    tok = get_tokenizer(args.tokenizer)
    return PackedTextStream(tok, args.seq, start_doc=start_doc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/300m_hybrid.json")
    ap.add_argument("--ckpt_dir", default="checkpoints")
    ap.add_argument("--hub_repo", default=None, help="HF repo id for off-box checkpoints")
    ap.add_argument("--tokenizer", default="meta-llama/Meta-Llama-3-8B")
    ap.add_argument("--steps", type=int, default=200000)      # total optimizer steps
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)           # micro-batch
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min_lr", type=float, default=3e-5)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--session_minutes", type=float, default=0, help="wall-clock cap; 0=off")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = MorphConfig.from_json(args.config)
    if args.smoke:
        cfg.max_seq_len = args.seq
    model = MorphForCausalLM(cfg).to(device)
    params = model.num_params() / 1e6
    print(f"[pretrain] device={device} params={params:.1f}M backend="
          f"{model.model.layers[0]['mixer'].__class__.__name__}")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd,
                              betas=(0.9, 0.95))
    step, tokens, docs = 0, 0, 0

    # ---- resume: prefer hub, then local ----
    os.makedirs(args.ckpt_dir, exist_ok=True)
    resume_path = None
    if args.hub_repo:
        resume_path = pull_latest_from_hub(args.hub_repo, args.ckpt_dir)
    resume_path = resume_path or latest_local(args.ckpt_dir)
    if resume_path:
        ck = load_checkpoint(resume_path, model=model, optimizer=optim, map_location=device)
        step, tokens, docs = ck["step"], ck["tokens"], ck["docs_consumed"]
        print(f"[pretrain] resumed from {resume_path} @ step {step}, {tokens/1e9:.2f}B tokens")

    stream = build_stream(args, cfg, start_doc=docs)
    biter = stream.batches(args.batch)

    use_amp = device == "cuda"
    amp_dtype = torch.bfloat16
    t0 = time.time()
    deadline = t0 + args.session_minutes * 60 * 0.95 if args.session_minutes else None

    model.train()
    while step < args.steps:
        optim.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(args.grad_accum):
            batch = next(biter).to(device)
            ids, labels = batch[:, :-1].contiguous(), batch.clone()
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                out = model(ids, labels=labels[:, :ids.size(1)])
                loss = out["loss"] / args.grad_accum
            loss.backward()
            accum_loss += loss.item()
            tokens += ids.numel()

        lr = cosine_lr(step, args.warmup, args.steps, args.lr, args.min_lr)
        for g in optim.param_groups:
            g["lr"] = lr
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optim.step()
        step += 1

        if step % args.log_every == 0:
            sm = out.get("self_model_loss")
            print(f"step {step} loss {accum_loss:.4f} lm {out.get('lm_loss',0):.4f} "
                  f"sm {float(sm) if sm is not None else 0:.4f} lr {lr:.2e} "
                  f"tok {tokens/1e9:.3f}B")

        if step % args.save_every == 0 or step >= args.steps:
            docs = getattr(stream, "docs_consumed", docs)
            path = os.path.join(args.ckpt_dir, f"step_{step}.pt")
            save_checkpoint(path, model=model, optimizer=optim, scheduler=None,
                            step=step, tokens=tokens, docs_consumed=docs)
            if args.hub_repo:
                push_to_hub(path, args.hub_repo)
            print(f"[pretrain] saved {path}")

        if deadline and time.time() > deadline:
            docs = getattr(stream, "docs_consumed", docs)
            path = os.path.join(args.ckpt_dir, f"step_{step}.pt")
            save_checkpoint(path, model=model, optimizer=optim, scheduler=None,
                            step=step, tokens=tokens, docs_consumed=docs)
            if args.hub_repo:
                push_to_hub(path, args.hub_repo)
            print(f"[pretrain] session cap reached, checkpointed @ step {step}. Resume next session.")
            return

    print(f"[pretrain] done @ step {step}, {tokens/1e9:.2f}B tokens")


if __name__ == "__main__":
    main()
