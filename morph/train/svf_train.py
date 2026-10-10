"""Stage 3: train SVF experts, a prompt router, and evaluate two-pass adaptation.

    # one expert per domain: only the singular-value scales z train, the base stays frozen
    python -m morph.train.svf_train expert --base CKPT --domain math=DIR --out experts/math.safetensors
    # pass-1 router: domain probabilities from the mean-pooled prompt hidden state
    python -m morph.train.svf_train router --base CKPT --domains general=DIR math=DIR ... --out router.pt
    # two-pass evaluation: base / each expert / oracle / router-mixed z, per domain
    python -m morph.train.svf_train evaluate --base CKPT --domains ... --experts math=PATH ... --router router.pt

Experts are trained supervised (next-token CE on the domain). The paper uses RL on task
rewards; a 44M–170M base model has no instruction-following behaviour to reward, so the
supervised variant is the meaningful one here. A "general" domain has no trained expert:
its z is the identity (the base model), so the router can always fall back to the base.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import nullcontext
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from morph.data.shards import ShardSet, WindowSampler
from morph.eval.quality_bench import load_model
from morph.model.svf import (apply_svf, get_expert, identity_expert, load_expert, mix_experts,
                             n_expert_params, save_expert, set_expert, svf_modules)
from morph.train.distributed import pick_precision


def _pairs(items: List[str]) -> Dict[str, str]:
    out = {}
    for it in items:
        k, _, v = it.partition("=")
        if not v:
            raise ValueError(f"expected name=path, got {it!r}")
        out[k] = v
    return out


def _autocast(device, amp):
    return torch.autocast(device.type, dtype=amp) if amp is not None else nullcontext()


def load_base(ckpt: str, device):
    """Base checkpoint with SVF applied; the self-model head is dropped (not part of adaptation)."""
    model = load_model(argparse.Namespace(ckpt=ckpt, weights=None, config=None), device)
    model.self_model = None
    names = apply_svf(model)
    model.to(device)
    return model, names


def _split(ddir: str, prefer: str, seq: int) -> ShardSet:
    """The preferred split, falling back to val when the preferred one has no windows."""
    try:
        return ShardSet(ddir, prefer, seq)
    except (ValueError, KeyError):
        return ShardSet(ddir, "val", seq)


def windows(ds: ShardSet, n: int, seed: int) -> List[int]:
    rng = np.random.default_rng(seed)
    return rng.choice(ds.n_windows, size=min(n, ds.n_windows), replace=False).tolist()


def batch(ds: ShardSet, ids: List[int], device) -> torch.Tensor:
    return torch.from_numpy(np.stack([ds.window(w) for w in ids])).to(device)


@torch.no_grad()
def eval_ce(model, ds: ShardSet, ids: List[int], device, amp, bs: int, skip: int = 0) -> float:
    """Mean CE over windows, scoring only targets after the first `skip` positions (the pass-1 prompt)."""
    model.eval()
    tot = n = 0.0
    for i in range(0, len(ids), bs):
        mb = batch(ds, ids[i:i + bs], device)
        y = mb[:, 1:].clone()
        y[:, :skip] = -100
        with _autocast(device, amp):
            out = model(mb[:, :-1], y)
        k = float((y != -100).sum())
        tot += float(out["ce"]) * k
        n += k
    return tot / max(1.0, n)


# ---------------------------------------------------------------- experts
def cmd_expert(a):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = pick_precision(device)
    torch.manual_seed(a.seed)
    model, names = load_base(a.base, device)
    model.set_grad_checkpointing(True)
    name, ddir = next(iter(_pairs([a.domain]).items()))
    train = ShardSet(ddir, "train", a.seq)
    val = ShardSet(ddir, "val", a.seq)
    val_ids = windows(val, a.eval_windows, 1)
    gen = ShardSet(a.general, "val", a.seq) if a.general else None
    gen_ids = windows(gen, a.eval_windows, 2) if gen else []
    zs = [m.z for m in svf_modules(model).values()]
    opt = torch.optim.AdamW(zs, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp == torch.float16))
    sampler = WindowSampler(train.n_windows, a.batch, a.seed)
    base_ce = eval_ce(model, val, val_ids, device, amp, a.micro)
    print(f"[svf] expert {name}: {len(names)} matrices, {n_expert_params(model) / 1e3:.1f}k trainable "
          f"| base val CE {base_ce:.4f}", flush=True)
    log = []
    t0 = time.time()
    for step in range(a.steps):
        lr = a.lr * min(1.0, (step + 1) / max(1, a.warmup))
        for g in opt.param_groups:
            g["lr"] = lr
        model.train()
        ids = sampler.step_ids(step)
        for i in range(0, len(ids), a.micro):
            mb = batch(train, ids[i:i + a.micro], device)
            with _autocast(device, amp):
                out = model(mb[:, :-1], mb[:, 1:])
                reg = sum(((z - 1.0) ** 2).mean() for z in zs) / len(zs)
                loss = (out["loss"] + a.z_reg * reg) * (a.micro / len(ids))
            scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(zs, 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        if (step + 1) % a.eval_every == 0 or step + 1 == a.steps:
            rec = {"step": step + 1, "train_ce": float(out["ce"]),
                   "val_ce": eval_ce(model, val, val_ids, device, amp, a.micro),
                   "elapsed_min": (time.time() - t0) / 60}
            if gen:
                rec["general_ce"] = eval_ce(model, gen, gen_ids, device, amp, a.micro)
            rec["z_mean"] = float(torch.cat([z.detach().flatten() for z in zs]).mean())
            log.append(rec)
            print("[svf]", json.dumps(rec), flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    save_expert(a.out, get_expert(model), {"domain": name, "base": a.base, "base_val_ce": base_ce, "log": log})
    print(f"[svf] saved {a.out}", flush=True)


# ---------------------------------------------------------------- router
class Router(nn.Module):
    """Pass 1: domain logits from the mean-pooled final hidden state of the prompt (base model, z = 1)."""

    def __init__(self, d: int, domains: List[str]):
        super().__init__()
        self.domains = list(domains)
        self.lin = nn.Linear(d, len(domains))

    def forward(self, h):
        return self.lin(h)


@torch.no_grad()
def prompt_features(model, ds: ShardSet, ids: List[int], prompt: int, device, amp, bs: int) -> torch.Tensor:
    set_expert(model, None)
    model.eval()
    feats = []
    for i in range(0, len(ids), bs):
        mb = batch(ds, ids[i:i + bs], device)[:, :prompt]
        with _autocast(device, amp):
            hidden, _, _ = model.model(mb)
        feats.append(hidden.float().mean(1).cpu())
    return torch.cat(feats)


def cmd_router(a):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = pick_precision(device)
    model, _ = load_base(a.base, device)
    domains = _pairs(a.domains)
    X, Y, Xv, Yv = [], [], [], []
    for k, (name, ddir) in enumerate(domains.items()):
        ds_tr = _split(ddir, "train", a.prompt)
        ds_va = ShardSet(ddir, "val", a.prompt)
        X.append(prompt_features(model, ds_tr, windows(ds_tr, a.n_train, 10 + k), a.prompt, device, amp, a.micro))
        Xv.append(prompt_features(model, ds_va, windows(ds_va, a.n_val, 20 + k), a.prompt, device, amp, a.micro))
        Y.append(torch.full((len(X[-1]),), k))
        Yv.append(torch.full((len(Xv[-1]),), k))
    X, Y, Xv, Yv = torch.cat(X), torch.cat(Y), torch.cat(Xv), torch.cat(Yv)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    router = Router(X.size(1), list(domains))
    opt = torch.optim.AdamW(router.parameters(), lr=1e-2, weight_decay=1e-2)
    for _ in range(a.steps):
        loss = F.cross_entropy(router((X - mu) / sd), Y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = float((router((Xv - mu) / sd).argmax(-1) == Yv).float().mean())
    torch.save({"state": router.state_dict(), "domains": list(domains), "mu": mu, "sd": sd,
                "prompt": a.prompt, "val_acc": acc}, a.out)
    print(f"[svf] router {list(domains)} held-out accuracy {acc:.3f} -> {a.out}", flush=True)


def load_router(path: str):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    r = Router(blob["mu"].numel(), blob["domains"])
    r.load_state_dict(blob["state"])
    return r, blob


# ---------------------------------------------------------------- evaluate
def cmd_evaluate(a):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = pick_precision(device)
    model, _ = load_base(a.base, device)
    domains = _pairs(a.domains)
    paths = _pairs(a.experts)
    experts = {n: load_expert(p) for n, p in paths.items()}
    router, rb = load_router(a.router)
    bank = [experts.get(d, identity_expert(model)) for d in rb["domains"]]   # general -> identity
    P = rb["prompt"]
    res: Dict[str, Dict[str, float]] = {}
    for k, (dom, ddir) in enumerate(domains.items()):
        ds = ShardSet(ddir, "val", a.seq)
        ids = windows(ds, a.n_eval, 100 + k)
        row = {}
        set_expert(model, None)
        row["base"] = eval_ce(model, ds, ids, device, amp, a.micro, skip=P)
        for en, ex in experts.items():
            set_expert(model, ex)
            row[f"expert:{en}"] = eval_ce(model, ds, ids, device, amp, a.micro, skip=P)
        row["oracle"] = row.get(f"expert:{dom}", row["base"])
        # two-pass: per window, pass 1 routes on the first P tokens, pass 2 scores the rest with the mix
        feats = prompt_features(model, ds, ids, P, device, amp, a.micro)
        with torch.no_grad():
            probs = F.softmax(router((feats - rb["mu"]) / rb["sd"]), -1)
        tot = n = 0.0
        hits = 0
        for i, w in enumerate(ids):
            set_expert(model, mix_experts(bank, probs[i].tolist()))
            ce = eval_ce(model, ds, [w], device, amp, 1, skip=P)
            k_tok = a.seq - P
            tot += ce * k_tok
            n += k_tok
            hits += int(rb["domains"][int(probs[i].argmax())] == dom)
        row["two_pass"] = tot / n
        row["route_acc"] = hits / len(ids)
        res[dom] = row
        print(f"[svf] {dom}: " + " ".join(f"{k} {v:.4f}" for k, v in row.items()), flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("expert")
    e.add_argument("--base", required=True)
    e.add_argument("--domain", required=True, help="name=shard_dir")
    e.add_argument("--general", default="", help="shard dir whose val split measures forgetting")
    e.add_argument("--out", required=True)
    e.add_argument("--steps", type=int, default=300)
    e.add_argument("--batch", type=int, default=16)
    e.add_argument("--micro", type=int, default=8)
    e.add_argument("--seq", type=int, default=1024)
    e.add_argument("--lr", type=float, default=2e-3)
    e.add_argument("--warmup", type=int, default=20)
    e.add_argument("--z_reg", type=float, default=0.0, help="weight of mean (z-1)^2 (stay close to the base)")
    e.add_argument("--eval_every", type=int, default=50)
    e.add_argument("--eval_windows", type=int, default=64)
    e.add_argument("--seed", type=int, default=0)
    r = sub.add_parser("router")
    r.add_argument("--base", required=True)
    r.add_argument("--domains", nargs="+", required=True, help="name=shard_dir ...")
    r.add_argument("--out", required=True)
    r.add_argument("--prompt", type=int, default=128)
    r.add_argument("--n_train", type=int, default=512)
    r.add_argument("--n_val", type=int, default=128)
    r.add_argument("--micro", type=int, default=16)
    r.add_argument("--steps", type=int, default=300)
    v = sub.add_parser("evaluate")
    v.add_argument("--base", required=True)
    v.add_argument("--domains", nargs="+", required=True)
    v.add_argument("--experts", nargs="+", required=True)
    v.add_argument("--router", required=True)
    v.add_argument("--seq", type=int, default=1024)
    v.add_argument("--n_eval", type=int, default=64)
    v.add_argument("--micro", type=int, default=8)
    v.add_argument("--out", default="")
    a = ap.parse_args(argv)
    {"expert": cmd_expert, "router": cmd_router, "evaluate": cmd_evaluate}[a.cmd](a)


if __name__ == "__main__":
    main()
