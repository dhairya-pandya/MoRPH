"""Stage-1 pretraining: resumable, session-capped, 1..N GPUs (torchrun), T4-fp16 or bf16.

    # single process (CPU smoke with random tokens)
    PYTHONPATH=. python -m morph.train.pretrain --model_config configs/model/xs.json \
        --train_config configs/train/proxy_xs.json --set synthetic=true --set max_steps=4 ...
    # Kaggle T4x2
    PYTHONPATH=. torchrun --nproc_per_node=2 -m morph.train.pretrain --model_config configs/model/s.json \
        --train_config configs/train/main_s.json --set data_dir=/kaggle/input/... --set time_limit_hours=11.5

Loop per optimizer step: WSD LR + self-model lambda ramp -> gradient accumulation over
micro-batches (DDP no_sync) -> unscale/clip -> Muon + AdamW step. Checkpoints every
`ckpt_minutes`, at the session deadline, on SIGTERM/SIGINT, and at the end; the newest
checkpoint (local, extra dirs or HF Hub) is resumed automatically.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from morph.data.shards import ShardSet, SyntheticSet, WindowSampler, micro_batches, val_batches
from morph.eval.diagnostics import state_diagnostics
from morph.model import MorphConfig, MorphForCausalLM
from morph.train import distributed as D
from morph.train.checkpoint import Hub, ckpt_name, find_resume, prune_local, save_atomic
from morph.train.config import TrainConfig, apply_overrides
from morph.train.optim import build_optimizers, set_lr, split_params
from morph.train.schedule import decay_start_step, sm_lambda, wsd_lr

MAX_CONSECUTIVE_SKIPS = 20


def git_sha() -> str:
    if os.environ.get("MORPH_GIT_SHA"):
        return os.environ["MORPH_GIT_SHA"]
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                                       cwd=os.path.dirname(__file__)).decode().strip()
    except Exception:
        return "unknown"


def autocast_ctx(device, amp_dtype):
    return torch.autocast(device.type, dtype=amp_dtype) if amp_dtype is not None else nullcontext()


def probe_micro_batch(model, per_rank: int, tcfg, device, amp_dtype) -> int:
    """Largest micro-batch dividing the per-rank batch whose fwd+bwd (+ optimizer state) fits in ~88% of memory."""
    divisors = [c for c in (32, 16, 8, 4, 2, 1) if per_rank % c == 0]
    if device.type != "cuda":
        return next(c for c in divisors if c <= 4)
    muon, rest = split_params(model)
    opt_bytes = 4 * sum(p.numel() for p in muon) + 8 * sum(p.numel() for p in rest)
    total = torch.cuda.get_device_properties(device).total_memory
    vocab = model.cfg.vocab_size
    for c in divisors:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            x = torch.randint(0, vocab, (c, tcfg.seq_len), device=device)
            with autocast_ctx(device, amp_dtype):
                out = model(x, x)
            out["loss"].backward()
            peak = torch.cuda.max_memory_allocated(device)
            model.zero_grad(set_to_none=True)
            del out, x
            if peak + opt_bytes < 0.88 * total:
                return c
        except torch.OutOfMemoryError:
            model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    raise RuntimeError("even micro_batch=1 does not fit; lower seq_len or enable grad_ckpt")


class MetricsLog:
    def __init__(self, path: str, enabled: bool, wandb_run=None):
        self.path, self.enabled, self.wandb = path, enabled, wandb_run

    def write(self, rec: dict):
        if not self.enabled:
            return
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if self.wandb is not None:
            self.wandb.log({f"{rec['type']}/{k}": v for k, v in rec.items() if isinstance(v, (int, float))},
                           step=rec.get("step"))


@torch.no_grad()
def evaluate(model, val_mbs, diag_ids, device, amp_dtype) -> dict:
    was = model.training
    model.eval()
    tot, n = 0.0, 0
    for mb in val_mbs:
        mb = mb.to(device)
        with autocast_ctx(device, amp_dtype):
            out = model(mb[:, :-1], mb[:, 1:])
        tot += float(out["ce"]) * mb.size(0)
        n += mb.size(0)
    res = {"val_ce": tot / max(1, n)}
    res["val_ppl"] = float(torch.tensor(res["val_ce"]).exp())
    res.update(state_diagnostics(model, diag_ids.to(device), amp_dtype))
    model.train(was)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_config", required=True)
    ap.add_argument("--train_config", required=True)
    ap.add_argument("--set", action="append", default=[], help="TrainConfig override key=value")
    ap.add_argument("--mset", action="append", default=[], help="MorphConfig override key=value")
    args = ap.parse_args(argv)

    info = D.setup()
    device = info.device
    mcfg = apply_overrides(MorphConfig.from_json(args.model_config), args.mset)
    tcfg = apply_overrides(TrainConfig.from_json(args.train_config), args.set)
    run_dir = os.path.join(tcfg.out_dir, tcfg.run_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    if info.is_main:
        os.makedirs(ckpt_dir, exist_ok=True)
    D.barrier(info)
    torch.manual_seed(tcfg.seed)
    amp_dtype = D.pick_precision(device, tcfg.precision)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ---- data ----
    if tcfg.synthetic:
        train_ds = SyntheticSet(mcfg.vocab_size, tcfg.seq_len, seed=tcfg.seed)
        val_ds = SyntheticSet(mcfg.vocab_size, tcfg.seq_len, n_windows=4096, seed=tcfg.seed + 1)
    else:
        train_ds = ShardSet(tcfg.data_dir, "train", tcfg.seq_len)
        val_ds = ShardSet(tcfg.data_dir, "val", tcfg.seq_len)
        if train_ds.meta.get("vocab_size", 0) > mcfg.vocab_size:
            raise ValueError(f"data vocab {train_ds.meta['vocab_size']} > model vocab {mcfg.vocab_size}")
    sampler = WindowSampler(train_ds.n_windows, tcfg.global_batch, tcfg.seed)
    if tcfg.global_batch % info.world:
        raise ValueError(f"global_batch {tcfg.global_batch} not divisible by world size {info.world}")
    per_rank = tcfg.global_batch // info.world

    # ---- model / optimizers ----
    model = MorphForCausalLM(mcfg).to(device)
    model.set_grad_checkpointing(tcfg.grad_ckpt)
    micro = tcfg.micro_batch or probe_micro_batch(model, per_rank, tcfg, device, amp_dtype)
    if per_rank % micro:
        raise ValueError(f"per-rank batch {per_rank} not divisible by micro_batch {micro}")
    accum = per_rank // micro
    optimizers = build_optimizers(model, tcfg)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

    hub = None
    if info.is_main and tcfg.hub_repo:
        if os.environ.get("HF_TOKEN"):
            hub = Hub(tcfg.hub_repo, tcfg.run_name, tcfg.hub_keep)
        else:
            print("[pretrain] hub_repo set but HF_TOKEN missing: checkpoints stay local", flush=True)

    # ---- resume ----
    step, tokens = 0, 0
    path = find_resume(ckpt_dir, tcfg.resume_dirs, hub) if info.is_main else None
    if info.enabled:
        box = [path]
        dist.broadcast_object_list(box, src=0)
        path = box[0]
    if path:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if ck["arch_hash"] != mcfg.arch_hash():
            raise ValueError(f"checkpoint {path} has a different architecture ({ck['arch_hash']} != {mcfg.arch_hash()})")
        model.load_state_dict(ck["model"])
        for opt, sd in zip(optimizers, ck["optimizers"]):
            opt.load_state_dict(sd)
        if ck.get("scaler") and scaler.is_enabled():
            scaler.load_state_dict(ck["scaler"])
        step, tokens = ck["step"], ck["tokens"]
        diff = {k: (ck["train_config"].get(k), v) for k, v in tcfg.to_dict().items() if ck["train_config"].get(k) != v}
        if info.is_main:
            print(f"[pretrain] resumed {path} @ step {step} ({tokens / 1e9:.3f}B tokens); config changes: {diff}", flush=True)
            old_total = TrainConfig.from_dict(ck["train_config"]).total_steps
            if step >= decay_start_step(old_total, tcfg.decay_frac) and tcfg.total_steps != old_total:
                print("[pretrain] WARNING: budget changed after LR decay began; schedule will jump", flush=True)
        del ck

    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    if info.is_main and hub is not None and not os.path.exists(metrics_path) and step > 0:
        got = hub.download(f"{tcfg.run_name}/metrics.jsonl", tcfg.out_dir)
        if got and os.path.abspath(got) != os.path.abspath(metrics_path):
            os.replace(got, metrics_path)
    wandb_run = None
    if info.is_main and tcfg.wandb and os.environ.get("WANDB_API_KEY"):
        import wandb
        wandb_run = wandb.init(project="morph", name=tcfg.run_name, id=tcfg.run_name, resume="allow",
                               config={"model": mcfg.to_dict(), "train": tcfg.to_dict()})
    log = MetricsLog(metrics_path, info.is_main, wandb_run)

    if tcfg.compile:
        for layer in model.model.layers:
            layer.compile()
    ddp = DDP(model, device_ids=[info.local_rank] if device.type == "cuda" else None) if info.enabled else model

    # fixed eval data (rank 0)
    val_mbs = val_batches(val_ds, max(1, tcfg.eval_tokens // tcfg.seq_len), micro) if info.is_main else []
    n_diag = max(1, tcfg.diag_tokens // tcfg.seq_len)
    diag_ids = torch.from_numpy(np.stack([val_ds.window(w)[:-1] for w in range(n_diag)])) if info.is_main else None

    total_steps = tcfg.total_steps
    end_step = min(total_steps, step + tcfg.max_steps) if tcfg.max_steps else total_steps
    flops_per_token = 6 * model.num_params()
    peak = D.peak_flops(device)
    if info.is_main:
        print(f"[pretrain] {tcfg.run_name}: {model.num_params() / 1e6:.1f}M params | device={device} "
              f"{torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu'} x{info.world} | "
              f"amp={amp_dtype} | micro={micro} accum={accum} global={tcfg.global_batch}x{tcfg.seq_len} | "
              f"steps {step}->{end_step}/{total_steps} | backend={mcfg.ssm_backend} ckpt={tcfg.grad_ckpt} "
              f"compile={tcfg.compile}", flush=True)

    stop = {"signal": False}

    def _on_signal(signum, frame):
        stop["signal"] = True
    for s in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(s, _on_signal)
        except ValueError:
            pass

    t_start = time.time()
    deadline = t_start + tcfg.time_limit_hours * 3600 - tcfg.reserve_minutes * 60 if tcfg.time_limit_hours else None
    last_ckpt = time.time()
    t_log, tok_log = time.time(), tokens
    next_milestone = (int(tokens // tcfg.milestone_tokens) + 1) * tcfg.milestone_tokens if tcfg.milestone_tokens else None
    skips = 0
    sums = torch.zeros(4, device=device)
    gnorm = torch.tensor(0.0)

    def save(tag: str):
        if not info.is_main:
            return
        state = {
            "model": model.state_dict(), "optimizers": [o.state_dict() for o in optimizers],
            "scaler": scaler.state_dict() if scaler.is_enabled() else None,
            "step": step, "tokens": tokens, "rng": torch.get_rng_state(),
            "model_config": mcfg.to_dict(), "train_config": tcfg.to_dict(),
            "arch_hash": mcfg.arch_hash(), "git_sha": git_sha(), "tag": tag,
        }
        p = os.path.join(ckpt_dir, ckpt_name(step))
        save_atomic(state, p)
        prune_local(ckpt_dir, tcfg.keep_ckpts)
        if hub is not None:
            hub.prune(keep=max(0, tcfg.hub_keep - 1))
            hub.upload(p)
            if os.path.exists(metrics_path):
                hub.upload(metrics_path)
        print(f"[pretrain] saved {p} ({tag})", flush=True)

    def save_weights(label: str):
        if not info.is_main:
            return
        from safetensors.torch import save_file
        sd = {k: v.detach().to(torch.float16).contiguous().cpu() for k, v in model.state_dict().items()
              if not (mcfg.tie_embeddings and k == "lm_head.weight")}
        p = os.path.join(run_dir, "milestones", f"weights_{label}.safetensors")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        save_file(sd, p, metadata={"model_config": json.dumps(mcfg.to_dict()), "tokens": str(tokens)})
        if hub is not None:
            hub.upload(p, f"milestones/{os.path.basename(p)}")

    model.train()
    while step < end_step:
        lr = wsd_lr(step, total_steps, tcfg.warmup_steps, tcfg.decay_frac, tcfg.lr, tcfg.min_lr_frac)
        set_lr(optimizers, lr)
        lam = sm_lambda(step, tcfg.warmup_steps, mcfg.self_model_lambda)
        mbs = micro_batches(train_ds, sampler, step, info.rank, info.world, micro)
        sums.zero_()
        for i, mb in enumerate(mbs):
            mb = mb.to(device, non_blocking=True)
            sync = ddp.no_sync() if (info.enabled and i < len(mbs) - 1) else nullcontext()
            with sync:
                with autocast_ctx(device, amp_dtype):
                    out = ddp(mb[:, :-1], mb[:, 1:], sm_lambda=lam)
                scaler.scale(out["loss"] / accum).backward()
            sums += torch.stack([out["loss"].detach().float(), out["ce"].float(),
                                 out.get("sm_loss", torch.zeros((), device=device)).float(),
                                 out["z_loss"].float()]) / accum
        for opt in optimizers:
            scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        finite = bool(torch.isfinite(gnorm))
        if scaler.is_enabled():
            for opt in optimizers:
                scaler.step(opt)
            scaler.update()
        elif finite:
            for opt in optimizers:
                opt.step()
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        skips = 0 if finite else skips + 1
        if skips >= MAX_CONSECUTIVE_SKIPS:
            raise RuntimeError(f"{skips} consecutive non-finite steps at step {step}; lower lr or check data")
        step += 1
        tokens += tcfg.tokens_per_step

        if step % tcfg.log_every == 0 or step == end_step:
            vals = D.all_mean(info, sums).tolist()
            now = time.time()
            tok_s = (tokens - tok_log) / max(1e-9, now - t_log)
            t_log, tok_log = now, tokens
            rec = {"type": "train", "step": step, "tokens": tokens, "loss": vals[0], "ce": vals[1],
                   "sm": vals[2], "z": vals[3], "lr": lr, "lam": lam, "gnorm": float(gnorm),
                   "scale": float(scaler.get_scale()) if scaler.is_enabled() else 1.0,
                   "tok_s": tok_s, "elapsed_h": (now - t_start) / 3600}
            if peak:
                rec["mfu"] = flops_per_token * tok_s / (peak * info.world)
            if device.type == "cuda":
                rec["mem_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
            log.write(rec)
            if info.is_main:
                print(f"step {step}/{total_steps} loss {vals[0]:.4f} ce {vals[1]:.4f} sm {vals[2]:.4f} "
                      f"lr {lr:.2e} lam {lam:.3f} gn {float(gnorm):.2f} {tok_s / 1e3:.1f}k tok/s "
                      f"{tokens / 1e9:.3f}B" + (f" mfu {rec['mfu']:.1%}" if 'mfu' in rec else ""), flush=True)

        if step % tcfg.eval_every == 0 or step == end_step:
            if info.is_main:
                res = evaluate(model, val_mbs, diag_ids, device, amp_dtype)
                log.write({"type": "eval", "step": step, "tokens": tokens, **res})
                print(f"[eval] step {step} val_ce {res['val_ce']:.4f} ppl {res['val_ppl']:.2f} " +
                      " ".join(f"{k} {v:.3f}" for k, v in res.items() if k.startswith(("erank", "sm_r2"))), flush=True)
            D.barrier(info)

        if next_milestone is not None and tokens >= next_milestone:
            save_weights(f"{tokens / 1e9:.2f}B")
            next_milestone += tcfg.milestone_tokens

        want_stop = stop["signal"] or (deadline is not None and time.time() > deadline)
        want_ckpt = (time.time() - last_ckpt) > tcfg.ckpt_minutes * 60
        want_stop = D.any_flag(info, want_stop)
        if D.any_flag(info, want_ckpt) or want_stop:
            save("deadline" if want_stop else "periodic")
            last_ckpt = time.time()
            D.barrier(info)
        if want_stop:
            if info.is_main:
                print(f"[pretrain] stopping at step {step} (session limit/signal); resume by relaunching", flush=True)
            break
    else:
        save("final" if step >= total_steps else "max_steps")
        if step >= total_steps:
            save_weights("final")

    if hub is not None:
        hub.wait()
    if wandb_run is not None:
        wandb_run.finish()
    D.barrier(info)
    D.teardown(info)


if __name__ == "__main__":
    main()
