"""GPU gate: run first on new hardware. Checks precision + SSD backends, then measures the
real training path (short synthetic `pretrain` runs) for throughput and memory.

    PYTHONPATH=. python -m morph.tools.gpu_gate --out gate            # all scenarios
    PYTHONPATH=. python -m morph.tools.gpu_gate --out gate --quick    # XS only

Writes <out>/gate.json and prints projected hours for the proxy and main budgets.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def backend_parity() -> dict:
    from morph.model.ssd import ssd, ssd_chunked
    res = {}
    if not torch.cuda.is_available():
        return res
    torch.manual_seed(0)
    b, l, h, p, n = 2, 512, 8, 64, 128
    dev = "cuda"
    x = torch.randn(b, l, h, p, device=dev, dtype=torch.float16)
    dt = F.softplus(torch.randn(b, l, h, device=dev) - 2)
    A = -torch.rand(h, device=dev) * 4 - 0.5
    B = torch.randn(b, l, n, device=dev, dtype=torch.float16)
    C = torch.randn(b, l, n, device=dev, dtype=torch.float16)
    ref = ssd_chunked(x, dt, A, B, C, 64)[0].float()
    for backend in ("torch", "fla", "mamba_ssm"):
        try:
            torch.cuda.synchronize()
            t = time.time()
            for _ in range(5):
                y = ssd(x, dt, A, B, C, 64, backend).float()
            torch.cuda.synchronize()
            err = float((y - ref).abs().max() / ref.abs().max())
            res[backend] = {"ok": err < 2e-2, "rel_err": err, "ms": (time.time() - t) / 5 * 1e3}
        except Exception as e:
            res[backend] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}
    return res


def run_scenario(name, model_cfg, nproc, sets, out_dir, steps) -> dict:
    run_out = os.path.join(out_dir, "runs")
    common = ["--model_config", model_cfg, "--train_config", os.path.join(ROOT, "configs/train/proxy_xs.json"),
              "--set", "synthetic=true", "--set", f"out_dir={run_out}", "--set", f"run_name={name}",
              "--set", f"max_steps={steps}", "--set", "log_every=5", "--set", "eval_every=100000",
              "--set", "time_limit_hours=0", "--set", "ckpt_minutes=100000"]
    for s in sets:
        common += ["--set", s]
    if nproc > 1:
        cmd = [sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={nproc}", "-m", "morph.train.pretrain"]
    else:
        cmd = [sys.executable, "-m", "morph.train.pretrain"]
    t = time.time()
    r = subprocess.run(cmd + common, cwd=ROOT, env=dict(os.environ, PYTHONPATH=ROOT),
                       capture_output=True, text=True, timeout=3600)
    res = {"name": name, "nproc": nproc, "sets": sets, "wall_s": round(time.time() - t, 1), "rc": r.returncode}
    head = [ln for ln in r.stdout.splitlines() if ln.startswith("[pretrain]")]
    res["header"] = head[0] if head else ""
    if r.returncode != 0:
        err = r.stderr or r.stdout
        i = err.find("Error")
        res["error"] = err[max(0, i - 2500): i + 500] if i >= 0 else err[-3000:]
        return res
    recs = [json.loads(ln) for ln in open(os.path.join(run_out, name, "metrics.jsonl")) if '"train"' in ln]
    last = recs[-1]
    res.update(tok_s=round(last["tok_s"]), mem_gb=round(last.get("mem_gb", 0), 2),
               mfu=round(last.get("mfu", 0), 4), loss=round(last["loss"], 3))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="gate")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--steps", type=int, default=15)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    n_gpu = torch.cuda.device_count()
    report = {"torch": torch.__version__, "cuda": torch.version.cuda, "n_gpu": n_gpu,
              "gpu": torch.cuda.get_device_name(0) if n_gpu else "cpu",
              "capability": torch.cuda.get_device_capability(0) if n_gpu else None}
    print(json.dumps(report), flush=True)
    report["ssd_backends"] = backend_parity()
    print("ssd backends:", json.dumps(report["ssd_backends"]), flush=True)

    xs = os.path.join(ROOT, "configs/model/xs.json")
    s = os.path.join(ROOT, "configs/model/s.json")
    scen = [("xs_ckpt", xs, 1, ["grad_ckpt=true"]),
            ("xs_nockpt", xs, 1, ["grad_ckpt=false"]),
            ("xs_compile", xs, 1, ["grad_ckpt=true", "compile=true"])]
    if not args.quick:
        g = max(1, n_gpu)
        scen += [("s_ckpt", s, g, ["grad_ckpt=true", "global_batch=128"]),
                 ("s_compile", s, g, ["grad_ckpt=true", "compile=true", "global_batch=128"])]
    report["scenarios"] = []
    for name, cfg, nproc, sets in scen:
        r = run_scenario(name, cfg, nproc, sets, args.out, args.steps)
        report["scenarios"].append(r)
        print(json.dumps({k: v for k, v in r.items() if k != "error"}), flush=True)
        if "error" in r:
            print(f"[gate] {name} FAILED:\n{r['error']}", flush=True)
    for r in report["scenarios"]:
        if r.get("tok_s"):
            budget = 5e8 if r["name"].startswith("xs") else 5e9
            print(f"[gate] {r['name']}: {r['tok_s'] / 1e3:.1f}k tok/s, {r['mem_gb']} GB -> "
                  f"{budget / r['tok_s'] / 3600:.1f} h for {budget / 1e9:.1f}B tokens", flush=True)
    with open(os.path.join(args.out, "gate.json"), "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
