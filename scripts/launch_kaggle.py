"""Push MORPH jobs to Kaggle as private script kernels via the local `kaggle` CLI.

Every kernel clones this repo at a pinned commit (must already be pushed to GitHub), runs
one job, and leaves its artifacts in /kaggle/working (the kernel output).

    python scripts/launch_kaggle.py prep                     # CPU: tokenize FineWeb-Edu -> shards
    python scripts/launch_kaggle.py gate                     # T4x2: backend parity + throughput
    python scripts/launch_kaggle.py proxy R0 R1              # T4x2: two proxy runs, one per GPU
    python scripts/launch_kaggle.py main                     # T4x2: main run session (resumes from HF Hub)
    python scripts/launch_kaggle.py status <job>             # kernel status
    python scripts/launch_kaggle.py fetch <job> [dir]        # download output + log

Kernel slugs: morph-<job>[-<runs>]. HF Hub checkpoints need a Kaggle secret HF_TOKEN attached
to the kernel (one-time, in the Kaggle editor); without it runs keep checkpoints in the output.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

KAGGLE = os.path.expanduser("~/.local/bin/kaggle")
REPO_URL = "https://github.com/dhairya-pandya/MoRPH.git"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_KERNEL = "morph-data-prep"

# proxy ablation grid (XS, 0.5B tokens); values are MorphConfig overrides
PROXY_RUNS = {
    "R0": ["self_model_probe=true"],                                   # baseline: readout probe only
    "R1": ["self_model_lambda=0.1"],                                   # paper-faithful
    "R2": ["self_model_lambda=1.0"],                                   # paper-faithful, strong
    "R3": ["self_model_lambda=1.0", "self_model_detach=true"],         # mechanism control
    "R4": ["self_model_probe=true", "kv_lora_rank=32"],                # half latent, baseline
    "R5": ["self_model_lambda=0.1", "kv_lora_rank=32"],                # half latent + self-modeling
}

HEADER = r'''
import glob, json, os, subprocess, sys, threading, time
SHA = {sha!r}
SRC = "/tmp/MoRPH"
W = "/kaggle/working"

def sh(cmd):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)

sh(f"git clone -q {repo!r} {{SRC}} && cd {{SRC}} && git checkout -q {{SHA}}")
os.environ["PYTHONPATH"] = SRC
os.environ["MORPH_GIT_SHA"] = SHA
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    print("HF_TOKEN secret: available", flush=True)
except Exception as e:
    print("HF_TOKEN secret: not attached (%s)" % type(e).__name__, flush=True)
sh("nvidia-smi || true")
sh("python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count())'")

def data_dir():
    hits = glob.glob("/kaggle/input/**/meta.json", recursive=True)
    if not hits:
        raise SystemExit("no prepared data found under /kaggle/input")
    return os.path.dirname(hits[0])
'''

JOBS = {
    "prep": r'''
sh(f"cd {SRC} && python -m morph.data.prepare --out {W}/data --cache_dir /tmp/hf --train_tokens 5.2e9")
''',
    "gate": r'''
sh("pip install -q flash-linear-attention || true")
sh(f"cd {SRC} && python -m morph.tools.gpu_gate --out {W}/gate")
''',
    "proxy": r'''
RUNS = {runs!r}
DATA = data_dir()
os.makedirs(f"{{W}}/logs", exist_ok=True)
procs = []
for gpu, (name, msets) in enumerate(RUNS.items()):
    cmd = [sys.executable, "-m", "morph.train.pretrain", "--model_config", f"{{SRC}}/configs/model/xs.json",
           "--train_config", f"{{SRC}}/configs/train/proxy_xs.json",
           "--set", f"run_name=proxy_{{name}}", "--set", f"data_dir={{DATA}}", "--set", f"out_dir={{W}}/runs",
           "--set", "keep_ckpts=1", "--set", "time_limit_hours=11.6", "--set", "reserve_minutes=12"]
    for m in msets:
        cmd += ["--mset", m]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu % 2))
    log = open(f"{{W}}/logs/{{name}}.log", "w")
    procs.append((name, subprocess.Popen(cmd, cwd=SRC, env=env, stdout=log, stderr=subprocess.STDOUT)))
while any(p.poll() is None for _, p in procs):
    time.sleep(600)
    for name, _ in procs:
        lines = open(f"{{W}}/logs/{{name}}.log").read().splitlines()
        print(f"[{{name}}]", lines[-1] if lines else "", flush=True)
for name, p in procs:
    print(f"=== {{name}} exit {{p.returncode}}", flush=True)
    print("\n".join(open(f"{{W}}/logs/{{name}}.log").read().splitlines()[-40:]), flush=True)
''',
    "profile": r'''
for extra in ["", "--mset ssm_chunk=128", "--mset ssm_chunk=32", "--mset self_model_enabled=false"]:
    sh(f"cd {SRC} && python -m morph.tools.profile_step --config configs/model/xs.json --micro 8 {extra}")
sh(f"cd {SRC} && python -m morph.tools.profile_step --config configs/model/s.json --micro 8")
''',
    "main": r'''
DATA = data_dir()
sh(f"cd {{SRC}} && torchrun --nproc_per_node=2 -m morph.train.pretrain --model_config configs/model/s.json "
   f"--train_config configs/train/main_s.json --set data_dir={{DATA}} --set out_dir={{W}}/runs "
   f"--set time_limit_hours=11.6 --set reserve_minutes=15")
''',
}


def head_sha() -> str:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    remote = subprocess.check_output(["git", "ls-remote", REPO_URL], cwd=ROOT).decode()
    if sha not in remote:
        raise SystemExit(f"HEAD {sha[:10]} is not on GitHub yet — push first")
    return sha


def username() -> str:
    with open(os.path.expanduser("~/.kaggle/kaggle.json")) as f:
        return json.load(f)["username"]


def slug(job: str, runs=()) -> str:
    return DATA_KERNEL if job == "prep" else "morph-" + "-".join([job] + [r.lower() for r in runs])


def push(job: str, runs=()):
    sha = head_sha()
    user = username()
    s = slug(job, runs)
    body = HEADER.format(sha=sha, repo=REPO_URL)
    if job == "proxy":
        body += JOBS["proxy"].format(runs={r: PROXY_RUNS[r] for r in runs})
    elif job == "main":
        body += JOBS["main"].format()
    else:
        body += JOBS[job]
    meta = {
        "id": f"{user}/{s}", "title": s, "code_file": "job.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_internet": True,
        "enable_gpu": job != "prep", "enable_tpu": False,
        "dataset_sources": [], "competition_sources": [], "model_sources": [],
        "kernel_sources": [] if job in ("prep", "gate", "profile") else [f"{user}/{DATA_KERNEL}"],
    }
    if job != "prep":
        meta["machine_shape"] = "NvidiaTeslaT4"
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "job.py"), "w") as f:
            f.write(body)
        with open(os.path.join(d, "kernel-metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)
        subprocess.run([KAGGLE, "kernels", "push", "-p", d], check=True)
    print(f"pushed {user}/{s} @ {sha[:10]}")


def main(argv):
    if not argv:
        raise SystemExit(__doc__)
    cmd, rest = argv[0], argv[1:]
    if cmd in ("prep", "gate", "main", "profile"):
        push(cmd)
    elif cmd == "proxy":
        if not rest or any(r not in PROXY_RUNS for r in rest) or len(rest) > 2:
            raise SystemExit(f"proxy needs 1-2 of {list(PROXY_RUNS)}")
        push("proxy", rest)
    elif cmd == "status":
        subprocess.run([KAGGLE, "kernels", "status", f"{username()}/{rest[0]}"])
    elif cmd == "fetch":
        out = rest[1] if len(rest) > 1 else os.path.join(ROOT, "kaggle_out", rest[0])
        os.makedirs(out, exist_ok=True)
        subprocess.run([KAGGLE, "kernels", "output", f"{username()}/{rest[0]}", "-p", out])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
