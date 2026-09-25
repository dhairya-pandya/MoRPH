"""Trainer gate: WSD, Muon, param coverage, exact resume, DDP (gloo) equivalence."""
import glob
import os
import subprocess
import sys
import tempfile

import torch

from morph.model import MorphConfig, MorphForCausalLM
from morph.train.config import TrainConfig
from morph.train.optim import build_optimizers, newton_schulz, split_params
from morph.train.pretrain import main as pretrain_main
from morph.train.schedule import sm_lambda, wsd_lr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TINY_M = ["--mset", "vocab_size=256", "--mset", "d_model=64", "--mset", "n_heads=2", "--mset", "head_dim=32",
          "--mset", "kv_lora_rank=16", "--mset", "mlp_hidden=128", "--mset", "ssm_d_state=8",
          "--mset", "ssm_headdim=16", "--mset", "ssm_chunk=16"]


def _args(out, *extra):
    return (["--model_config", os.path.join(ROOT, "configs/model/xs.json"),
             "--train_config", os.path.join(ROOT, "configs/train/proxy_xs.json")] + TINY_M +
            ["--set", "synthetic=true", "--set", "seq_len=32", "--set", "global_batch=4", "--set", "micro_batch=2",
             "--set", "warmup_steps=2", "--set", "total_tokens=1024", "--set", "log_every=1", "--set", "eval_every=1000",
             "--set", "eval_tokens=64", "--set", "diag_tokens=64", "--set", f"out_dir={out}",
             "--set", "time_limit_hours=0", "--set", "ckpt_minutes=1000"] + list(extra))


def _final(out):
    p = sorted(glob.glob(os.path.join(out, "proxy_xs", "checkpoints", "ckpt_*.pt")))[-1]
    return torch.load(p, weights_only=False)


def test_wsd_schedule_shape():
    lrs = [wsd_lr(s, 100, 10, 0.2, 1.0) for s in range(101)]
    assert abs(lrs[0] - 0.1) < 1e-9 and lrs[9] == 1.0 and lrs[50] == 1.0 and lrs[79] == 1.0
    assert lrs[80] == 1.0 and lrs[90] < 1.0 and lrs[100] == 0.0
    assert all(a >= b for a, b in zip(lrs[80:], lrs[81:]))
    assert sm_lambda(0, 10, 0.5) == 0.05 and sm_lambda(50, 10, 0.5) == 0.5


def test_newton_schulz_is_near_orthogonal():
    torch.manual_seed(0)
    X = newton_schulz(torch.randn(32, 64))
    sv = torch.linalg.svdvals(X)
    assert sv.min() > 0.5 and sv.max() < 1.3


def test_every_param_in_exactly_one_optimizer():
    model = MorphForCausalLM(MorphConfig(vocab_size=256, d_model=64, n_layers=6, attn_layers=[2, 4], cla_groups=[[2, 4]],
                                         n_heads=2, head_dim=32, kv_lora_rank=16, mlp_hidden=128, ssm_d_state=8, ssm_headdim=16))
    for opt_name in ("muon", "adamw"):
        opts = build_optimizers(model, TrainConfig(optimizer=opt_name))
        ids = [id(p) for o in opts for g in o.param_groups for p in g["params"]]
        assert len(ids) == len(set(ids)) == len(list(model.parameters()))
    muon, rest = split_params(model)
    assert all(p.ndim == 2 for p in muon) and model.model.embed.weight in rest


def test_resume_is_exact():
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        pretrain_main(_args(a, "--set", "max_steps=6"))
        pretrain_main(_args(b, "--set", "max_steps=3"))
        pretrain_main(_args(b, "--set", "max_steps=3"))     # resumes from step 3
        ca, cb = _final(a), _final(b)
        assert ca["step"] == cb["step"] == 6
        for k in ca["model"]:
            assert torch.equal(ca["model"][k], cb["model"][k]), k


def test_ddp_two_processes_matches_single_process():
    env = dict(os.environ, PYTHONPATH=ROOT, OMP_NUM_THREADS="1")
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        pretrain_main(_args(a, "--set", "max_steps=3"))
        cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=2", "--master_port=29517",
               "-m", "morph.train.pretrain"] + _args(b, "--set", "max_steps=3", "--set", "micro_batch=1")
        r = subprocess.run(cmd, env=env, cwd=ROOT, capture_output=True, text=True, timeout=600)
        assert r.returncode == 0, r.stderr[-3000:]
        ca, cb = _final(a), _final(b)
        for k in ca["model"]:
            assert torch.allclose(ca["model"][k], cb["model"][k], atol=2e-4, rtol=1e-3), k
