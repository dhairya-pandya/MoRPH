"""Stage-3 SVF gate: exact wrapping, z-only training, expert algebra, end-to-end CLI on synthetic domains."""
import json
import os
import tempfile

import numpy as np
import torch

from morph.data.prepare import write_shards
from morph.model import MorphConfig, MorphForCausalLM
from morph.model.svf import (SVFLinear, apply_svf, get_expert, identity_expert, load_expert, mix_experts,
                             save_expert, set_expert, svf_modules)
from morph.train.checkpoint import save_atomic


def tiny():
    return MorphConfig(vocab_size=256, d_model=64, n_layers=6, attn_layers=[2, 4], cla_groups=[[2, 4]],
                       n_heads=2, head_dim=32, kv_lora_rank=16, mlp_hidden=128, ssm_d_state=8,
                       ssm_headdim=16, ssm_chunk=16)


def test_svf_linear_with_unit_scales_equals_linear():
    torch.manual_seed(0)
    lin = torch.nn.Linear(24, 40, bias=True)
    x = torch.randn(3, 5, 24)
    assert torch.allclose(SVFLinear(lin)(x), lin(x), atol=1e-5)


def test_apply_svf_keeps_outputs_and_trains_only_z():
    torch.manual_seed(0)
    model = MorphForCausalLM(tiny()).eval()
    ids = torch.randint(0, 256, (2, 20))
    with torch.no_grad():
        ref = model(ids)["logits"]
    names = apply_svf(model)
    assert names and all(isinstance(m, SVFLinear) for m in svf_modules(model).values())
    with torch.no_grad():
        assert torch.allclose(model(ids)["logits"], ref, atol=1e-4)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all(n.endswith(".z") for n in trainable)
    model.self_model = None
    model.train()
    model(ids, ids)["loss"].backward()
    assert all(m.z.grad is not None and m.z.grad.abs().sum() > 0 for m in svf_modules(model).values())


def test_expert_mixing_and_roundtrip():
    model = MorphForCausalLM(tiny())
    apply_svf(model)
    ident = identity_expert(model)
    a = {k: v * 2.0 for k, v in ident.items()}
    m = mix_experts([a, ident], [0.25, 0.75])
    assert all(torch.allclose(m[k], torch.full_like(v, 1.25)) for k, v in ident.items())
    set_expert(model, a)
    assert all(torch.allclose(v, a[k]) for k, v in get_expert(model).items())
    with tempfile.TemporaryDirectory() as d:
        save_expert(f"{d}/a.safetensors", a, {"domain": "x"})
        b = load_expert(f"{d}/a.safetensors")
    assert all(torch.equal(a[k], b[k]) for k in a)
    set_expert(model, None)
    assert all(torch.equal(v, ident[k]) for k, v in get_expert(model).items())


def _domain(d, lo, hi, seed):
    rng = np.random.default_rng(seed)
    docs = (rng.integers(lo, hi, size=200).tolist() for _ in range(400))
    write_shards(docs, d, eos_id=0, train_tokens=30000, val_every=5, val_tokens=8000, shard_tokens=100000)
    meta = json.load(open(f"{d}/meta.json"))
    meta["vocab_size"] = 256
    json.dump(meta, open(f"{d}/meta.json", "w"))


def test_cli_end_to_end_on_synthetic_domains():
    from morph.train.svf_train import main as svf
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as d:
        _domain(f"{d}/low", 1, 100, 0)
        _domain(f"{d}/high", 150, 250, 1)
        cfg = tiny()
        save_atomic({"model": MorphForCausalLM(cfg).state_dict(), "model_config": cfg.to_dict()}, f"{d}/base.pt")
        common = ["--seq", "64"]
        svf(["expert", "--base", f"{d}/base.pt", "--domain", f"low={d}/low", "--general", f"{d}/high",
             "--out", f"{d}/low.safetensors", "--steps", "30", "--batch", "8", "--micro", "4", "--lr", "0.05",
             "--eval_every", "30", "--eval_windows", "16"] + common)
        meta = json.loads(__import__("safetensors").safe_open(f"{d}/low.safetensors", "pt").metadata()["meta"])
        assert meta["log"][-1]["val_ce"] < meta["base_val_ce"]   # z alone improves the domain loss
        svf(["router", "--base", f"{d}/base.pt", "--domains", f"low={d}/low", f"high={d}/high",
             "--out", f"{d}/router.pt", "--prompt", "16", "--n_train", "32", "--n_val", "16", "--micro", "8"])
        assert torch.load(f"{d}/router.pt", weights_only=False)["val_acc"] > 0.7      # well above chance (0.5) even with an untrained base
        svf(["evaluate", "--base", f"{d}/base.pt", "--domains", f"low={d}/low", f"high={d}/high",
             "--experts", f"low={d}/low.safetensors", "--router", f"{d}/router.pt", "--n_eval", "8",
             "--micro", "4", "--out", f"{d}/res.json"] + common)
        res = json.load(open(f"{d}/res.json"))
        assert res["low"]["expert:low"] < res["low"]["base"]
        assert res["low"]["route_acc"] > 0.6 and os.path.exists(f"{d}/res.json")
