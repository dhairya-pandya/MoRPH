"""Stage-2 Mixture-of-Depths gate: identity at init, temporal order, causal routing, gradients, init-from."""
import torch

from morph.model import MorphConfig, MorphForCausalLM


def cfg(**kw):
    base = dict(vocab_size=256, d_model=64, n_layers=6, max_seq_len=64, attn_layers=[2, 4],
                cla_groups=[[2, 4]], n_heads=2, head_dim=32, kv_lora_rank=16, mlp_hidden=128,
                ssm_d_state=8, ssm_headdim=16, ssm_chunk=8)
    base.update(kw)
    return MorphConfig(**base)


def pair():
    """(plain model, MoD model with identical non-router weights)."""
    torch.manual_seed(0)
    plain = MorphForCausalLM(cfg())
    mod = MorphForCausalLM(cfg(mod_enabled=True, mod_layers=[1, 3, 5]))
    missing, unexpected = mod.load_state_dict(plain.state_dict(), strict=False)
    assert not unexpected and all(".router." in k for k in missing) and missing
    return plain.eval(), mod.eval()


def test_capacity_one_with_fresh_router_equals_plain_model():
    plain, mod = pair()
    ids = torch.randint(0, 256, (2, 24))
    with torch.no_grad():
        a = plain(ids)["logits"]
        b = mod(ids, mod_capacity=1.0)["logits"]
    assert torch.allclose(a, b, atol=1e-5), (a - b).abs().max()   # also proves gathered order is temporal


def test_topk_executes_capacity_fraction_and_trains_router():
    _, mod = pair()
    mod.train()
    ids = torch.randint(0, 256, (2, 20))
    out = mod(ids, ids, mod_capacity=0.5)
    assert set(out["mod_frac"]) == {1, 3, 5} and all(abs(f - 0.5) < 1e-9 for f in out["mod_frac"].values())
    assert "mod_aux" in out
    out["loss"].backward()
    for i in (1, 3, 5):
        g = mod.model.layers[i].router.proj.weight.grad
        assert g is not None and g.abs().sum() > 0


def test_causal_mode_is_causal_and_respects_router_sign():
    _, mod = pair()
    torch.manual_seed(1)
    for i in (1, 3, 5):
        torch.nn.init.normal_(mod.model.layers[i].router.proj.weight, std=0.5)
    ids = torch.randint(0, 256, (1, 20))
    ids2 = ids.clone()
    ids2[0, 12:] = (ids2[0, 12:] + 5) % 256
    with torch.no_grad():
        a = mod(ids, mod_mode="causal")["logits"]
        b = mod(ids2, mod_mode="causal")["logits"]
    assert torch.allclose(a[:, :12], b[:, :12], atol=1e-5)


def test_negative_router_skips_layer_positive_runs_it():
    plain, mod = pair()
    ids = torch.randint(0, 256, (1, 16))
    for i in (1, 3, 5):
        torch.nn.init.constant_(mod.model.layers[i].router.proj.bias, 50.0)     # always route, scale 2*sigmoid(50)=2
    with torch.no_grad():
        out = mod(ids, mod_mode="causal")
    assert all(f == 1.0 for f in out["mod_frac"].values())
    for i in (1, 3, 5):
        torch.nn.init.constant_(mod.model.layers[i].router.proj.bias, -50.0)    # never route
    with torch.no_grad():
        out = mod(ids, mod_mode="causal")
        h_skip = out["hidden"]
    assert all(f == 0.0 for f in out["mod_frac"].values())
    # skipping layers 1,3,5 must equal a plain forward that bypasses them
    with torch.no_grad():
        x = plain.model.embed(ids)
        lat = None
        for i, layer in enumerate(plain.model.layers):
            if i in (1, 3, 5):
                continue
            x, lat2, _, _ = layer(x, lat)
            lat = lat2 if lat2 is not None else lat
        ref = plain.model.norm_f(x)
    assert torch.allclose(h_skip, ref, atol=1e-5)


def test_stage1_checkpoint_continues_as_mod_model():
    """init_from: plain checkpoint -> MoD model trains with capacity annealing and causal eval."""
    import glob
    import json
    import os
    import tempfile
    from morph.train.pretrain import main as pretrain_main
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tiny = ["--mset", "vocab_size=256", "--mset", "d_model=64", "--mset", "n_heads=2", "--mset", "head_dim=32",
            "--mset", "kv_lora_rank=16", "--mset", "mlp_hidden=128", "--mset", "ssm_d_state=8",
            "--mset", "ssm_headdim=16", "--mset", "ssm_chunk=16", "--mset", "self_model_lambda=1.0"]
    common = ["--set", "synthetic=true", "--set", "seq_len=32", "--set", "global_batch=4", "--set", "micro_batch=2",
              "--set", "warmup_steps=2", "--set", "total_tokens=1024", "--set", "log_every=1",
              "--set", "eval_tokens=256", "--set", "diag_tokens=64", "--set", "time_limit_hours=0",
              "--set", "ckpt_minutes=1000", "--set", "compile=false"]
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        pretrain_main(["--model_config", f"{root}/configs/model/xs.json", "--train_config",
                       f"{root}/configs/train/proxy_xs.json"] + tiny + common +
                      ["--set", f"out_dir={a}", "--set", "max_steps=2", "--set", "eval_every=1000"])
        ck = sorted(glob.glob(f"{a}/proxy_xs/checkpoints/ckpt_*.pt"))[-1]
        pretrain_main(["--model_config", f"{root}/configs/model/xs_mod.json", "--train_config",
                       f"{root}/configs/train/stage2_mod_xs.json"] + tiny + common +
                      ["--set", f"out_dir={b}", "--set", "max_steps=4", "--set", "eval_every=4",
                       "--set", f"init_from={ck}", "--set", "mod_anneal_steps=4"])
        recs = [json.loads(l) for l in open(f"{b}/mod_xs/metrics.jsonl")]
        train = [r for r in recs if r["type"] == "train"]
        ev = [r for r in recs if r["type"] == "eval"][-1]
        assert train[0]["mod_cap"] == 1.0 and train[-1]["mod_cap"] < 1.0
        assert "val_ce_causal" in ev and 0.0 <= ev["mod_exec_causal"] <= 1.0
