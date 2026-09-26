"""Model v2 gate: SSD parity, Mamba2 layout, causality, true CLA, self-modeling, losses."""
import torch
import torch.nn.functional as F

from morph.model import MorphConfig, MorphForCausalLM
from morph.model.cla import kv_bytes_per_token
from morph.model.losses import chunked_cross_entropy
from morph.model.mamba2_block import Mamba2Mixer
from morph.model.ssd import ssd_chunked, ssd_reference


def tiny_cfg(**kw):
    base = dict(vocab_size=256, d_model=64, n_layers=6, max_seq_len=64, attn_layers=[2, 4],
                cla_groups=[[2, 4]], n_heads=2, head_dim=32, kv_lora_rank=16, mlp_hidden=128,
                ssm_d_state=8, ssm_headdim=16, ssm_chunk=8)
    base.update(kw)
    return MorphConfig(**base)


def test_ssd_chunked_matches_reference_values_and_grads():
    torch.manual_seed(0)
    for (b, l, h, p, n, q) in [(2, 37, 3, 4, 5, 8), (1, 32, 2, 8, 6, 16), (2, 5, 2, 4, 4, 16)]:
        x = torch.randn(b, l, h, p, requires_grad=True)
        dt = F.softplus(torch.randn(b, l, h)).requires_grad_()
        A = -(torch.rand(h) * 2 + 0.1)
        B = torch.randn(b, l, n, requires_grad=True)
        C = torch.randn(b, l, n, requires_grad=True)
        y1, s1 = ssd_chunked(x, dt, A, B, C, q)
        g1 = torch.autograd.grad(y1.square().sum(), (x, dt, B, C))
        y2, s2 = ssd_reference(x, dt, A, B, C)
        g2 = torch.autograd.grad(y2.square().sum(), (x, dt, B, C))
        assert torch.allclose(y1, y2, atol=1e-4), (y1 - y2).abs().max()
        assert torch.allclose(s1, s2, atol=1e-4)
        for a, c in zip(g1, g2):
            assert torch.allclose(a, c, atol=1e-3, rtol=1e-3), (a - c).abs().max()


def test_mamba2_param_layout_matches_mamba_ssm():
    cfg = tiny_cfg()
    m = Mamba2Mixer(cfg)
    di, N, H = cfg.ssm_d_inner, cfg.ssm_d_state, cfg.ssm_nheads
    shapes = {k: tuple(v.shape) for k, v in m.state_dict().items()}
    assert shapes == {
        "in_proj.weight": (2 * di + 2 * N + H, cfg.d_model),
        "conv1d.weight": (di + 2 * N, 1, cfg.ssm_d_conv),
        "conv1d.bias": (di + 2 * N,),
        "dt_bias": (H,), "A_log": (H,), "D": (H,),
        "norm.weight": (di,),
        "out_proj.weight": (cfg.d_model, di),
    }
    assert (torch.exp(m.A_log) >= 1).all() and (torch.exp(m.A_log) <= 16).all()


def test_forward_backward_and_all_params_get_grads():
    cfg = tiny_cfg()
    model = MorphForCausalLM(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 24))
    out = model(ids, ids)
    assert out["loss"].ndim == 0 and "sm_loss" in out
    out["loss"].backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, missing


def test_whole_model_is_causal():
    torch.manual_seed(0)
    cfg = tiny_cfg()
    model = MorphForCausalLM(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 20))
    ids2 = ids.clone()
    ids2[0, 13:] = (ids2[0, 13:] + 7) % cfg.vocab_size
    with torch.no_grad():
        a = model(ids)["logits"]
        b = model(ids2)["logits"]
    assert torch.allclose(a[:, :13], b[:, :13], atol=1e-5)
    assert not torch.allclose(a[:, 13:], b[:, 13:], atol=1e-3)


def test_true_cla_consumer_reuses_producer_latent():
    cfg = tiny_cfg(n_layers=8, attn_layers=[2, 4, 6], cla_groups=[[2, 4]])
    model = MorphForCausalLM(cfg)
    layers = model.model.layers
    assert layers[2].mixer.latent is not None      # producer
    assert layers[4].mixer.latent is None          # consumer owns no down-projection
    assert layers[6].mixer.latent is not None      # singleton group
    seen = {}
    layers[2].mixer.latent.register_forward_hook(lambda m, i, o: seen.__setitem__("prod", o))
    orig = layers[4].mixer.forward

    def spy(x, latent=None):
        seen["cons"] = latent
        return orig(x, latent)
    layers[4].mixer.forward = spy
    model(torch.randint(0, cfg.vocab_size, (1, 10)))
    assert seen["cons"] is seen["prod"]
    assert kv_bytes_per_token(cfg) == 2 * cfg.kv_lora_rank * 2   # 2 groups x rank x fp16


def test_self_model_gradient_reaches_targets_only_when_not_detached():
    torch.manual_seed(0)
    ids = torch.randint(0, 256, (2, 16))
    for detach, expect in [(False, True), (True, False)]:
        cfg = tiny_cfg(self_model_detach=detach, self_model_targets=[2])
        model = MorphForCausalLM(cfg)
        hidden, collected = model.model(ids, collect=[2])
        sm, _ = model.self_model(hidden.detach(), [collected[2]])   # isolate the target path
        sm.backward()
        g = model.model.layers[0].mlp.w_down.weight.grad
        got = g is not None and g.abs().sum() > 0
        assert got == expect, (detach, got)


def test_self_model_loss_is_invariant_to_target_scale():
    cfg = tiny_cfg()
    model = MorphForCausalLM(cfg)
    src = torch.randn(2, 8, cfg.d_model)
    t = torch.randn(2, 8, cfg.d_model)
    a, _ = model.self_model(src, [t, t])
    b, _ = model.self_model(src, [t * 50.0, t * 50.0])
    assert torch.allclose(a, b, atol=1e-5)


def test_chunked_cross_entropy_matches_reference():
    torch.manual_seed(0)
    h = torch.randn(3, 10, 16, requires_grad=True)
    w = torch.randn(50, 16, requires_grad=True)
    y = torch.randint(0, 50, (3, 10))
    y[0, :3] = -100
    ce, zsq = chunked_cross_entropy(h, w, y, chunk=7)
    logits = h @ w.T
    ref = F.cross_entropy(logits.reshape(-1, 50), y.reshape(-1), ignore_index=-100)
    lse = torch.logsumexp(logits, -1)[y != -100]
    assert torch.allclose(ce, ref, atol=1e-5) and torch.allclose(zsq, (lse ** 2).mean(), atol=1e-4)
    g1 = torch.autograd.grad(ce, (h, w))
    g2 = torch.autograd.grad(ref, (h, w))
    for a, b in zip(g1, g2):
        assert torch.allclose(a, b, atol=1e-5)


def test_grad_checkpointing_gives_same_grads():
    torch.manual_seed(0)
    cfg = tiny_cfg()
    model = MorphForCausalLM(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    model(ids, ids)["loss"].backward()
    g1 = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    model.set_grad_checkpointing(True)
    model(ids, ids)["loss"].backward()
    for a, p in zip(g1, model.parameters()):
        assert torch.allclose(a, p.grad, atol=1e-5)


def test_presets_param_counts():
    xs = MorphForCausalLM(MorphConfig.from_json("configs/model/xs.json"))
    s = MorphForCausalLM(MorphConfig.from_json("configs/model/s.json"))
    assert 40e6 < xs.num_params() < 48e6
    assert 160e6 < s.num_params() < 180e6


def test_probe_mode_leaves_backbone_gradients_untouched():
    torch.manual_seed(0)
    ids = torch.randint(0, 256, (2, 16))
    probe = MorphForCausalLM(tiny_cfg(self_model_probe=True))
    plain = MorphForCausalLM(tiny_cfg(self_model_enabled=False))
    plain.load_state_dict({k: v for k, v in probe.state_dict().items() if not k.startswith("self_model")})
    grads = []
    for model in (probe, plain):
        model(ids, ids)["loss"].backward()
        grads.append({n: p.grad for n, p in model.named_parameters() if not n.startswith("self_model")})
    for n in grads[1]:
        assert torch.allclose(grads[0][n], grads[1][n], atol=1e-6), n


def test_fused_linear_ce_matches_reference():
    from morph.model.losses import fused_linear_cross_entropy
    torch.manual_seed(0)
    h = torch.randn(3, 10, 16, requires_grad=True)
    w = torch.randn(50, 16, requires_grad=True)
    y = torch.randint(0, 50, (3, 10))
    y[1, :4] = -100
    z = 1e-2
    loss, ce, zsq = fused_linear_cross_entropy(h, w, y, chunk=7, z_coef=z)
    logits = h @ w.T
    ref_ce = F.cross_entropy(logits.reshape(-1, 50), y.reshape(-1), ignore_index=-100)
    lse = torch.logsumexp(logits, -1)[y != -100]
    ref = ref_ce + z * (lse ** 2).mean()
    assert torch.allclose(loss, ref, atol=1e-5) and torch.allclose(ce, ref_ce, atol=1e-5)
    g1 = torch.autograd.grad(loss * 3.0, (h, w))
    g2 = torch.autograd.grad(ref * 3.0, (h, w))
    for a, b in zip(g1, g2):
        assert torch.allclose(a, b, atol=1e-5), (a - b).abs().max()
