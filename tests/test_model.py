"""Stage-0 gate: shapes, gradients, CLA sharing, self-modeling loss, param count.

Runs on CPU (pure-PyTorch SSM fallback). No GPU / mamba-ssm required.
"""
import torch

from morph.model import MorphConfig, MorphForCausalLM
from morph.model.mla_attention import MLAAttention
from morph.model.cla import assign_cla_groups


def tiny_cfg(**kw):
    base = dict(vocab_size=256, d_model=128, n_layers=7, max_seq_len=64,
                n_heads=4, kv_lora_rank=32, qk_rope_head_dim=16, qk_nope_head_dim=16,
                v_head_dim=32, ssm_d_state=16, ssm_headdim=32, self_model_dim=64)
    base.update(kw)
    return MorphConfig(**base)


def test_forward_shapes_and_loss():
    cfg = tiny_cfg()
    model = MorphForCausalLM(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 32))
    out = model(ids, labels=ids)
    assert out["logits"].shape == (2, 32, cfg.vocab_size)
    assert out["loss"].ndim == 0
    assert "self_model_loss" in out  # self-modeling active by default


def test_backward_grads_flow():
    cfg = tiny_cfg()
    model = MorphForCausalLM(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    out = model(ids, labels=ids)
    out["loss"].backward()
    n_grad = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
    n_par = sum(1 for p in model.parameters() if p.requires_grad)
    assert n_grad > 0.9 * n_par, f"only {n_grad}/{n_par} params got grads"


def test_cla_sharing_reduces_kv_modules():
    # pattern with 4 attention layers, group size 2 -> 2 shared latent-KV modules
    cfg = tiny_cfg(n_layers=12, layer_pattern="mmaa", cla_group_size=2)
    model = MorphForCausalLM(cfg)
    n_shared = len(model.model.shared_kv)
    n_attn = sum(model.model.is_attn)
    assert n_attn == 6
    groups = assign_cla_groups(cfg.block_types(), cfg.cla_group_size)
    assert n_shared == max(groups.values()) + 1
    assert n_shared < n_attn  # CLA actually shares


def test_self_model_off_toggle():
    cfg = tiny_cfg(self_model_enabled=False)
    model = MorphForCausalLM(cfg)
    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    out = model(ids, labels=ids)
    assert "self_model_loss" not in out


def test_kv_latent_is_small():
    # the cached tensors per attention layer group are (rank + rope) per token,
    # vs MHA which caches n_heads * head_dim * 2. Assert the compression ratio.
    cfg = tiny_cfg()
    mha_bytes = cfg.n_heads * (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim) * 2
    mla_bytes = cfg.kv_lora_rank + cfg.qk_rope_head_dim
    assert mla_bytes < mha_bytes / 2, (mla_bytes, mha_bytes)
