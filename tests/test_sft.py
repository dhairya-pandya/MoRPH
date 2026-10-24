"""Stage-4 SFT gate: ChatML masks, masked shards, masked training, sampling."""
import json
import os
import tempfile

import numpy as np
import torch

from morph.data.sft import IM_END, IM_START, prompt_ids, render, write_sft_shards
from morph.data.shards import ShardSet, inputs_targets, val_batches


class Enc:
    def __init__(self, ids):
        self.ids = ids


class FakeTok:
    """Character-level stand-in for the SmolLM2 tokenizer (ids 3..250), specials 0/1/2."""
    special = {"<|endoftext|>": 0, IM_START: 1, IM_END: 2}

    def token_to_id(self, s):
        return self.special[s]

    def encode(self, text, add_special_tokens=False):
        return Enc([3 + (ord(c) % 240) for c in text])

    def get_vocab_size(self):
        return 256


CONV = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi?"},
        {"role": "assistant", "content": "hello"}]


def test_render_masks_only_assistant_reply_and_its_end_token():
    tok = FakeTok()
    ids, mask = render(tok, CONV)
    assert len(ids) == len(mask)
    learned = [i for i, m in zip(ids, mask) if m]
    assert learned == tok.encode("hello").ids + [2]          # reply + <|im_end|>, nothing else
    assert ids.count(1) == 3 and ids.count(2) == 3
    p = prompt_ids(tok, CONV[:2])
    assert p[-len(tok.encode("assistant\n").ids) - 1] == 1   # ends with an open assistant turn


def _write(d, n=300):
    tok = FakeTok()
    convs = [CONV] * n
    return write_sft_shards(iter(convs), tok, d, eos_id=0, max_tokens=8000, val_convs=iter([CONV] * 40),
                            shard_tokens=3000, meta_extra={"vocab_size": 256})


def test_masked_shards_roundtrip_and_targets():
    with tempfile.TemporaryDirectory() as d:
        meta = _write(d)
        assert meta["masked"] and len(meta["train"]["mask_files"]) == len(meta["train"]["files"]) > 1
        ds = ShardSet(d, "train", seq_len=32)
        assert ds.masks is not None
        w, m = ds.window(3), ds.window_mask(3)
        assert len(w) == len(m) == 33
        x, y = inputs_targets(val_batches(ShardSet(d, "val", 32), 4, 2)[0], torch.device("cpu"))
        assert x.shape == y.shape and (y == -100).any() and (y != -100).any()


def test_training_on_masked_shards_runs_and_scores_only_replies():
    from morph.train.pretrain import main as pretrain_main
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tiny = ["--mset", "vocab_size=256", "--mset", "d_model=64", "--mset", "n_heads=2", "--mset", "head_dim=32",
            "--mset", "kv_lora_rank=16", "--mset", "mlp_hidden=128", "--mset", "ssm_d_state=8",
            "--mset", "ssm_headdim=16", "--mset", "ssm_chunk=16"]
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as out:
        _write(d)
        pretrain_main(["--model_config", f"{root}/configs/model/xs.json", "--train_config",
                       f"{root}/configs/train/proxy_xs.json"] + tiny +
                      ["--set", f"data_dir={d}", "--set", "seq_len=32", "--set", "global_batch=4",
                       "--set", "micro_batch=2", "--set", "max_steps=6", "--set", "warmup_steps=2",
                       "--set", "total_tokens=4096", "--set", "log_every=1", "--set", "eval_every=6",
                       "--set", "eval_tokens=128", "--set", "diag_tokens=64", "--set", f"out_dir={out}",
                       "--set", "time_limit_hours=0", "--set", "ckpt_minutes=1000", "--set", "compile=false"])
        recs = [json.loads(l) for l in open(f"{out}/proxy_xs/metrics.jsonl")]
        train = [r["ce"] for r in recs if r["type"] == "train"]
        assert len(train) == 6 and train[-1] < train[0]       # the repeated reply is quickly learnable
        assert any(r["type"] == "eval" for r in recs)


def test_sample_stops_at_stop_token():
    from morph.eval.generate import sample
    from morph.model import MorphConfig, MorphForCausalLM
    cfg = MorphConfig(vocab_size=256, d_model=64, n_layers=6, attn_layers=[2, 4], cla_groups=[[2, 4]], n_heads=2,
                      head_dim=32, kv_lora_rank=16, mlp_hidden=128, ssm_d_state=8, ssm_headdim=16, ssm_chunk=16)
    model = MorphForCausalLM(cfg)
    out = sample(model, [5, 6, 7], max_new=20, temperature=0.0, top_p=1.0, stop=list(range(256)),
                 device=torch.device("cpu"))
    assert len(out) == 1
    out = sample(model, [5, 6, 7], max_new=5, temperature=0.8, top_p=0.9, stop=[], device=torch.device("cpu"))
    assert len(out) == 5
