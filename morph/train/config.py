"""TrainConfig — run/optimization/checkpoint settings (JSON + `--set key=value` overrides)."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields
from typing import List
import json


@dataclass
class TrainConfig:
    run_name: str = "run"
    out_dir: str = "runs"                 # run artifacts live in out_dir/run_name
    data_dir: str = "data"
    synthetic: bool = False               # random tokens instead of shards (smoke tests)

    # batch / budget
    seq_len: int = 2048
    global_batch: int = 128               # sequences per optimizer step (all ranks)
    micro_batch: int = 0                  # per rank; 0 = probe the largest that fits
    total_tokens: float = 5e9
    max_steps: int = 0                    # >0 caps steps (smoke tests); does not change the schedule

    # optimization
    optimizer: str = "muon"               # muon | adamw
    lr: float = 2e-3
    min_lr_frac: float = 0.0
    warmup_steps: int = 500
    decay_frac: float = 0.2               # WSD: final fraction of the planned steps decays (1 - sqrt)
    weight_decay: float = 0.1
    muon_momentum: float = 0.95
    adam_betas: List[float] = field(default_factory=lambda: [0.9, 0.95])
    grad_clip: float = 1.0
    seed: int = 1337

    # systems
    grad_ckpt: bool = True
    compile: bool = False
    precision: str = "auto"               # auto | bf16 | fp16 | fp32

    # logging / eval
    log_every: int = 10
    eval_every: int = 250
    eval_tokens: int = 1_000_000
    diag_tokens: int = 65_536
    wandb: bool = False

    # checkpoints / sessions
    ckpt_minutes: float = 45.0
    keep_ckpts: int = 2
    milestone_tokens: float = 5e8
    time_limit_hours: float = 0.0         # 0 = no session deadline
    reserve_minutes: float = 10.0
    hub_repo: str = ""                    # e.g. user/morph-runs (private); needs HF_TOKEN
    hub_keep: int = 2
    resume_dirs: List[str] = field(default_factory=list)

    @property
    def tokens_per_step(self) -> int:
        return self.global_batch * self.seq_len

    @property
    def total_steps(self) -> int:
        return max(1, -(-int(self.total_tokens) // self.tokens_per_step))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown TrainConfig keys: {sorted(unknown)}")
        return cls(**d)

    @classmethod
    def from_json(cls, path: str) -> "TrainConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))


def apply_overrides(obj, pairs: List[str]):
    """Apply `key=value` strings; values parsed as JSON when possible (so 0.1, true, [1,2] work)."""
    names = {f.name: f for f in fields(obj)}
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"override {pair!r} must be key=value")
        k, v = pair.split("=", 1)
        if k not in names:
            raise ValueError(f"unknown key {k!r} for {type(obj).__name__}")
        try:
            val = json.loads(v)
        except json.JSONDecodeError:
            val = v
        setattr(obj, k, val)
    if hasattr(obj, "__post_init__"):
        obj.__post_init__()
    return obj
