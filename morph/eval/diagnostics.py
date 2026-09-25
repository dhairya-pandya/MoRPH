"""Self-modeling diagnostics: how simple / cacheable are the model's internal states?

- effective rank (Roy & Vetterli 2007): exp(entropy(s / sum(s))) of the centered states;
- rank-r energy: fraction of variance in the top-r principal directions, r = kv_lora_rank
  (the most a rank-r latent could capture linearly);
- self-model R^2: 1 - MSE / Var of the head's prediction of each (RMS-normalized) target;
- weight width: std of weights per module family (the paper's second complexity measure).
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
from typing import Dict

import torch

from morph.model.self_model_head import rms_normalize


def _svals(X: torch.Tensor) -> torch.Tensor:
    """Singular values of the centered (tokens x d) matrix via the d x d Gram matrix."""
    X = X.float()
    X = X - X.mean(0, keepdim=True)
    gram = (X.T @ X).double().cpu()
    return torch.linalg.eigvalsh(gram).clamp_min(0).flip(0).sqrt()


def effective_rank(X: torch.Tensor) -> float:
    s = _svals(X)
    p = s / s.sum().clamp_min(1e-12)
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


def energy_at(X: torch.Tensor, r: int) -> float:
    e = _svals(X).pow(2)
    return float(e[:r].sum() / e.sum().clamp_min(1e-12))


def weight_width(model) -> Dict[str, float]:
    fam = defaultdict(list)
    for name, p in model.named_parameters():
        if p.ndim != 2:
            continue
        if "embed" in name:
            key = "embed"
        elif "self_model" in name:
            key = "sm_head"
        elif "mixer.mixer.in_proj" in name:
            key = "ssm_in"
        elif "mixer.mixer.out_proj" in name:
            key = "ssm_out"
        elif "q_proj" in name or "kv_up" in name or "latent" in name:
            key = "attn_qkv"
        elif "o_proj" in name:
            key = "attn_o"
        elif "mlp" in name:
            key = "mlp"
        else:
            key = "other"
        fam[key].append(p.detach().float().flatten())
        if key not in ("embed", "sm_head"):
            fam["all_blocks"].append(p.detach().float().flatten())
    return {k: float(torch.cat(v).std()) for k, v in fam.items()}


@torch.no_grad()
def state_diagnostics(model, input_ids: torch.Tensor, amp_dtype=None) -> Dict[str, float]:
    """Diagnostics on the states entering each attention layer (the cached-latent sources)."""
    cfg = model.cfg
    layers = sorted(set(cfg.attn_layers) | set(model.targets))
    was_training = model.training
    model.eval()
    ctx = torch.autocast(input_ids.device.type, dtype=amp_dtype) if amp_dtype else nullcontext()
    with ctx:
        out = model(input_ids, collect=layers)
    hidden, collected = out["hidden"], out["collected"]
    res: Dict[str, float] = {}
    pred = None
    if model.self_model is not None:
        pred = model.self_model.proj(hidden).float().view(*hidden.shape[:2], len(model.targets), cfg.d_model)
    for li in layers:
        X = rms_normalize(collected[li]).reshape(-1, cfg.d_model)
        res[f"erank/L{li}"] = effective_rank(X)
        res[f"energy{cfg.kv_lora_rank}/L{li}"] = energy_at(X, cfg.kv_lora_rank)
        if pred is not None and li in model.targets:
            k = model.targets.index(li)
            P = pred[:, :, k].reshape(-1, cfg.d_model)
            var = (X - X.mean(0)).pow(2).mean()
            res[f"sm_r2/L{li}"] = float(1 - (P - X).pow(2).mean() / var.clamp_min(1e-12))
    for k, v in weight_width(model).items():
        res[f"wstd/{k}"] = v
    model.train(was_training)
    return res
