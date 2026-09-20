"""Resumable checkpointing with optional off-box (Hugging Face Hub) storage.

Kaggle/Colab local disk is ephemeral, so each save can be pushed to the HF Hub and
the next session pulls the latest. A checkpoint bundles model + optimizer + LR
scheduler + RNG + the data stream's consumed-document offset + step/token counters,
so training resumes bit-for-bit-close across sessions.
"""
from __future__ import annotations

import os
import glob
import torch


def save_checkpoint(path: str, *, model, optimizer, scheduler, step: int, tokens: int,
                    docs_consumed: int, extra: dict | None = None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "step": step,
        "tokens": tokens,
        "docs_consumed": docs_consumed,
        "torch_rng": torch.get_rng_state(),
        "extra": extra or {},
    }, path)


def load_checkpoint(path: str, *, model, optimizer=None, scheduler=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if ckpt.get("torch_rng") is not None:
        try:
            torch.set_rng_state(ckpt["torch_rng"])
        except Exception:
            pass
    return ckpt


def latest_local(ckpt_dir: str) -> str | None:
    files = glob.glob(os.path.join(ckpt_dir, "step_*.pt"))
    if not files:
        return None
    return max(files, key=lambda p: int(p.split("step_")[-1].split(".")[0]))


def push_to_hub(local_path: str, repo_id: str, token: str | None = None):
    """Upload a checkpoint file to a HF Hub model repo. No-op if hub unavailable."""
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token or os.environ.get("HF_TOKEN"))
        api.create_repo(repo_id, repo_type="model", exist_ok=True)
        api.upload_file(path_or_fileobj=local_path,
                        path_in_repo=os.path.basename(local_path),
                        repo_id=repo_id, repo_type="model")
        return True
    except Exception as e:  # keep training alive even if upload fails
        print(f"[checkpoint] hub push skipped: {e}")
        return False


def pull_latest_from_hub(repo_id: str, ckpt_dir: str, token: str | None = None) -> str | None:
    try:
        from huggingface_hub import HfApi, hf_hub_download
        api = HfApi(token=token or os.environ.get("HF_TOKEN"))
        files = [f for f in api.list_repo_files(repo_id) if f.startswith("step_") and f.endswith(".pt")]
        if not files:
            return None
        latest = max(files, key=lambda p: int(p.split("step_")[-1].split(".")[0]))
        return hf_hub_download(repo_id, latest, local_dir=ckpt_dir, token=token)
    except Exception as e:
        print(f"[checkpoint] hub pull skipped: {e}")
        return None
