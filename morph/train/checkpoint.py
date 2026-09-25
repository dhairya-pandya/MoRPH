"""Resumable checkpoints: atomic local files, retention, HF Hub mirror, resume discovery.

A checkpoint (`ckpt_<step>.pt`) bundles model, optimizers, grad scaler, step/tokens, RNG,
both configs and the git SHA. Writes go to a temp file then `os.replace`, so a session
killed mid-save never leaves a truncated "latest" file. Each save can be mirrored to a
private HF model repo (`<hub_repo>/<run_name>/...`) in the background; the newest
checkpoint across local dirs, extra resume dirs (e.g. Kaggle inputs) and the hub wins.
"""
from __future__ import annotations

import glob
import os
import re
from typing import List, Optional, Tuple

import torch

CKPT_RE = re.compile(r"ckpt_(\d+)\.pt$")


def ckpt_step(path: str) -> int:
    m = CKPT_RE.search(os.path.basename(path))
    return int(m.group(1)) if m else -1


def ckpt_name(step: int) -> str:
    return f"ckpt_{step:07d}.pt"


def save_atomic(obj: dict, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def list_local(dirs: List[str]) -> List[str]:
    found = []
    for d in dirs:
        if d and os.path.isdir(d):
            found += [p for p in glob.glob(os.path.join(d, "**", "ckpt_*.pt"), recursive=True)]
    return sorted(found, key=ckpt_step)


def prune_local(ckpt_dir: str, keep: int):
    files = list_local([ckpt_dir])
    for p in files[:-keep] if keep > 0 else []:
        os.remove(p)


class Hub:
    """Background mirror of a run folder to a private HF model repo. All failures are logged, never raised."""

    def __init__(self, repo: str, run_name: str, keep: int = 2):
        from huggingface_hub import HfApi
        self.repo, self.prefix, self.keep = repo, run_name, keep
        self.api = HfApi(token=os.environ.get("HF_TOKEN"))
        self.pending = []
        try:
            self.api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
        except Exception as e:
            print(f"[hub] create_repo failed: {e}", flush=True)

    def upload(self, local: str, remote_name: Optional[str] = None):
        remote = f"{self.prefix}/{remote_name or os.path.basename(local)}"
        try:
            self.pending.append(self.api.upload_file(path_or_fileobj=local, path_in_repo=remote,
                                                     repo_id=self.repo, repo_type="model", run_as_future=True))
        except Exception as e:
            print(f"[hub] upload {remote} failed: {e}", flush=True)

    def wait(self):
        for f in self.pending:
            try:
                f.result()
            except Exception as e:
                print(f"[hub] upload failed: {e}", flush=True)
        self.pending = []

    def remote_ckpts(self) -> List[str]:
        try:
            files = self.api.list_repo_files(self.repo, repo_type="model")
        except Exception as e:
            print(f"[hub] list failed: {e}", flush=True)
            return []
        return sorted([f for f in files if f.startswith(self.prefix + "/") and CKPT_RE.search(f)], key=ckpt_step)

    def prune(self, keep: Optional[int] = None):
        """Wait for pending uploads, then delete all but the newest `keep` remote checkpoints."""
        self.wait()
        keep = self.keep if keep is None else keep
        remote = self.remote_ckpts()
        old = remote[:-keep] if keep > 0 else remote
        for f in old:
            try:
                self.api.delete_file(f, repo_id=self.repo, repo_type="model")
            except Exception as e:
                print(f"[hub] delete {f} failed: {e}", flush=True)

    def download(self, remote: str, local_dir: str) -> Optional[str]:
        from huggingface_hub import hf_hub_download
        try:
            p = hf_hub_download(self.repo, remote, repo_type="model", local_dir=local_dir,
                                token=os.environ.get("HF_TOKEN"))
            return p
        except Exception as e:
            print(f"[hub] download {remote} failed: {e}", flush=True)
            return None


def find_resume(ckpt_dir: str, extra_dirs: List[str], hub: Optional[Hub]) -> Optional[str]:
    """Path of the newest checkpoint (downloading from the hub if that is newest)."""
    local = list_local([ckpt_dir] + list(extra_dirs))
    best_local: Tuple[int, Optional[str]] = (ckpt_step(local[-1]), local[-1]) if local else (-1, None)
    if hub is not None:
        remote = hub.remote_ckpts()
        if remote and ckpt_step(remote[-1]) > best_local[0]:
            got = hub.download(remote[-1], os.path.join(ckpt_dir, "_hub"))
            if got:
                return got
    return best_local[1]
