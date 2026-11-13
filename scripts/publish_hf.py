"""Publish MORPH weights to a Hugging Face model repo.

    HF_TOKEN=... python scripts/publish_hf.py --weights weights_final.safetensors \
        --repo dhairyapandya/morph-s [--sft weights_sft.safetensors] [--public]

Uploads the fp16 safetensors, config.json (read from the weights' metadata), MODEL_CARD.md as
README.md and any result JSONs passed with --extra. The repo is private unless --public.
"""
import argparse
import json
import os
import shutil
import sys
import tempfile


def model_config(weights: str) -> dict:
    from safetensors import safe_open
    with safe_open(weights, "pt") as f:
        meta = f.metadata() or {}
    if "model_config" not in meta:
        sys.exit(f"{weights}: no model_config in safetensors metadata")
    return json.loads(meta["model_config"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="base model weights (safetensors)")
    ap.add_argument("--sft", default=None, help="instruction-tuned weights, uploaded as sft/model.safetensors")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--card", default=os.path.join(os.path.dirname(__file__), "..", "MODEL_CARD.md"))
    ap.add_argument("--extra", nargs="*", default=[], help="extra files (eval JSONs, samples)")
    ap.add_argument("--public", action="store_true")
    args = ap.parse_args(argv)

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN not set")
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(args.repo, private=not args.public, exist_ok=True)

    with tempfile.TemporaryDirectory() as d:
        shutil.copy(args.weights, os.path.join(d, "model.safetensors"))
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump(model_config(args.weights), f, indent=2)        # loads with MorphConfig.from_json
        shutil.copy(args.card, os.path.join(d, "README.md"))
        if args.sft:
            os.makedirs(os.path.join(d, "sft"))
            shutil.copy(args.sft, os.path.join(d, "sft", "model.safetensors"))
        for p in args.extra:
            os.makedirs(os.path.join(d, "eval"), exist_ok=True)
            shutil.copy(p, os.path.join(d, "eval", os.path.basename(p)))
        api.upload_folder(folder_path=d, repo_id=args.repo, commit_message="Upload MORPH weights")
    print(f"https://huggingface.co/{args.repo} ({'public' if args.public else 'private'})")


if __name__ == "__main__":
    main()
