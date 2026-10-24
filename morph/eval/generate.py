"""Sample text from a checkpoint: plain continuation (base model) or a ChatML reply (SFT model).

    PYTHONPATH=. python -m morph.eval.generate --ckpt CKPT --prompt "The mitochondria is"
    PYTHONPATH=. python -m morph.eval.generate --ckpt SFT_CKPT --chat --prompt "Explain photosynthesis briefly."

Recomputes the full sequence every step (no cache) — fine for short demos; MORPH's small
latent cache and constant SSM state are what a production decoder would keep instead.
"""
from __future__ import annotations

import argparse
from typing import List

import torch

from morph.data.sft import IM_END, prompt_ids


@torch.no_grad()
def sample(model, ids: List[int], max_new: int, temperature: float, top_p: float, stop: List[int],
           device, amp_dtype=None, seed: int = 0) -> List[int]:
    g = torch.Generator(device="cpu").manual_seed(seed)
    out = list(ids)
    model.eval()
    for _ in range(max_new):
        x = torch.tensor([out[-model.cfg.max_seq_len:]], device=device)
        ctx = torch.autocast(device.type, dtype=amp_dtype) if amp_dtype else torch.no_grad()
        with ctx:
            logits = model(x)["logits"][0, -1].float().cpu()
        if temperature <= 0:
            nxt = int(logits.argmax())
        else:
            probs = torch.softmax(logits / temperature, -1)
            sp, si = probs.sort(descending=True)
            keep = sp.cumsum(0) - sp < top_p                     # nucleus: smallest set with mass >= top_p
            sp = sp * keep
            nxt = int(si[torch.multinomial(sp / sp.sum(), 1, generator=g)])
        out.append(nxt)
        if nxt in stop:
            break
    return out[len(ids):]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--tokenizer", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--prompt", action="append", required=True)
    ap.add_argument("--chat", action="store_true", help="wrap the prompt as a ChatML user turn")
    ap.add_argument("--system", default="You are a helpful assistant.")
    ap.add_argument("--max_new", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    args = ap.parse_args(argv)
    from tokenizers import Tokenizer
    from morph.eval.quality_bench import load_model
    from morph.train.distributed import pick_precision
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args, device)
    tok = Tokenizer.from_pretrained(args.tokenizer)
    stop = [tok.token_to_id("<|endoftext|>")] + ([tok.token_to_id(IM_END)] if args.chat else [])
    for p in args.prompt:
        if args.chat:
            ids = prompt_ids(tok, [{"role": "system", "content": args.system}, {"role": "user", "content": p}])
        else:
            ids = tok.encode(p, add_special_tokens=False).ids
        new = sample(model, ids, args.max_new, args.temperature, args.top_p, stop, device, pick_precision(device))
        print(f"### {p}\n{tok.decode([t for t in new if t not in stop])}\n", flush=True)


if __name__ == "__main__":
    main()
