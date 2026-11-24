---
license: mit
language: en
library_name: pytorch
pipeline_tag: text-generation
tags: [mamba2, hybrid, mla, self-modeling, small-lm, from-scratch]
datasets: [HuggingFaceFW/fineweb-edu, HuggingFaceTB/smol-smoltalk]
model-index:
- name: MORPH-S
  results:
  - task: {type: text-generation}
    dataset: {name: HellaSwag (validation, zero-shot), type: Rowan/hellaswag}
    metrics:
    - {type: accuracy, value: 32.7, name: base}
    - {type: accuracy, value: 32.4, name: sft}
---

# MORPH-S

A 170M-parameter language model trained from scratch on two free Kaggle T4 GPUs. It is a
Mamba2 / latent-attention hybrid built to keep inference memory small, and it was trained with a
self-modeling auxiliary loss. This repo holds the base model, an instruction-tuned variant and
three small domain experts.

| | |
|---|---|
| Parameters | 169.7M (138.3M without embeddings) |
| Training data | 3.0B tokens of FineWeb-Edu, then 0.15B tokens of smol-smoltalk for the chat variant |
| Validation perplexity (FineWeb-Edu) | 17.4 |
| HellaSwag, zero-shot (10,042 items) | 32.7% base, 32.4% chat (chance 25%) |
| Cache at 32k tokens (fp16) | 23.5 MB, against 2,013 MB for multi-head attention of the same depth |
| License | MIT |

This is a research model. It is small and lightly trained, so it makes factual mistakes and
repeats itself. See [Limitations](#limitations).

## Files

| Path | What it is |
|---|---|
| `model.safetensors` | Base model, fp16 |
| `sft/model.safetensors` | Instruction-tuned model (ChatML format), fp16 |
| `config.json` | Model configuration, loads with `MorphConfig.from_json` |
| `svf/{math,code,stories}.safetensors` | SVF experts, 77.6k numbers each |
| `svf/router.pt` | Router that picks an expert from the first 128 tokens of a prompt |
| `svf/svf_eval.json` | Evaluation of the experts and router |
| `eval/` | HellaSwag results and sample chat replies |

The architecture is custom, so it does not load with `transformers`. Use the code at
[github.com/dhairya-pandya/MoRPH](https://github.com/dhairya-pandya/MoRPH). The tokenizer is
`HuggingFaceTB/SmolLM2-135M` (vocabulary 49,152).

## Quick start

```bash
git clone https://github.com/dhairya-pandya/MoRPH && cd MoRPH && pip install -r requirements.txt
huggingface-cli download dhairyapandya/morph-s --local-dir morph-s

# chat with the instruction-tuned model
PYTHONPATH=. python -m morph.eval.generate --weights morph-s/sft/model.safetensors \
  --config morph-s/config.json --chat --prompt "What is photosynthesis?"

# plain text continuation with the base model
PYTHONPATH=. python -m morph.eval.generate --weights morph-s/model.safetensors \
  --config morph-s/config.json --prompt "The water cycle begins when"
```

Generation recomputes the full sequence for each new token, so it is slow for long outputs.
The cache-based decoder that the memory figures describe is not implemented.

### Using an SVF expert

```python
import argparse, torch
from morph.eval.quality_bench import load_model
from morph.model.svf import apply_svf, set_expert, load_expert

model = load_model(argparse.Namespace(ckpt=None, weights="morph-s/model.safetensors",
                                      config="morph-s/config.json"), torch.device("cpu")).eval()
model.self_model = None          # the self-modeling head is only used in training
apply_svf(model)                 # one scale vector per weight matrix
set_expert(model, load_expert("morph-s/svf/code.safetensors"))
```

To run the two-pass setup (router picks the expert, then the model runs with it), see
`python -m morph.train.svf_train evaluate --help`. The router file is a pickled torch
dictionary, so loading it needs `weights_only=False`; only load it from this repo.

## Architecture

```
tokens -> embedding (49,152 x 640, tied with the output head)
  24 layers, each: pre-norm mixer + pre-norm SwiGLU MLP (1728), residual
    20 x Mamba2 mixer   (20 heads x 64, d_state 128, conv width 4, chunked SSD)
     4 x MLA attention  (layers 4, 9, 14, 19; 10 heads x 64; no positional encoding)
  RMSNorm -> LM head
         \-> self-model head (training only): predicts the states entering layers 4, 9, 14, 19
```

- **Few attention layers, mid-stack.** One attention layer per five Mamba2 layers, with Mamba2
  at both ends of the stack. Hybrid ablations in the literature find that front-loaded attention
  hurts.
- **Latent attention without positions (NoPE MLA).** Keys and values are up-projections of a
  128-dimensional latent, which is the only tensor cached per token. The Mamba2 layers carry
  position information.
- **Cross-layer sharing.** Layer 9 reuses the latent produced at layer 4, and layer 19 reuses
  the one from layer 14, each with its own up-projection. That leaves two cached latents per
  token: 512 bytes per token in fp16.
- **Mamba2 in plain PyTorch.** The chunked SSD algorithm is written in PyTorch with
  `mamba_ssm`-compatible parameters, so it runs on a T4 without building CUDA kernels.

Inference memory in fp16, for one sequence:

| | per token | fixed | at 32k tokens |
|---|---|---|---|
| Multi-head attention, same depth and heads | 61,440 B | none | 2,013 MB |
| MORPH-S | 512 B | 6.7 MB of SSM state | 23.5 MB |

## Training

**Pretraining.** FineWeb-Edu (sample-10BT), SmolLM2 tokenizer, 3.0B tokens in 11,445 steps of
128 sequences x 2048 tokens. Muon for matrices and AdamW for embeddings, norms and heads. Peak
learning rate 2e-3, 500 warmup steps, constant, then a 1 - sqrt(progress) decay to zero over the last 20% of tokens
(WSD). Weight decay 0.1, z-loss 1e-4, fp16 with loss scaling, a fused chunked cross-entropy
and gradient checkpointing. About 8 to 9k tokens per second on two T4s, spread over a dozen
Kaggle sessions of up to 11.5 hours. Training resumed exactly from time-based checkpoints at
the start of each session.

**Self-modeling loss.** The self-model head is a linear layer on the final hidden state. It
predicts the RMS-normalised residual states entering the four attention layers, and its
squared error is added to the loss with weight 1.0. Gradients flow into the targets as well as
the predictions, which follows [Premakumar et al. (2024)](https://arxiv.org/abs/2407.10188).
The head is not used at inference.

**Instruction tuning.** 0.15B tokens from `HuggingFaceTB/smol-smoltalk` formatted as ChatML
(`<|im_start|>role ... <|im_end|>`), loss on assistant replies only (72% of tokens), peak
learning rate 3e-4, one session of about 5 hours. The self-model loss stays on.

**SVF experts.** The base model is frozen. Each linear layer in the blocks (126 matrices) is
decomposed with an SVD, and an expert is one vector per matrix that rescales the singular
values. Each expert was trained for 300 steps on about 8M tokens of one domain (open-web-math,
codeparrot-clean, TinyStories).

## Results

### Language modeling and chat

| | FineWeb-Edu val ppl | HellaSwag (0-shot) | reply ppl on smol-smoltalk val |
|---|---|---|---|
| Base | 17.4 | 32.7% | not applicable |
| Chat (SFT) | not measured | 32.4% | 4.07 |

HellaSwag uses all 10,042 validation items and picks the ending with the highest mean
per-token log-likelihood. Chat tuning cost 0.4 points.

The instruction-tuned model follows the chat format and gives a coherent answer to simple
factual questions. It fails at constrained tasks: asked for a haiku about the sea, it produced
a repeating numbered list. All three test replies are in `eval/sft_s_samples.txt`.

### SVF experts and two-pass routing

Cross-entropy on 64 held-out 1024-token windows per domain (lower is better). Two-pass means
the router picks the expert from the first 128 tokens.

| text | base | its own expert | two-pass | change vs base | router accuracy |
|---|---|---|---|---|---|
| general | 2.813 | n/a (base) | 2.817 | +0.1% | 91% |
| math | 3.206 | 2.990 | 3.005 | -6.3% | 84% |
| code | 3.353 | 2.761 | 2.765 | -17.5% | 98% |
| stories | 2.435 | 1.955 | 1.955 | -19.7% | 100% |

An expert left on all the time costs general text between 1.1% (math) and 6.2% (stories),
which is why the router has a "general" option that applies no expert.

### Does self-modeling help?

This was tested on a 44M-parameter proxy (0.5B tokens, one seed per setting) before the main
run. Details are in
[docs/results/proxy_ablation.md](https://github.com/dhairya-pandya/MoRPH/blob/main/docs/results/proxy_ablation.md).

- With weight 1.0 it lowers the effective rank of the states feeding the cache (328 to 301 at
  the first attention layer) and costs 0.18% validation loss.
- Detaching the targets removes the effect and costs 1.7%, so the mechanism needs the gradient
  into the targets.
- It did not make a halved cache latent cheaper. With latent size 32, self-modeling added
  0.52% loss over the same-size baseline.
- The local learning coefficient estimated with SGLD roughly doubled with self-modeling. The
  original paper found the opposite for small classifiers.

I did not train a MORPH-S without the self-modeling loss, so the main run does not show
whether it helps at this scale. Mixture-of-Depths was also implemented and tested on the proxy.
It gave no speed-up on a T4 and cost 1 to 2% loss, so MORPH-S does not use it.

## Intended use and limitations

**Intended use.** Studying hybrid architectures, small-model training on free hardware,
self-modeling and SVF-style adaptation. The code, configs and per-run results are public so the
experiments can be repeated.

**Not for.** Anything where a wrong answer matters: medical, legal, financial or safety
decisions, or factual lookup. It has had no safety training or red-teaming, no refusal tuning,
and no evaluation of bias or toxicity.

**Known limitations.**
- Weak knowledge and reasoning, as expected from 3B training tokens at 170M parameters. It
  invents facts, repeats phrases and loses track in long answers.
- English only. The tokenizer and both training sets are English.
- The training context is 2048 tokens. Long-context behaviour was not tested, and with four
  attention layers, exact recall of distant tokens is likely weaker than in a full-attention
  model of the same size.
- The 32k-context memory figures are computed from the architecture (`morph.eval.kv_bench`),
  not measured with a production decoder.
- The only benchmark is HellaSwag. Nothing was run on MMLU, ARC, code or math tasks.
- The self-modeling findings come from single-seed runs at 44M parameters.
- SVF experts were evaluated on cross-entropy, not on task accuracy.

## Reproducing

Everything was run with `scripts/launch_kaggle.py` on Kaggle T4x2 notebooks:

```bash
python scripts/launch_kaggle.py prep          # tokenise FineWeb-Edu (CPU)
python scripts/launch_kaggle.py main          # one pretraining session, resumes automatically
python scripts/launch_kaggle.py sft-prep && python scripts/launch_kaggle.py sft
python scripts/launch_kaggle.py exp svf-s     # SVF experts and router
```

Configs are in `configs/`, and results with their setups are in `docs/results/`.

## Citation

```bibtex
@misc{pandya2026morph,
  title  = {MORPH-S: a small Mamba2 and latent-attention hybrid trained with self-modeling},
  author = {Dhairya Pandya},
  year   = {2026},
  url    = {https://github.com/dhairya-pandya/MoRPH}
}
```

Built on: Mamba2 (Dao and Gu 2024), DeepSeek-V2 MLA, cross-layer attention (Brandon et al.
2024), Kimi Linear (arXiv 2510.26692), the hybrid design study (arXiv 2510.04800), Muon and
Moonlight (arXiv 2502.16982), Mixture-of-Depths (Raposo et al. 2024), Transformer-squared and
SVF (Sakana AI), self-modeling (Premakumar et al., arXiv 2407.10188), the SmolLM2 tokenizer,
FineWeb-Edu and smol-smoltalk.
