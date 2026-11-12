# Stage 4: instruction tuning MORPH-S

Base: the finished main run (3B tokens, val ppl 17.4). Data: HuggingFaceTB/smol-smoltalk in
ChatML, loss on assistant replies only (72% of tokens). 0.15B tokens, 1,145 steps of
64 × 2048, Muon + AdamW, peak LR 3e-4 with 100 warmup steps and decay over the last 30%,
self-model λ = 1.0 kept on. One Kaggle T4×2 session, about 5 hours.

| step | val CE (assistant tokens) | val ppl |
|---|---|---|
| 200 | 1.663 | 5.27 |
| 400 | 1.546 | 4.69 |
| 600 | 1.486 | 4.42 |
| 800 | 1.446 | 4.24 |
| 1000 | 1.413 | 4.11 |
| 1145 | **1.405** | **4.07** |

HellaSwag (10,042 items, zero-shot): base 32.7% → SFT 32.4%. Instruction tuning costs 0.4
points of this completion benchmark, a small price for following the chat format.

Samples (temperature 0.7, top-p 0.9, 200 new tokens, system prompt "You are a helpful
assistant."): [samples/sft_s_samples.txt](samples/sft_s_samples.txt).
- **What is photosynthesis?** A coherent, mostly correct explanation: light to chemical
  energy, ATP and NADPH, CO2 to glucose, storage as starch. Some repetition at the end.
- **Write a haiku about the sea.** Ignores the 5-7-5 form and falls into a numbered list that
  repeats itself, a typical failure of small models.
- **Three tips for learning Python.** Follows the requested format exactly (three bold-titled
  tips and a closing line); the content overlaps between tips.

Note: the first SFT launch silently resumed the pretraining checkpoint (found through
`resume_dirs`) and skipped training. Runs now resume only from checkpoints under their own
run name (`morph/train/checkpoint.py:find_resume`).
