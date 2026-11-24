# Training validation results (Sections 9.4 and 9.5)

<!-- SUMMARY TABLE: filled in last -->

## Criteria written before the comparisons (2026-11-24 03:12 UTC)

- **S3 (SFT reference).** The scaled recipe (Qwen3.5-2B-Base, one 2,048-token sequence per rank, four ranks, the
  300-step LR schedule stopped after 30 steps) runs on the chips and on the CPU from the same checkpoint and data
  order, in BF16 and in FP32. Yardstick Y = the mean absolute per-step loss difference over steps 1–30 between two
  chip runs that differ only in the dataloader seed. Pass if, in each precision, the mean absolute per-step
  chip–CPU difference over steps 1–30 is below Y/2. Also reported: the step-1 difference and the maximum.
- **R2 (zero learning rate).** Pass if every weight transfer reports 0.00% changed BF16 weights, and the step-30
  test accuracy is within ±2.7 points (a 95% binomial interval at 58% and 1,319 questions) of step 0's.
- **S4/R4 (cross-evaluation).** The CPU reference (Hugging Face transformers, BF16, greedy) and libtt score the same
  prompts. Agreement holds if the CPU accuracy lies within the 95% binomial interval of libtt's on the same
  questions (±5.5 points at 300 questions); also reported: the share of identical extracted answers.
- **R1/R3 (seeds, RL reference).** No threshold: report the per-step mean and min–max across seeds, and compare the
  reference with that band.

## S0: facts of the published SFT run (no new compute)

The run is `sft300b` (log `~/tt-work/train35/logs/sft300b.log`, 2026-11-16).

| Fact | Value | Source |
| --- | --- | --- |
| Parameters, gradients, AdamW state | BF16 (`training.dtype = bfloat16`, `mixed_precision_param = bfloat16`), gradient reduction BF16 | `tt35/config_registry.py` `qwen35_9b`, config dump in the log |
| AdamW | lr 2×10⁻⁵, betas (0.9, 0.95), eps 10⁻⁸, weight decay 0.1, for-loop implementation; gradient clipping at norm 1.0 | `qwen35_9b_sft`, TorchTitan `default_adamw` |
| libtt | branch `p2p-peer-addresses` at 2fe91a3f (the point-to-point fix, later merged as #555), wheel built 2026-11-16 03:30 UTC | `~/tt-work/moe/wheels/p2p` |
| TorchTitan | google-pytorch/torchtitan 94063ded, unmodified; recipe module `tt35` | `git -C ~/tt-work/train35/torchtitan status` |
| Seeds | dataloader seed 42 (shuffle on), `deterministic = false` | config dump |
| Data | MetaMathQA revision aa4f34d3; 40,084 samples (10.1% of 395,000) in 300 steps, 9,830,400 tokens of which 7,156,732 count toward the loss | replay of the data pipeline: `exp/s0/count_samples.py 300` |

GSM8K errors of the saved completions (`exp/s0/classify.py`; tokens counted with the recipe's tokenizer; 512-token cap):

| Model | Correct | of which by the last-number fallback | Errors | at the 512-token cap | no "The answer is" | phrase with a wrong number | Completions at the cap | Mean tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Base | 848 | 834 | 471 | 451 | 20 | 0 | 611 | 422.6 |
| Fine-tuned (`sft300b`) | 1,152 | 0 | 167 | 6 | 0 | 161 | 6 | 140.2 |

The Base model rarely stops: 611 of its 1,319 completions run to the cap, and 451 of its 471 errors are such
completions. Only 14 of its correct answers use "The answer is"; the other 834 are scored by the last-number
fallback, 160 of them in completions that hit the cap. The fine-tuned model stops, always uses the phrase, and its
errors are wrong numbers. The published 23-point gain therefore mixes answer format and stopping with reasoning; S5
separates them.

## R0: facts of the published RL run (no new compute)

The run is `long3` (`~/tt-work/miles/runs/long3`, 2026-11-16/17); endpoint checks in `~/tt-work/rl-report-20261117`.

| Fact | Value |
| --- | --- |
| Miles | radixark/miles 7767e158 plus a two-file patch: `platforms/tpu.py` adds `disk-delta` to the TPU platform's weight-transfer modes; `weight_update/protocols/delta.py` replaces two `torch.cuda` calls with `torch.accelerator` / the platform's device |
| TorchTitan (Miles' trainer) | pytorch/torchtitan 45e33cd4 plus Miles' own `docker/tpu_patch/torchtitan_torch_tpu.patch`, unchanged |
| SGLang-TorchTPU | sgl-project/sglang-torchtpu d09a3418 and sglang 51911183, each with local patches (diff hashes in `provenance.json`) |
| libtt | training: a 0.2.1-labelled build of the `miles-attention` and `dense-flat` branches (later merged as #559 and #566); serving and the endpoint checks: 0.2.1 (39235edd) |
| Seeds | `seed = 1234`, `rollout_seed = 42`, rollout shuffle on |
| Precision | FP32 parameters and AdamW state, BF16 compute (`bf16 = True`), FP32 gradient reduction |
| Test scoring | the last `\boxed{}` of the response, compared numerically (Miles' `math` reward and an independent exact checker agree on all 2,638 endpoint answers) |
| Training questions | 90 steps × 16 = 1,440 distinct questions of GSM8K's 7,473 (shuffled, less than one epoch) |
| Prompt | the system prompt already asks for the answer in `\boxed{}` |

Test errors at the endpoints (`exp/r0/classify.py`, 1,024-token limit):

| Policy | Correct | Errors | hit the limit | no boxed answer | boxed, wrong value | Mean tokens |
| --- | --- | --- | --- | --- | --- | --- |
| Step 0 | 765 (58.00%) | 554 | 10 | 0 | 544 | 191.2 |
| Step 90 | 953 (72.25%) | 366 | 104 | 0 | 262 | 366.5 |

The step-0 policy already boxes 1,310 of its 1,319 answers. The gain is wrong values becoming right (544 → 262
wrong), not formatting; responses get longer and more of them reach the limit (10 → 104).
