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

## S1: do the BF16 weight updates survive? (lost-update hypothesis: confirmed)

The published 9B recipe on libtt 0.2.2, from the Base checkpoint, stopped after 30 of its 300 steps (log
`exp/logs/s1c_9b.log`). After steps 1, 5, 10, 25 and 30, the hook in `exp/ttx/config_registry.py` read 61 tensors
(2.89 billion elements: embeddings, LM head, final norm, and every tensor of layers 0 and 16 (GatedDeltaNet) and 3 and
31 (full attention)) before and after the optimizer step, and recomputed AdamW's update u in FP32 from the optimizer
state. Analysis `exp/s1/analyze.py s1` (full output `exp/s1/analyze.out`).

The run's losses equal the published run's (`sft300b`, libtt 2fe91a3f) at all 30 steps to the five printed digits,
so libtt 0.2.2 reproduces it.

| Step | LR | Elements changed | of feed-forward | of attention projections | of embeddings / LM head | update / weight (stored) | update / weight (AdamW's u) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8.0×10⁻⁷ | 0.49% | 1.75% | 1.30% | 0.01% / 0.03% | 4.4×10⁻⁶ | 3.0×10⁻⁵ |
| 5 | 4.0×10⁻⁶ | 2.2% | 7.8% | 5.6% | 0.07% / 0.18% | 4.8×10⁻⁵ | 1.2×10⁻⁴ |
| 10 | 8.0×10⁻⁶ | 2.7% | 8.9% | 6.4% | 0.13% / 0.77% | 6.6×10⁻⁵ | 1.6×10⁻⁴ |
| 25 | 2.0×10⁻⁵ (peak) | 3.6% | 10.7% | 8.2% | 0.30% / 1.65% | 1.1×10⁻⁴ | 2.1×10⁻⁴ |
| 30 | 2.0×10⁻⁵ | 3.3% | 9.6% | 7.4% | 0.30% / 1.43% | 9.9×10⁻⁵ | 1.9×10⁻⁴ |

(Attention projections: the GatedDeltaNet and full-attention projections together; ratios are norms over all 61
tensors.) At the peak learning rate, about 89% of feed-forward and 92% of attention weights, and over 98% of the
embedding and LM-head weights, keep their stored value in a step: AdamW's update is smaller than half a BF16 unit in
the last place. The stored change is about half the norm of AdamW's update. One rounding of (weight − u) into BF16
predicts the changed fraction to within 0.1 point of what the chips store, so the loss comes from BF16 storage
itself, not from the device's arithmetic. The hypothesis holds: with BF16 parameters and AdamW state, most updates
of this recipe round away, and S2 measures what that costs.

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

## R2: zero-learning-rate control (pass)

The published RL setup on libtt 0.2.2 (training and serving), seeds 1234 / 42, for 30 steps with the learning rate
set to 0 (`exp/rl/runs/r2_lr0`; versions in `versions.txt`). Rollouts, rewards, the GRPO loss, the backward pass,
AdamW's step and the disk-delta weight transfer all run as in the published run.

- All 31 weight transfers (the initial sync and one after each step) report 0.00% of BF16 weights changed and
  0.00 GB sent. After step 30 the engine's checkpoint file is byte-identical to the original Qwen3-0.6B weights
  (MD5 `fed58aa6b81fb7e1…` for both).
- Test accuracy (all 1,319 questions, greedy, 1,024 tokens): 766 correct (58.07%) at step 0 and at step 30, with the
  same truncation rate (0.76%). Mean response length 191.3 and 191.1 tokens: a few greedy responses differ slightly,
  as batch composition changes the server's arithmetic, but no answer's correctness changes.
- The mean training reward stays at 0.572 over the 30 steps (no learning).
- Step 0 on libtt 0.2.2 (58.07%) matches the published step 0 on 0.2.1 (58.00%, 765 correct) within one answer.

Both criteria hold: the loop itself (rollouts, transfer, reload) does not change the policy.

## S3: reference run on the CPU (pass; with a constant offset from the fused loss)

No TPU was available (user's decision, 2026-11-24): the reference is the CPU, so the recipe is scaled down. Qwen3.5-2B
Base (an untied copy of the checkpoint: `lm_head.weight` = the embeddings, as TorchTitan's 2B model keeps a separate
output layer), one 2,048-token sequence per rank on four ranks (8,192 tokens a step instead of 32,768), the 300-step
learning-rate schedule stopped after 30 steps, BF16 and FP32 parameters and AdamW state. The chips run
`ttx.qwen35_2b_sft_{bf16,fp32}_small`; the CPU runs the same recipe as four gloo ranks with PyTorch's cross-entropy
and attention (`*_small_cpu`; logs `exp/logs/s3_{tt,cpu}_{bf16,fp32}.log`; `exp/losses.py --diff`).

| Comparison (steps 1–30) | Mean abs. per-step difference | Max | Step 1 |
| --- | --- | --- | --- |
| Yardstick: chips, dataloader seed 42 vs 43 (BF16) | 0.0183 | 0.0615 | −0.0125 |
| Chips vs CPU, BF16 | 0.0063 | 0.0079 | +0.0075 |
| Chips vs CPU, FP32 | 0.0064 | 0.0078 | +0.0075 |

Both precisions pass (0.0063 < Y/2 = 0.0092). The difference is a near-constant offset (chips higher by +0.0063,
standard deviation 0.0009 over the 30 steps, in both precisions), present at step 1 before any update, and the
training trajectories otherwise agree: the effect of FP32 storage on the mean loss over steps 11–30 is +0.0302 on the
chips and +0.0303 on the CPU.

**The offset is a bias in the logged value of the fused loss, now fixed** (libtt branch `cross-entropy-accurate-exp`).
One-step checks on the chips, step-1 loss (CPU reference 0.41594):

| Chips' loss path | Step-1 loss | − CPU |
| --- | --- | --- |
| Fused loss, libtt 0.2.2 (tt-train's cross-entropy kernel, #546) | 0.42341 | +0.0075 |
| PyTorch's cross-entropy instead | 0.41697 | +0.0010 |
| … and PyTorch attention instead of splash attention | 0.41677 | +0.0008 |
| Fused loss with the forward kernel's accurate exp | 0.42252 | +0.0066 |
| Fused loss with the accurate exp and log(V) exact for ignored rows | 0.41660 | +0.0007 |

Two causes. (1) Most of it: TorchTitan's fused loss zeroes the hidden state of every ignored token (prompt and
padding, label −100), whose loss is then exactly log(V), and subtracts log(V) per ignored token from the sum, which
assumes the kernel returns that value in FP32 as the TPU kernel does. tt-train's kernel returns BF16 losses:
log(248,320) = 12.4225 becomes 12.4375, 0.015 per ignored token, and with 0.37 ignored tokens per answer token in this
MetaMathQA batch, +0.0056 per answer token (about +0.17 for ToolACE, where 92% of tokens are prompt). libtt's wrapper
now returns log(V) exactly for all-zero rows. (2) The rest: on Blackhole the kernel's sum of exponentials used a
BF16-precision exp (`_sfpu_exp_21f_bf16_tti_`); the forward now uses the accurate FP32 exp. The backward
normalizes the softmax by its own sum and ignores the saved log-sum-exp, so gradients were never affected (step-1
gradient norms: 19.50 fused, 19.63 with PyTorch's loss, 19.38 on the CPU). The ToolACE run below uses both fixes (the
kernel as a `TT_METAL_KERNEL_PATH` overlay, the wrapper patched in place); the MetaMathQA runs and the published
curve carry the bias.

## S2: precision A/B on Qwen3.5-2B-Base (BF16 storage acts as a smaller learning rate)

The published 9B recipe (MetaMathQA, 4 × 2,048 tokens per chip, 300 steps, warm-up 25 steps, cosine decay) on
Qwen3.5-2B-Base (the untied copy), on libtt 0.2.2, three arms that differ only as listed (`exp/logs/s2_*.log`; update
snapshots `exp/s2/*`; GSM8K `exp/eval/out/results.jsonl`, zero-shot, 512 tokens, as published). The held-out loss is
on 626 MetaMathQA samples the 300-step data pipeline never reads (a fixed 1/400 by hash of the query, minus all 30,473
queries the pipeline processes, `exp/s0/consumed_queries.py`); the training data are the published stream.

| Arm | Held-out loss: step 25 | step 100 | step 200 | step 300 | Stored change / weight at step 25 | GSM8K (strict = lenient) |
| --- | --- | --- | --- | --- | --- | --- |
| Base (no training) | — | — | — | — | — | 56.18% lenient, 0.76% strict |
| BF16 parameters and AdamW state, peak LR 2×10⁻⁵ (the 9B's) | 0.2133 | 0.2010 | 0.1920 | 0.1902 | 1.06×10⁻⁴ (3.3% of elements change) | 74.98% |
| FP32, BF16 compute, peak LR 2×10⁻⁵ | 0.2314 | 0.2250 | 0.2049 | 0.1930 | 1.68×10⁻⁴ (99.9%) | 72.93% |
| FP32, BF16 compute, peak LR 5×10⁻⁶ | 0.2145 | 0.2006 | 0.1927 | **0.1884** | 5.0×10⁻⁵ (99.7%) | **75.36%** |

With BF16 storage, the 2B loses updates as the 9B does in S1 (13% of feed-forward elements change at the peak LR).
That does not hold the recipe back: at the same learning rate, FP32 storage is worse until the rate has decayed and
ends behind (held-out 0.1930 vs 0.1902; GSM8K 72.9% vs 75.0%). At a quarter of the learning rate, FP32 tracks the BF16
curve and ends slightly ahead (0.1884; 75.4%). BF16 rounding thus acts as a smaller effective learning rate (its stored
change, 1.06×10⁻⁴ of the weight norm per step, lies between the two FP32 arms'), and the published rate suits it. The
2B Base model's GSM8K errors are mostly format, as the 9B's (S0): 673 of its 1,319 answers run to the cap and only 10
are correct with "The answer is".
