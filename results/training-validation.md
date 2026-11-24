# Training validation results (Sections 9.4 and 9.5)

All new runs use libtt 0.2.2 (2a4333a9) unless noted; logs are on the QuietBox under `~/tt-work/techreport/exp`
(paths relative to it below) and `~/tt-work/techreport/fc`.

| Item | Ran | Headline | Logs |
| --- | --- | --- | --- |
| S0 facts of the published SFT run | yes | BF16 parameters and AdamW state; 38,915 MetaMathQA samples; 451 of the Base model's 471 GSM8K errors hit the 512-token cap | `s0/` |
| S1 do BF16 updates survive? | yes | No: at the peak LR ~90% of feed-forward and attention weights keep their value per step; 0.2.2 reproduces the published losses exactly | `logs/s1c_9b.log`, `s1/` |
| S2 precision A/B (Qwen3.5-2B) | yes, plus an FP32 arm at LR 5×10⁻⁶ | BF16 storage acts as a smaller LR: FP32 at the same LR ends worse (GSM8K 72.9 vs 75.0%), at ¼ the LR slightly better (75.4%) | `logs/s2_*.log` |
| S3 reference run | CPU, scaled down (user's choice: no TPU) | Pass: chips vs CPU 0.0063 per step < Y/2 = 0.0092; the offset was a fused-loss bias, diagnosed and fixed | `logs/s3_*.log` |
| S4 cross-evaluation | teacher-forced instead of generation (user's choice; ~12 h on the CPU) | CPU picks the chips' token at 98.6% (Base) / 99.7% (fine-tuned) of positions; all differences near-ties | `eval/out/tf_*` |
| S5 format controls | yes | Most of the 23-point GSM8K gain is length and format: at 1,024 tokens the Base solves 82.9% vs 87.4% | `eval/out/results.jsonl` |
| S6 evals during training | replaced (user: function calling) | ToolACE held-out loss every 25 steps instead | `logs/fc_9b_toolace.log` |
| S7 unrelated test set | replaced | BFCL for the function-calling run | `../fc/out/` |
| S8 seeds (SFT) | no (time) | — | — |
| S9 longer run, step breakdown | no; a 4×-accumulation ToolACE run instead (user) | see "Function calling" | `logs/fc_9b_toolace_ga4.log` |
| Function calling (ToolACE → BFCL) | yes (user's addition) | BFCL 62.5% → 85.6% (same prompt), 91.0% (non-thinking prompt); Base at most 73.2% with format forgiven | `logs/fc_9b_toolace.log`, `../fc/out/` |
| Fused-loss fix (libtt) | yes | Logged loss was +0.015 per ignored token (BF16 log V) and +0.0009 from a BF16-precision exp; fixed on libtt branch `cross-entropy-accurate-exp` | `logs/s3_diag*`, `logs/test_*` |
| R0 facts of the published RL run | yes | Gain is not format: step 0 boxes 99%; wrong values 544 → 262 | `r0/` |
| R1 seeds | 2 new seeds × 30 steps (user's choice) | Step 30: 68.6% and 67.1% (seeds 1235, 1236; libtt 0.2.2) vs the published 70.7% (seed 1234, 0.2.1); step 10: 66.0% and 58.8% | `rl/runs/r1_*` |
| R2 zero-learning-rate control | yes | Pass: 31 empty transfers, byte-identical weights, 58.07% at steps 0 and 30 | `rl/runs/r2_lr0` |
| R3 reference run | CPU log-probability check only (no CPU Miles loop) | Trainer's batch-mean log-probability within 0.0003 of an FP32 CPU reference, engine within 0.0008 | `r3/` |
| R4 cross-evaluation of the policy | no (time) | — | — |
| R5 answer format | yes (rescoring) | Lenient scoring changes the endpoints by 0 / +5 answers | `r0/lenient.out` |
| R6 precision A/B (RL) | no (time) | — | — |
| R7 resume, harder target | no | — | — |

Gaps G1–G9 from the plan:

| Gap | Status |
| --- | --- |
| G1 reference run on other hardware | Partly closed: no TPU; a scaled-down SFT run (2B, 8,192 tokens a step, 30 steps) agrees with a CPU reference in both precisions (S3); for RL, only the trainer's log-probabilities are checked against the CPU (R3), not the loop's updates. **Open at full scale.** |
| G2 training continues after warm-up | Closed: S1/S2 explain the flat MetaMathQA curve (BF16 updates round away; FP32 at a matched rate is no faster); the ToolACE held-out loss falls through step 225 (0.302 at 25, 0.252 at 225). |
| G3 precision of parameters and optimizer state | Closed: BF16 for SFT (stated in 9.4), FP32 for RL, and why it matters (S1, S2). |
| G4 control for answer format (SFT) | Closed: S0, S5; for BFCL, the lenient reading. |
| G5 evaluations during training, unrelated test set | Closed differently (user's choice): held-out loss every 25 steps, and BFCL for the ToolACE run; no GSM8K curve along the MetaMathQA run (S6 dropped). |
| G6 more than one RL run | Partly closed: three runs to step 30 (R1); steps 30–90 remain one run. |
| G7 controls for the loop and for format (RL) | Closed: R2 (zero learning rate), R0 and R5 (format). |
| G8 evaluation on a reference backend | Partly closed: teacher-forced CPU comparison of serving (S4) and of the RL trainer (R3); no reference-trained checkpoint, no independent CPU accuracy. |
| G9 versions and patches in the text | Closed: 9.5 names the Miles patch and versions; every run records its versions. |


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
| Data | MetaMathQA revision aa4f34d3; 38,915 samples (9.9% of 395,000) in 300 steps, 9,830,400 token slots (each chip's 8,192-token row holds whole samples and padding), 7,156,732 of them counting toward the loss | replay of the data pipeline (`exp/s0/count_samples.py 300` counts 40,084 row starts, which include 1,169 padding runs; the sample count excludes them) |

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
predicts the changed fraction to within 0.25 point of what the chips store (within 0.1 point at the peak rate), so the loss comes from BF16 storage
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
One-step checks on the chips, step-1 loss (CPU reference 0.41595):

| Chips' loss path | Step-1 loss | − CPU |
| --- | --- | --- |
| Fused loss, libtt 0.2.2 (tt-train's cross-entropy kernel, #546) | 0.42341 | +0.0075 |
| PyTorch's cross-entropy instead | 0.41697 | +0.0010 |
| … and PyTorch attention instead of splash attention | 0.41677 | +0.0008 |
| Fused loss with the forward kernel's accurate exp | 0.42252 | +0.0066 |
| Fused loss with the accurate exp and log(V) exact for ignored rows | 0.41660 | +0.0007 |
| Fused loss with FP32 losses (libtt branch at bb56c4c6) | 0.41697 | +0.0010 |

Three causes. (1) Most of it: TorchTitan's fused loss zeroes the hidden state of every ignored token (prompt and
padding, label −100), whose loss is then exactly log(V), and subtracts log(V) per ignored token from the sum, which
assumes the kernel returns that value in FP32 as the TPU kernel does. tt-train's kernel returns BF16 losses:
log(248,320) = 12.4225 becomes 12.4375, 0.015 per ignored token, and with 0.39 ignored tokens per answer token in this
step's batch (2,313 against 5,879), +0.0059 per answer token, the measured 0.42252 − 0.41660 (about +0.17 for ToolACE, where 92% of tokens are prompt). libtt's wrapper
returned log(V) exactly for all-zero rows (first commit, 56d7f486); the branch now returns FP32 losses from the kernel
instead (bb56c4c6). (2) The rest: on Blackhole the kernel's sum of exponentials used a
BF16-precision exp (`_sfpu_exp_21f_bf16_tti_`); the forward now uses the accurate FP32 exp. The backward
normalizes the softmax by its own sum and ignores the saved log-sum-exp, so gradients were never affected (step-1
gradient norms: 19.50 fused, 19.63 with PyTorch's loss, 19.38 on the CPU). (3) Found with FP32 losses: the kernel's
FPU row sum and final add read FP32 operands as TF32, truncating every loss by about 4e-4; bb56c4c6 finishes each row
in FP32 on the SFPU, and the fused loss then equals PyTorch's cross-entropy on the chips (0.41697; log
`logs/s3_diag_celoss_f32.log`, libtt loaded from the branch's wheel, `run_tr_wheel.sh`). The ToolACE run below uses fixes (1) and (2) as in
56d7f486 (the kernel as a `TT_METAL_KERNEL_PATH` overlay, the wrapper patched in place), so its logged losses read
about 0.0004 low from (3), at most one in the third decimal of the losses the report quotes; the MetaMathQA runs and the published
curve carry the bias.

## S2: precision A/B on Qwen3.5-2B-Base (BF16 storage acts as a smaller learning rate)

The published 9B recipe (MetaMathQA, 8,192 tokens per chip in whole samples of up to 2,048 tokens, 300 steps, warm-up 25 steps, cosine decay) on
Qwen3.5-2B-Base (the untied copy), on libtt 0.2.2, three arms that differ only as listed (`exp/logs/s2_*.log`; update
snapshots `exp/s2/*`; GSM8K `exp/eval/out/results.jsonl`, zero-shot, 512 tokens, as published). The held-out loss is
on 626 MetaMathQA samples the 300-step data pipeline never reads (a fixed 1/400 by hash of the query, minus all 30,468
queries the pipeline processes, `exp/s0/consumed_queries.py`); the training data are the published stream. Each
validation reads 4 × 8,192 tokens per rank, the same batches each time: about 85% of the held-out samples.

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

## R5: control for answer format (RL; no new compute)

The prompt's system message already asks for the answer in `\boxed{}` (R0), so the "instruction to box" setting is the
published one. A lenient scorer that falls back to the response's last number when there is no `\boxed{}`
(`exp/r0/lenient.py`) changes the endpoints by 0 and +5 answers: step 0 765 → 765, step 90 953 → 958. The 14.25-point
gain is not a formatting effect.

## S4: cross-evaluation of serving (teacher-forced; deviation from the plan)

As planned, the CPU reference would generate answers for the same 300 questions. Hugging Face transformers decodes
Qwen3.5's GatedDeltaNet layers token by token in reference PyTorch code on the CPU: one batch of 64 questions had not
finished after 70 minutes, putting the two 9B models at about 12 hours. With the user's agreement (2026-11-24) the
check is teacher-forced instead (`exp/eval/tf_agree.py`): the chips' greedy completions (libtt 0.2.2, SGLang-TorchTPU,
zero-shot, 512 tokens; token ids and the server's log-probabilities saved) are run through the CPU reference
(transformers, BF16 weights, one forward pass per completion) on the fixed 300-question subset (`subset300.json`,
seed 0), and at every completion position the CPU's most likely token is compared with the chips' choice. This tests
whether serving on the chips computes the same next-token choices as an independent implementation; it does not give
an independent CPU accuracy.

| Model (300 questions) | Tokens | Chips' token = CPU's top choice | Completions the CPU reproduces token for token | CPU margin where they differ (median / 90th pct / max) | Mean abs. log-prob. difference |
| --- | --- | --- | --- | --- | --- |
| 9B Base | 125,033 | 98.58% | 6% (median completion 451 tokens) | 0.125 / 0.375 / 0.625 | 0.018 |
| 9B fine-tuned on MetaMathQA (`sft300b`) | 41,087 | 99.70% | 67% (median 126 tokens) | 0.125 / 0.125 / 0.375 | 0.005 |

Yardstick (first 50 of the 300 questions, 9B Base, 20,376 positions): chips vs CPU in BF16 98.63%, chips vs CPU in
FP32 98.54%, CPU in BF16 vs CPU in FP32 99.50%. Where a completion first diverges, the CPU's top two tokens are
nearly tied at BF16 resolution (logits of this size are 0.125 apart in BF16; only first divergences were recorded:
282 of about 1,780 differing positions for the Base model, 100 of about 124 for the fine-tuned one), so serving
agrees with the reference within reduced precision; it is measurably coarser than BF16 arithmetic on the CPU (1.4%
vs 0.5% of positions differ). Free-running greedy decoding therefore diverges
from the reference somewhere in most long completions, as any two BF16 backends do. The libtt 0.2.2 accuracies of the
same checkpoints on all 1,319 questions are 64.44% (Base; published 64.29% on 0.2.1) and 87.34% (fine-tuned; published
87.34%), so serving the published checkpoints on 0.2.2 reproduces the published numbers within two answers.
Limitation (G8): with no full-scale reference training run, there is no "fine-tuned on the reference" row.

## Function calling: Qwen3.5-9B-Base fine-tuned on ToolACE, evaluated on BFCL (the report's SFT result)

Added at the user's request (2026-11-24), replacing the planned MetaMathQA rerun as the report's fine-tuning result.

- **Data.** ToolACE (Team-ACE/ToolACE, Apache-2.0, 11,300 samples): the system prompt (instructions and the functions in
  JSON) with each sample's first exchange, the user's request and the assistant's reply, a Python-style list of calls
  such as `[get_weather(city="Paris")]` or a statement that no function applies (`toolace_messages`). TorchTitan's
  ChatProcessor takes a [user, assistant] pair; `ttx`'s `SystemChatProcessor` also takes a leading system message and
  masks the loss up to the reply. 32 samples longer than 2,048 tokens are dropped; mean 698 tokens, 56 of them the
  reply. A fixed 2% (225 samples, by hash of the request) is held out; each validation scores about 85% of it (4 × 8,192
  tokens per rank, the same batches each time).
- **Recipe.** The published 9B recipe otherwise unchanged (`ttx.qwen35_9b_sft_toolace`): 8,192 tokens per chip (whole samples
  of up to 2,048 tokens), AdamW at 2×10⁻⁵, 25 warm-up steps, cosine decay, 300 steps (about 1.25 passes over the data:
  a replay of the pipeline reads 13,855 samples, of 11,046 in the training split), BF16 parameters and
  AdamW state; libtt 0.2.2 with both fused-loss fixes (S3), so the logged losses are unbiased. Log
  `exp/logs/fc_9b_toolace.log` (versions in `.versions`).
- **Run.** 29.47 s per step after the first (the MetaMathQA run's 29.6 s: the same token count), 2.5 hours. Held-out
  loss 0.4475 after step 1, 0.3016 at step 25, 0.2677 at 100, 0.2539 at 200, 0.2520 at 300 (flat from step 225 as the
  rate decays). Mean training loss of the last ten steps 0.167.
- **Evaluation.** BFCL v4's five single-turn categories (1,240 tests) with bfcl-eval 2026.3.23's own prompt for
  prompting models (`system_prompt_pre_processing_chat_model`: instructions, the required output format, the functions
  in JSON), its decoder (`default_decode_ast_prompting`) and AST checker (`ast_checker`; irrelevance passes when the
  reply holds no decodable call), rendered with the recipe's chat template; SGLang-TorchTPU on four chips, libtt
  0.2.2, greedy, 512 tokens (`fc/bfcl_prompts.py`, `fc/bfcl_eval_tt.py`). The scorer was checked with replies built
  from BFCL's possible answers (98–100% pass; the misses are the builder's) and with wrong replies (all fail;
  `fc/scorer_check.py`, output `fc/out/scorer_check.out`).

| Category (tests) | Base, thinking prompt | lenient | Base, non-thinking | lenient | Fine-tuned, thinking | Fine-tuned, non-thinking |
| --- | --- | --- | --- | --- | --- | --- |
| simple (400) | 58.25% | 67.0% | 59.0% | 61.25% | 84.75% | 93.5% |
| multiple (200) | 63.0% | 70.5% | 53.5% | 62.5% | 86.0% | 95.0% |
| parallel (200) | 51.5% | 70.5% | 42.5% | 47.0% | 85.0% | 91.5% |
| parallel multiple (200) | 45.0% | 64.5% | 41.0% | 56.0% | 77.0% | 83.0% |
| irrelevance (240) | 94.6% | 93.3% | 85.4% | 84.6% | 95.0% | 92.1% |
| **mean of the five** | **62.5%** | **73.2%** | **56.3%** | **62.3%** | **85.6%** | **91.0%** |

The two prompts differ only in the generation prompt's last tokens: the template's default opens a thinking block
(`<think>\n`); its non-thinking mode (`enable_thinking=False`, `fc/bfcl_prompts_mode.py`) ends in an empty block
(`<think>\n\n</think>\n\n`), exactly as the training sequences hold it, so prompt and reply tokenize as in training.
Lenient (`fc/lenient.py`) is the format control: the last bracketed span anywhere in the reply that decodes as a list of
calls, after removing the format template's placeholders ("func_name1=", "params_name1=") that the Base model copies
into otherwise correct calls, checked by the same AST checker; it changes nothing for the fine-tuned model.

The Base model's failures are mostly unreadable replies (thinking prompt: 375 of 461, the template copied after correct
reasoning, or placeholders inside calls; non-thinking: 434 of 525, placeholders inside calls). Forgiving its format
lifts it to 73.2% at best, so fine-tuning adds 12.4 points beyond answer format with the same (thinking) prompt and
17.9 points with each model at its best (91.0% vs 73.2%). With the thinking prompt the fine-tuned model writes a
rationale first (mean 205 tokens; 41 replies reach the cap) and its failures include 95 replies with no decodable call,
40 cut off at the cap and most of the rest pointing out a required parameter the question omits (as both prompts
instruct) where BFCL's answer expects a call;
with the non-thinking prompt it answers directly (mean 39 tokens; one reply reaches the cap) and its 106 failures are
73 calls with a wrong function or value, 14 replies with no decodable call and 19 irrelevance tests where it calls.

Prompt detail: the template's generation prompt opens a thinking block (`<think>\n`) while the training sequences held
an empty one (`<think>\n\n</think>\n\n`, whose first newline merges with the prompt's in tokenization; TorchTitan's
ChatProcessor masks by prompt length the same way). The ToolACE model therefore writes a short rationale and closes
the block before its call (1,223 of 1,240 replies); the MetaMathQA model writes its solution inside the open block.
Both models in each comparison see the same prompt.

## S5: format controls for the MetaMathQA GSM8K gain (most of the published gain is length and format)

Every setting applies to both models: the published 9B Base and its published MetaMathQA fine-tune (`sft300b`), served
on libtt 0.2.2 (`exp/eval/out/results.jsonl`; prompts `exp/eval/prompts.py`). Lenient = the published scorer ("The
answer is" N, else the last number); strict = only "The answer is" N. Instruction: one line, `End your response with
"The answer is N", where N is the final answer as a number.` 8-shot: Wei et al.'s eight GSM8K chain-of-thought
exemplars (which end "The answer is N.") as earlier chat turns.

| Setting (1,319 questions) | Base lenient | Base strict | Base at cap | Fine-tuned lenient | Fine-tuned strict | Fine-tuned at cap |
| --- | --- | --- | --- | --- | --- | --- |
| zero-shot, 512 tokens (published) | 64.44% | 1.06% | 611 | 87.34% | 87.34% | 6 |
| instruction, 512 | 77.26% | 72.18% | 528 | 86.05% | 85.97% | 6 |
| 8-shot, 512 | 71.87% | 58.98% | 461 | 85.22% | 85.22% | 1 |
| zero-shot, 1,024 tokens | 82.87% | 1.29% | 239 | 87.41% | 87.41% | 5 |

The Base model reasons at length and rarely writes "The answer is": with room to finish (1,024 tokens) it solves 82.87%
by the published scorer, 4.5 points below the fine-tuned model's 87.41%. With the answer format asked for, its strict
score rises from 1.06% to 72.18%, 13.8 points below the fine-tuned model under the same instruction. The published
23.05-point gain is therefore mostly the 512-token cap and answer format; the gain in solving that survives the
controls is about 4.5 points (zero-shot, 1,024 tokens), while the fine-tuned model is far more concise (mean 143 against
567 tokens) and reliably formatted. The fine-tuned model is insensitive to the settings (85.2–87.4%).

## R3: reference for the RL loop (CPU log-probability check; the full loop on a CPU was out of reach)

Miles' trainer runs only on CUDA or TPU platforms, so a CPU reference of the whole loop would mean porting Miles; with
the user's choice of scaled-down CPU references (2026-11-24) this check covers the numbers the loop's update depends
on instead. The seed runs save every rollout batch (`--save-debug-rollout-data`); for the first batch of run `r1_s2`
(128 answers, 28,084 tokens, sampled by the step-0 policy on the chips), `exp/r3/cpu_logprobs.py` recomputes each answer
token's log-probability on the CPU (transformers, FP32 weights and arithmetic, the original Qwen3-0.6B) and compares
with the serving engine's (SGLang-TorchTPU, BF16), saved with the batch; Miles logs its trainer's (TorchTitan on the
chips) for the same batch.

| Batch mean log-probability (mean over answers of each answer's mean) | Value | − CPU |
| --- | --- | --- |
| CPU reference, FP32 | −0.34197 | |
| Trainer on the chips (Miles' `rollout/log_probs`) | −0.34223 | −0.0003 |
| Engine on the chips (`rollout/rollout_log_probs`; recomputed from the batch: −0.34114) | −0.34114 | +0.0008 |
| Same for the first batch of `r1_s3` (23,138 tokens): CPU −0.29030, trainer −0.29043, engine −0.28972 | | −0.0001 / +0.0006 |

Per token, the CPU and the engine differ by 0.0166 on average (4.2% of tokens by more than 0.1, at most 2.8; `r1_s3`:
0.0152); Miles reports 0.0185 (`r1_s3`: 0.0165) between its trainer and the engine (`train/train_rollout_logprob_abs_diff`). The trainer's
log-probabilities, which set GRPO's importance ratios, agree with the FP32 reference to 0.0003 in the batch mean; the
engine's sampling numbers are the coarser ones, as in S4. Not covered: a reference of the loop's updates (G1 for RL).

## R1: seeds (two new seeds, 30 steps)

The published setup on libtt 0.2.2 (training and serving), 30 steps instead of 90 (user's choice, 2026-11-24, to fit the
time), test evaluations of all 1,319 questions after steps 10, 20 and 30 (greedy, 1,024 tokens); every run starts from
the same policy, whose step-0 accuracy on 0.2.2 is R2's 58.07%. Seeds differ in both Miles seeds (`--seed`,
`--rollout-seed`: sampling and question order). Runs `exp/rl/runs/r1_s2` and `r1_s3` (versions, rollouts and
evaluations saved).

| Run | Step 0 | Step 10 | Step 20 | Step 30 |
| --- | --- | --- | --- | --- |
| Published `long3` (seed 1234 / 42, libtt 0.2.1) | 58.00% | — | — | 70.74% |
| `r1_s2` (seed 1235 / 43) | 58.07% (R2) | 66.03% | 66.41% | 68.61% |
| `r1_s3` (seed 1236 / 44) | 58.07% (R2) | 58.83% | 63.38% | 67.10% |

After 30 steps the three runs span 67.1–70.7% (mean 68.8%), all well above the zero-learning-rate control's 58.07%;
early progress varies more between seeds (58.8–66.0% after 10 steps). The 95% binomial half-width at these accuracies
is about 2.5 points, so the published run's 70.7% is within the spread of a single run. The later points of the
published run (72.4% at 60 and 90) remain single-run results. Each run took 1.5 to 1.6 hours.
