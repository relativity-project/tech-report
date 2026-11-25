// Training with TorchTitan.
// Original throughput and short checks: libtt main at 5e041d48 and the Llama 3
// work in pull request #554.
// Fine-tuning (checked 2026-11-25 UTC; every number in results/training-validation.md with its script):
// ToolACE run ~/tt-work/techreport/exp/logs/fc_9b_toolace.log (libtt 0.2.2 2a4333a9 with the first version of the
// fused-loss fix, 56d7f486, later #568; the 0.2.2 release 2c68912a logs the same losses within 0.0003, F3; TorchTitan
// 94063ded, recipe ttx.qwen35_9b_sft_toolace); BFCL with bfcl-eval 2026.3.23 through SGLang-TorchTPU on the 0.2.2
// release 2c68912a, 1,024 tokens: ~/tt-work/techreport/fc/out_f2/results.jsonl and lenient.out (F2).
// MetaMathQA checks: S1 exp/logs/s1c_9b.log; S2 exp/logs/s2_*.log and exp/eval/out/results.jsonl; CPU reference
// exp/logs/f3_tt_*.log against exp/logs/s3_cpu_*.log (F3, release 2c68912a); serving exp/eval/out_f4/tf_results.jsonl
// (F4); S5 exp/eval/out/results.jsonl (9b-* tags); published run ~/tt-work/train35/logs/sft300b.log.

#import "fig-sft-loss.typ": fig-sft-loss

#let heavy = 0.8pt
#let light = 0.5pt

= Training <training>

Libtt runs TorchTitan's TPU training path on four Blackhole chips without changes to TorchTitan's source @torchtitan @torchtitantpu. This requires connecting the trainer to Tenstorrent's kernels, extending the compiler for gradient computations, and reducing training's memory requirements. The results below cover full-size model throughput and a completed fine-tuning run.

The trainer uses the TorchTPU compatibility layer described in @pytorch, with one process for each chip. Fully sharded data parallelism @simplefsdp divides the parameters, gradients, and optimizer state among the chips. In these four-chip runs, each chip keeps a quarter of each, gathers a layer's weights before using them, and receives its share of the summed gradients afterwards. Each chip trains on its own sequences.

== Training kernels

The TPU trainer uses hardware-specific Pallas kernels for attention and the combined final projection and loss. Libtt connects these operations to tt-metal's existing fused forward and backward kernels @ttmetal. Its compatibility layer adapts the trainer's calls, and the compiler carries them through to the kernels. For Qwen3.5’s gated RMS norm, libtt selects TorchTitan's ordinary PyTorch implementation.

The fused loss avoids storing a full tensor of vocabulary-wide logits. It processes tokens in chunks and recomputes each chunk's logits during the backward pass. Together with fused attention, this allowed Qwen3-8B to train with four sequences per chip instead of one, increasing throughput from 287 to 435 tokens per second per chip.

Libtt also extends the attention kernels for packed documents. A token may attend only to earlier tokens in its own document, leaving much of the attention matrix unused. The kernels recognize this structure and skip the masked blocks. For Llama 3 8B, this reduced a training step from 30.5 to 22.0 seconds.

== Backward pass

Training exposes costs that inference never sees, and libtt targets four of them: 

_Balancing embedding updates._ The gradient of a token embedding adds each token's gradient into the corresponding row of the embedding table, and real text uses some tokens far more often than others. libtt distributes the rows among cores in round-robin order, which balances these updates: Qwen3-1.7B's embedding gradient takes about 30 milliseconds instead of 238.

_Removing differentiation overhead._ libtt removes some of the work that automatic differentiation generates. The compiler combines gradients of adjacent slices into a concatenation, avoiding large zero-filled intermediate tensors, and expands the triangular solves in Qwen3.5’s recurrent layers into matrix products. Optimizer changes remove unnecessary broadcast constants, and matrix-product kernels use blocking suited to the large shapes found in training.

_Eliminating tile padding._ Tiled storage matters throughout the backward pass. Llama 3’s rotary-embedding expression creates a trailing dimension of two, which occupies 32 columns in tiled storage, a sixteenfold padding cost. Libtt rewrites the operation to work on separate halves of each head, which avoids that cost and preserves the values and gradients. This is one example of the layout work described in @compiler.

_Freeing collective memory._ Collectives need temporary storage, and libtt reduces it: Large reduce-scatter operations run in pieces so they no longer need one large contiguous allocation. This makes more of each chip's memory available to the model.

== Throughput and CPU checks <training-checks>

@training-runs reports throughput for full-size models on four chips.

#figure(
  placement: auto,
  caption: [Training on four chips. Parameters, gradients and AdamW's state in BF16. Text from C4 (for the Llama models, TorchTitan's small sample of it), packed into the sequences. For the Llama models, the mean of the steps after the first, which compiles the programs.],
  table(
    columns: 4,
    align: (left, right, right, right),
    table.hline(stroke: heavy),
    table.header(
      [Model], [Tokens per chip], [Seconds per step], [Tokens per second per chip],
    ),
    table.hline(stroke: light),
    [Qwen3-8B], [4 × 2,048], [18.8], [435],
    [Qwen3.5-9B], [4 × 2,048], [30.1], [272],
    [Llama 3 8B], [8,192], [22.0], [372],
    [Llama 3.2 3B], [8,192], [13.8], [595],
    [Llama 3.2 1B], [8,192], [5.82], [1,407],
    table.hline(stroke: heavy),
  ),
) <training-runs>

The checks against a CPU are separate from these full-size throughput measurements. Tested small models and selected layers are trained for a few steps from the same saved starting point on both the CPU and the chips. They include TorchTitan's debug models and the first two layers of Llama 3 8B with its full vocabulary and 8,192-token sequences. In these checks, the first losses agree to three digits or more (two, 0.25 per cent, for a fine-tuning batch whose prompts are masked), and subsequent losses differ by at most about one per cent. Different kernels, rounding, and summation order can produce small differences that accumulate across updates.

The DeepSeek-V3, Qwen3 MoE, and Qwen3.5 MoE debug models run unchanged in PyTorch eager mode with 32-bit floats. 

== Supervised Fine-tuning <fine-tuning>

Qwen3.5-9B-Base is fine-tuned to call functions on ToolACE @toolace: 11,300 examples whose system prompt lists functions in JSON and whose reply is the calls, as a Python list such as `[get_weather(city="Paris")]`, or a statement that none applies. The recipe adapts TorchTitan's Llama 3 fine-tuning example: a loss on the reply's tokens, whole samples of up to 2,048 tokens packed into 8,192 tokens per chip, 300 steps (1.25 passes over the data), AdamW at up to $2 times 10^(-5)$ with 25 warm-up steps and cosine decay, and BF16 parameters, gradients and optimizer state. A fixed 2 per cent is held out; weights load and save as Hugging Face checkpoints.

A step took 29.5 seconds (278 tokens per second per chip); the held-out loss falls from 0.448 after the first step to 0.302 by step 25 and 0.252 after 300 (@sft-loss).

#figure(
  placement: auto,
  fig-sft-loss,
  caption: [Fine-tuning Qwen3.5-9B-Base on ToolACE: training loss per step (grey), its 75-step centred mean (black) and held-out loss (circles).],
) <sft-loss>

Both models answer BFCL's five single-turn categories @bfcl (1,240 tests), scored by its own prompt, parser and syntax-tree checker and served greedily by SGLang-TorchTPU with up to 1,024 tokens.

#figure(
  placement: auto,
  caption: [Function calling on BFCL (per cent of tests passed) before and after fine-tuning, with the chat template's thinking and non-thinking prompts. Lenient reads any list of calls after removing copied template placeholders, a control for answer format. Mean: unweighted, over the five categories.],
  table(
    columns: 5,
    align: (left, right, right, right, right),
    table.hline(stroke: heavy),
    table.header([Category (tests)], [Base], [Base, lenient], [Fine-tuned], [Fine-tuned, non-thinking]),
    table.hline(stroke: light),
    [Simple (400)], [59.0], [67.8], [84.8], [93.5],
    [Multiple (200)], [63.5], [72.0], [86.0], [95.0],
    [Parallel (200)], [58.0], [76.0], [90.0], [91.5],
    [Parallel multiple (200)], [49.5], [70.5], [86.5], [83.0],
    [Irrelevance (240)], [94.2], [92.9], [95.0], [92.1],
    table.hline(stroke: light),
    [Mean], [64.8], [75.8], [88.5], [91.0],
    table.hline(stroke: heavy),
  ),
) <bfcl-results>

Fine-tuning raises the mean from 64.8 to 88.5 per cent with the same prompt, and to 91.0 per cent with the non-thinking prompt that matches the training format (@bfcl-results); read leniently, the Base model reaches at most 75.8 per cent.

We also checked the training itself on MetaMathQA @metamath, TorchTitan's own fine-tuning example. With BF16 parameters and optimizer state, many small updates are rounded away, but in effect this only lowers the learning rate: on a smaller model, the same training in FP32 ends no better. A scaled-down run on the chips closely follows the same run on a CPU, and in serving, the chips choose the same next token as a CPU reference at about 99 per cent of positions, with most differences at the limit of BF16 precision. The GSM8K improvement first reported for this run came mostly from shorter answers in the expected format.
#import "fig-rl-learning.typ": rl-learning-image

// Miles 7767e158 + two-file disk-delta portability patch. Original run: long3 (training: the miles-attention and
// dense-flat branches, later #559 and #566; serving libtt 0.2.1 39235edd). Endpoint checks: rl-report-20261117.
// Format breakdown: ~/tt-work/techreport/exp/r0/classify.py. Zero-learning-rate control and the seeds: libtt 0.2.2,
// ~/tt-work/techreport/exp/rl/runs/{r2_lr0,r1_s2,r1_s3} (versions.txt in each).
#block(breakable: false)[
== Reinforcement learning with Miles <reinforcement-learning>

Miles @miles is a reinforcement-learning framework that runs training and serving in one loop, which makes it a demanding test of framework compatibility. It runs its existing TPU workflow on libtt, reusing its TorchTitan trainer, SGLang-TorchTPU server and PyTorch model definitions, with no Tenstorrent-specific trainer or model port. A two-file patch to Miles 7767e158 enables its disk-based weight transfer on the TPU platform and replaces two CUDA-specific calls; TorchTitan carries Miles' own TPU patch.

On a TT-QuietBox 2, Qwen3-0.6B trains on two Blackhole chips with FSDP2 and generates on two with tensor parallelism. Each of 90 group-relative policy optimization (GRPO) steps samples eight answers to each of 16 GSM8K training questions @gsm8k, scores their boxed answers, updates the model, and transfers the weights. The run uses AdamW at $2 times 10^(-6)$, BF16 model computation and FP32 parameters and optimizer state; generation uses BF16 weights.

With identical prompts, greedy decoding and a 1,024-token response limit, accuracy on all 1,319 GSM8K test questions rises from 58.00 to 72.25 per cent, a gain of 14.25 points (@rl-learning-curve). Both endpoints were evaluated in fresh servers and the answers independently rescored, which also verifies that the saved policy reloads correctly. The gain is not answer format: the starting policy already boxes 99 per cent of its answers, and wrong values fall from 544 to 262. With the learning rate set to zero, 30 steps of the same loop change no weight (every transfer is empty, and the served weights stay byte-identical) and leave test accuracy at 58.07 per cent, so rollouts, transfer and reloading do not move the policy on their own. Two further runs with other seeds, on libtt 0.2.2, reach 68.6 and 67.1 per cent after 30 steps, against 70.7 per cent for the original run, which used libtt 0.2.1. For the first rollout batch of each, an FP32 CPU computation of the answers' log-probabilities agrees with the trainer's batch mean to within 0.0003.

#figure(
  placement: none,
  rl-learning-image,
  caption: [Qwen3-0.6B on GSM8K on four Blackhole chips. Left: training reward per step of the original 90-step run (grey) and its trailing 15-step mean (black), with the trailing means of two 30-step runs with other seeds (colour) and of the zero-learning-rate control (dashed). Right: test accuracy of the original run (circles: in-run evaluations; squares: fresh-server checks), of the two other seeds from their common starting point (colour), and of the control (dashed). All test points use greedy decoding and a 1,024-token response limit.],
) <rl-learning-curve>

Over the original run's final 15 steps, generation averages 46 seconds, log-probabilities 24 seconds, training 51 seconds, and weight transfer 1.5 seconds. These runs demonstrate the complete RL loop; distributed training checkpoint and resume remain follow-up work.

]
