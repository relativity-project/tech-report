// Training with TorchTitan.
// Original throughput and short checks: libtt main at 5e041d48 and the Llama 3
// work in pull request #554. Completed fine-tuning and GSM8K results checked
// November 15, 2026 (America/Los_Angeles): ~/tt-work/train35/logs/sft300b.log
// and gsm8k_{base,sft300b}.completions.json. Training wheel: p2p; evaluation: 4cbcaec3.

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

The checks against a CPU are separate from these full-size throughput measurements. Tested small models and selected layers are trained for a few steps from the same saved starting point on both the CPU and the chips. They include TorchTitan's debug models and the first two layers of Llama 3 8B with its full vocabulary and 8,192-token sequences. In these checks, the first losses agree to three digits or more, and subsequent losses differ by at most about one per cent. Different kernels, rounding, and summation order can produce small differences that accumulate across updates.

The DeepSeek-V3, Qwen3 MoE, and Qwen3.5 MoE debug models run unchanged in PyTorch eager mode with 32-bit floats. 

== Supervised Fine-tuning <fine-tuning>

The fine-tuning experiment follows a published checkpoint through training, export, and evaluation on four chips. Qwen3.5-9B-Base was trained on MetaMathQA @metamath, a collection of 395,000 math problems with worked solutions, using a recipe adapted from TorchTitan's Llama 3 fine-tuning example. The run completed 300 training steps, followed by a before-and-after evaluation on GSM8K.

Problems and solutions use Qwen's chat template, and only answer tokens contribute to the loss. Whole samples are packed into sequences of 2,048 tokens, with four sequences per chip: 32,768 tokens per step and about 9.8 million over the run. AdamW's learning rate rises to $2 times 10^(-5)$ over 25 steps and then follows a cosine decay. An adapter imports and exports the weights in the standard Hugging Face checkpoint format.

After the first step, a step averaged 29.6 seconds, or about 277 tokens per second per chip. The 300 steps took about 2.5 hours, and saving the final checkpoint took about 50 seconds. @sft-loss shows the complete run: the logged loss starts at 0.340 and averages 0.157 over the last ten steps.

#fig-sft-loss <sft-loss>

The evaluation uses all 1,319 questions of GSM8K's test set @gsm8k. The Base and fine-tuned models both run through SGLang-TorchTPU on four chips, with BF16 weights, the same chat format, and greedy decoding capped at 512 generated tokens per question. Each prompt contains one question and no worked examples. The scorer extracts the number after “The answer is”, or the last number if that phrase is absent.

The Base model answers 848 questions correctly (64.29 per cent), and the fine-tuned model answers 1,152 correctly (87.34 per cent), a gain of 23.05 percentage points under these evaluation settings. These results were checked against the saved completions.

#import "fig-rl-learning.typ": rl-learning-image

// Miles 7767e158 + two-file disk-delta portability patch. Original run: long3.
// Endpoint checks: rl-report-20261117; serving libtt 0.2.1 build 39235edd.
#block(breakable: false)[
== Reinforcement learning with Miles <reinforcement-learning>

Miles @miles is a reinforcement-learning framework that runs training and serving in one loop, which makes it a demanding test of framework compatibility. It runs its existing TPU workflow on libtt unchanged in structure, reusing its TorchTitan trainer, SGLang-TorchTPU server and PyTorch model definitions, with no Tenstorrent-specific trainer or model port.

On a TT-QuietBox 2, Qwen3-0.6B trains on two Blackhole chips with FSDP2 and generates on two with tensor parallelism. Each of 90 group-relative policy optimization (GRPO) steps samples eight answers to each of 16 GSM8K training questions @gsm8k, scores their boxed answers, updates the model, and transfers the weights. The run uses AdamW at $2 times 10^(-6)$, BF16 model computation and FP32 parameters and optimizer state; generation uses BF16 weights.

With identical prompts, greedy decoding and a 1,024-token response limit, accuracy on all 1,319 GSM8K test questions rises from 58.00 to 72.25 per cent, a gain of 14.25 points (@rl-learning-curve). Both endpoints were evaluated in fresh servers and the answers independently rescored, which also verifies that the saved policy reloads correctly.

#figure(
  placement: none,
  rl-learning-image,
  caption: [Qwen3-0.6B on GSM8K, one run on four Blackhole chips. Left: training reward and its trailing 15-rollout mean. Right: original test evaluations (circles) and matched fresh-server checks (squares). All test points use greedy decoding and a 1,024-token response limit.],
) <rl-learning-curve>

Over the final 15 steps, generation averages 46 seconds, log-probabilities 24 seconds, training 51 seconds, and weight transfer 1.5 seconds. This single run demonstrates the complete RL loop; distributed training checkpoint and resume remain follow-up work.

]
