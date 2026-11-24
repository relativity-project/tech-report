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

The fine-tuning experiment teaches a published base model to call functions, and follows it through training, export, and evaluation on four chips. Qwen3.5-9B-Base was trained on ToolACE @toolace: 11,300 examples in which a system prompt lists the available functions in JSON and the reply to a user's request is the calls to make, written as a Python list such as `[get_weather(city="Paris")]`, or a statement that no function applies. The recipe is adapted from TorchTitan's Llama 3 fine-tuning example. Samples use Qwen's chat template, and only the reply's tokens contribute to the loss. Whole samples are packed into sequences of 2,048 tokens, four per chip: 32,768 tokens per step, about 1.1 passes over the data in 300 steps. AdamW's learning rate rises to $2 times 10^(-5)$ over 25 steps and then follows a cosine decay. Parameters, gradients and AdamW's state are BF16, as in @training-runs. A fixed 2 per cent of ToolACE is held out, and an adapter imports and exports the weights in the standard Hugging Face checkpoint format.

After the first step, a step averaged 29.5 seconds, or about 278 tokens per second per chip, and the 300 steps took about 2.5 hours. @sft-loss shows the run: the held-out loss falls from 0.448 after the first step to 0.302 by step 25 and 0.252 at the end.

#figure(
  placement: auto,
  fig-sft-loss,
  caption: [Fine-tuning Qwen3.5-9B-Base on ToolACE, 300 steps on four chips: each step's training loss (grey), its ten-step mean (black), and the loss on the held-out 2 per cent (circles). Losses are means over the replies' tokens.],
) <sft-loss>

The evaluation uses the Berkeley Function Calling Leaderboard @bfcl: its five single-turn categories, 1,240 tests, with BFCL's own prompt (which states the required output format), parser and checker of the calls' syntax trees. Both models run through SGLang-TorchTPU on four chips with greedy decoding, capped at 512 tokens.

#figure(
  placement: auto,
  caption: [Function calling on BFCL's five single-turn categories (per cent of tests passed): Qwen3.5-9B-Base before and after fine-tuning on ToolACE. Lenient accepts any readable list of calls in the reply, a control for answer format.],
  table(
    columns: 4,
    align: (left, right, right, right),
    table.hline(stroke: heavy),
    table.header([Category (tests)], [Base], [Base, lenient], [Fine-tuned]),
    table.hline(stroke: light),
    [Simple (400)], [58.3], [63.3], [84.8],
    [Multiple (200)], [63.0], [67.5], [86.0],
    [Parallel (200)], [51.5], [64.0], [85.0],
    [Parallel multiple (200)], [45.0], [51.5], [77.0],
    [Irrelevance (240)], [94.6], [93.8], [95.0],
    table.hline(stroke: light),
    [Mean], [62.5], [68.0], [85.6],
    table.hline(stroke: heavy),
  ),
) <bfcl-results>

Fine-tuning raises the mean from 62.5 to 85.6 per cent (@bfcl-results). The Base model often writes a reply the parser cannot read, for example by copying the format's template after reasoning correctly; reading its replies leniently raises it only to 68.0 per cent, so 17.6 of the 23.1 points are not answer format. The fine-tuned model's 177 failures are mostly replies that point out a missing required parameter, as the prompt asks, where BFCL expects a call, and calls with a wrong value.

The training itself was also checked on MetaMathQA @metamath, TorchTitan's own fine-tuning example, with the same recipe. On libtt 0.2.2 it reproduces the earlier run's first 30 losses exactly. With BF16 parameters and AdamW state, most updates are smaller than half a BF16 unit in the last place and are lost: at the peak learning rate about 90 per cent of the feed-forward and attention weights keep their value in a step. On Qwen3.5-2B-Base this acts as a smaller learning rate: FP32 storage at the same rate ends slightly worse on GSM8K (72.9 against 75.0 per cent), and at a quarter of the rate slightly better (75.4 per cent). A scaled-down run agrees with a CPU reference, within half the difference between two data orders, and a CPU reference picks the chips' next token at 98.6 to 99.7 per cent of positions, differing only at near-ties. TODO-S5. Full results are in the repository's results note.
#import "fig-rl-learning.typ": rl-learning-image

// Miles 7767e158 + two-file disk-delta portability patch. Original run: long3.
// Endpoint checks: rl-report-20261117; serving libtt 0.2.1 build 39235edd.
#block(breakable: false)[
== Reinforcement learning with Miles <reinforcement-learning>

Miles @miles is a reinforcement-learning framework that runs training and serving in one loop, which makes it a demanding test of framework compatibility. It runs its existing TPU workflow on libtt, reusing its TorchTitan trainer, SGLang-TorchTPU server and PyTorch model definitions, with no Tenstorrent-specific trainer or model port. A two-file patch to Miles 7767e158 enables its disk-based weight transfer on the TPU platform and replaces two CUDA-specific calls; TorchTitan carries Miles' own TPU patch.

On a TT-QuietBox 2, Qwen3-0.6B trains on two Blackhole chips with FSDP2 and generates on two with tensor parallelism. Each of 90 group-relative policy optimization (GRPO) steps samples eight answers to each of 16 GSM8K training questions @gsm8k, scores their boxed answers, updates the model, and transfers the weights. The run uses AdamW at $2 times 10^(-6)$, BF16 model computation and FP32 parameters and optimizer state; generation uses BF16 weights.

With identical prompts, greedy decoding and a 1,024-token response limit, accuracy on all 1,319 GSM8K test questions rises from 58.00 to 72.25 per cent, a gain of 14.25 points (@rl-learning-curve). Both endpoints were evaluated in fresh servers and the answers independently rescored, which also verifies that the saved policy reloads correctly. The gain is not answer format: the starting policy already boxes 99 per cent of its answers, and wrong values fall from 544 to 262. With the learning rate set to zero, 30 steps of the same loop change no weight (every transfer is empty, and the served weights stay byte-identical) and leave test accuracy at 58.07 per cent, so rollouts, transfer and reloading do not move the policy on their own. TODO-R1: two further runs with other seeds.

#figure(
  placement: none,
  rl-learning-image,
  caption: [Qwen3-0.6B on GSM8K, one run on four Blackhole chips. Left: training reward and its trailing 15-rollout mean. Right: original test evaluations (circles) and matched fresh-server checks (squares). All test points use greedy decoding and a 1,024-token response limit.],
) <rl-learning-curve>

Over the final 15 steps, generation averages 46 seconds, log-probabilities 24 seconds, training 51 seconds, and weight transfer 1.5 seconds. This single run demonstrates the complete RL loop; distributed training checkpoint and resume remain follow-up work.

]
