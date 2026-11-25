#import "tables.typ": *

== Performance <measure>

The published Blackhole serving results were measured on a TT-QuietBox 2: Four Blackhole chips on two cards, an AMD Ryzen 7 9700X, and 256 gigabytes of host memory @blackhole. They come from the project's pages @relativity, where the scripts that produce them are also to be found. All of them were taken with version 0.2.2 of libtt. Single-request figures are medians of five runs after two to warm up. With several chips the model is divided by tensor parallelism over all of them.

=== One request

@single-table gives the speed of decoding and the time to the first token. Under SGLang-JAX the matrices of weights are in BFP8, the experts of mixture-of-experts models in BFP4, and everything else in BF16. Under SGLang-TorchTPU the weights are 8-bit floats.

#tab-single <single-table>

Speed does not necessarily grow in proportion to chips: Qwen3-8B on two chips is 1.72 times as fast as on one, and on four chips 2.65 times. The reason is that only the reading of weights is divided by four. The rest of a step is not: the small programs between the projections, the collectives, the choice of a token. Of the two dense models that fit on one chip, the larger gains more; Qwen3-14B is 3.09 times as fast on four chips.

=== Many requests

@batched-table gives the total speed when several are decoded together. Since each decode step reads the weights only once, the cost of a step grows much more slowly than the number of rows in it.

#tab-batched <batched-table>

#block(breakable: false)[
=== Comparison with a TPU

In @tpu-comparison we compare the exact same SGLang-TorchTPU code in two settings: Running with the libtt runtime on a Tenstorrent Blackhole P150 card and running with Google's libtpu runtime on a TPU v6e. 
Per unit of memory bandwidth, one Blackhole matches or exceeds one TPU at single-request decode. The TPU is 2.8 to 3.2× faster, close to its 3.2× bandwidth advantage (#link("https://docs.cloud.google.com/tpu/docs/v6e")[1,638 GB/s] versus #link("https://docs.tenstorrent.com/aibs/blackhole/")[512 GB/s]).

#figure(
  caption: [Comparing LLM inference for FP8 models: 128 greedy output tokens; medians of five single requests or three batches after warm-up. Decode uses a 5-token prompt; first-token latency uses 215 tokens.],
  table(
    columns: 7,
    align: (left, right, right, right, right, right, right),
    table.hline(stroke: heavy),
    table.header(
      table.cell(rowspan: 2, align: bottom + left)[Model],
      table.cell(colspan: 2, align: center)[Decode, tokens/s],
      table.cell(colspan: 2, align: center)[First token, ms],
      table.cell(colspan: 2, align: center)[16 requests, tokens/s],
      table.hline(start: 1, end: 3, stroke: 0.3pt),
      table.hline(start: 3, end: 5, stroke: 0.3pt),
      table.hline(start: 5, end: 7, stroke: 0.3pt),
      [BH P150], [TPU v6e], [BH P150], [TPU v6e], [BH P150], [TPU v6e],
    ),
    table.hline(stroke: light),
    [Qwen3-8B-FP8], [44.2], [130.6], [253.5], [108.2], [407.6], [1,270.1],
    [Qwen3.5-9B], [37.1], [119.7], [471.0], [123.3], [206.3], [1,177.6],
    [Qwen3-14B-FP8], [25.9], [73.5], [395.4], [198.0], [251.6], [806.1],
    table.hline(stroke: heavy),
  ),
) <tpu-comparison>

]

#block[
=== Upstream Tenstorrent comparison

We also compare with the upstream Tenstorrent software stack: a vLLM plugin that requires separate TTNN model definitions together with custom scheduling, execution, KV-cache handling and runtime compatibility patches. That stack is compared against two libtt-based stacks on the same chip: sglang served via TorchTPU, and sglang-jax.

These comparisons are not apples to apples: the three stacks do not use the same numerics. The Tenstorrent vLLM uses BFP8 for most weights, which is a blockwise FP8 format, and its FP4 equivalent BFP4 for others. Sglang-jax used BFP8 throughout so it is most comparable to the upstream stack, and sglang-torchtpu uses FP8 and has additional overhead from the FP8 scales. The results are in @upstream-comparison.

#figure(
  caption: [Same prompts, 128-token outputs, and timing method as @tpu-comparison; one chip, no prefix caching or speculation. Decode and first-token columns are the short-prompt / long-prompt measurements. "16 requests" is total throughput at 16 concurrent requests (median of three measured batches).],
  table(
    columns: 10,
    align: (left, right, right, right, right, right, right, right, right, right),
    table.hline(stroke: heavy),
    table.header(
      table.cell(rowspan: 2, align: bottom + left)[Model],
      table.cell(colspan: 3, align: center)[Decode, tokens/s],
      table.cell(colspan: 3, align: center)[First token, ms],
      table.cell(colspan: 3, align: center)[16 requests, tokens/s],
      table.hline(start: 1, end: 4, stroke: 0.3pt),
      table.hline(start: 4, end: 7, stroke: 0.3pt),
      table.hline(start: 7, end: 10, stroke: 0.3pt),
      [sglang-jax], [sglang-torchtpu], [Upstream],
      [sglang-jax], [sglang-torchtpu], [Upstream],
      [sglang-jax], [sglang-torchtpu], [Upstream],
    ),
    table.hline(stroke: light),
    [Qwen3-8B], [46.1], [44.2], [38.7], [122.6], [253.5], [130.7], [514.1], [407.6], [522.8],
    [Qwen3.5-9B], [43.4], [37.1], [22.4], [161.0], [471.0], [173.5], [410.3], [206.3], [—],
    [Qwen3-14B], [26.5], [25.9], [20.9], [183.2], [395.4], [209.8], [318.5], [251.6], [287.0],
    table.hline(stroke: heavy),
  ),
) <upstream-comparison>

As we can see, the sglang-jax implementation on libtt is as performant or better than the upstream software, which is due to extra fusions we perform on the compiler level. The sglang-torchtpu implementation is better than upstream at decode but less good at prefill and larger batches, which is due the additional overhead from dynamic quantization.
]


=== Sampling and speculative decoding

#tab-sampling <sampling-table>

@sampling-table shows the cost of drawing tokens at random instead of taking the most likely one. The JAX measurements generate device random bits with a single operation instead of JAX's default Threefry generator @random123, a choice that materially affects sampling cost @relativity. The cost is proportionally larger on four chips because the sampler does not speed up with more chips while the model does. The PyTorch server pays less for the `top_p` and `top_k` filters than the JAX server.

#tab-spec <spec-table>

Libtt also supports speculative decoding, and @spec-table shows the gain. Under SGLang-JAX on four chips, the model's own draft layer speeds up a short factual prompt by 2.6× and a long summary by 24%. The gain depends on how often the draft is right.
