#import "tables.typ": tab-replaced

= Serving <serving>

Libtt currently supports SGLang-TorchTPU and SGLang-JAX for large language model serving. The server handles request scheduling and batching, while libtt compiles each step and supplies the device kernels. This section uses SGLang-TorchTPU as its main example and closes with performance measurements (@measure).

The two phases of LLM serving, prefill and decode, stress different resources. Prefill processes the whole prompt at once, so it has many tokens per program and is compute-bound. Decode generates one token per request per step, so with small batches it has few tokens per program and is dominated by weight reads and program launches. The balance between the phases depends on prompt length, output length, and batch size. The server compiles programs for a fixed set of shapes, and libtt traces and replays repeated execution to cut launch overhead (@running).

== Adapting the server

SGLang-TorchTPU relies on TPU-specific kernels written in Pallas, which libtt cannot run. Libtt rewrites these kernels for Tenstorrent hardware, building on upstream Tenstorrent kernels, its own kernels, and ordinary framework operations. The compatibility layer in @pytorch installs the rewrites behind the interfaces the server already calls, so the server's source is unchanged on the supported paths. @replaced summarizes the main rewrites.

#tab-replaced <replaced>

== Projections and experts

Decode-time projections are limited by how fast weights stream from DRAM, so libtt targets three places where that traffic and the surrounding work can be cut.

_Projections._ A projection multiplies activations by a weight matrix held in DRAM, and at small decoding batches its speed depends on how quickly those weights reach the cores. Libtt adapts the matrix-product kernels to keep each input row resident on the cores while the weights stream past, and to spread reads across memory banks. The bank spreading matters most for transposed PyTorch weights, whose natural read order otherwise concentrates traffic on a few banks.

_Fusion._ A feed-forward block computes two projections of the same input, applies SiLU to one, and multiplies the results. Libtt fuses the two projections and performs the remaining operations before the output leaves the core (@fusions), so the intermediates never return to DRAM.

_Mixture of experts._ Each token selects a few feed-forward blocks, for example 8 of 128 experts in Qwen3-30B-A3B, so reading only the selected experts' weights cuts memory traffic. Libtt uses Tenstorrent's fused expert kernel, including 4-bit block-format weights, and integrates it across chips: each chip computes the selected experts it holds, and their outputs are summed (@chips). For SGLang-TorchTPU, we added support for BF16 and FP8 experts. FP8 weights are read directly and scaled inside the kernel, with one scale per expert and output column, while external inputs and outputs stay BF16. Expert selection is also kept cheap: the router finds the largest scores with a parallel scan on the data-movement processors, avoiding a full sort.

== Attention and recurrent state

_Paged attention._ Keys and values of earlier tokens stay in a paged cache on the device. Libtt exposes the upstream paged-attention kernels for decode and chunked prefill through custom calls. It adds in-place writes of new keys and values, so the whole cache is never converted between layouts, and it divides attention work across cores.

_Cache layout._ Libtt places a head's key and value side by side in one row. Stored as two rows, that dimension would pad to 32, wasting memory and adding data movement. The server keeps its own request and page management on top of this device layout.

_Recurrent layers._ Hybrid models such as Qwen3.5 and Qwen3.8 combine attention with gated DeltaNet layers @gdn, which carry a fixed-size state matrix per head from token to token. Libtt supplies kernels for the decode step and the short convolution that precedes it, along with a pool of states for concurrent requests. The states stay on the device between steps.

== Choosing and verifying tokens

_Greedy_ decoding takes the highest-scoring token. _Sampling_ scales the scores by a temperature, optionally filters out unlikely tokens, and draws from the resulting distribution. Libtt runs both on the device, including random-number generation. They operate on vocabulary-wide rows and can account for a substantial part of a short decode step (@sampling-table).

_Speculative decoding_ has a cheap draft propose several tokens, which the main model then verifies together @specdec. Libtt supports both a model's own draft layer and a separate draft model. Verifying candidates together can reuse work, but the speedup depends on verification cost and on how many candidates are accepted (@spec-table).

Recurrent layers complicate verification, because each candidate depends on the state produced by the one before it. Libtt's kernels process several steps in one call with the state held on the device, and save the state after each candidate. Verification can then keep the accepted prefix's state and discard the rejected suffix. Attention applies the corresponding cache updates for the candidate tokens.

#include "measure.typ"
