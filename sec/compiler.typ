#import "../style.typ": cpp
#import "figures.typ": fig-pipeline

= Compiling <compiler>

== The Compilation Stages

A framework passes the plugin a StableHLO program along with compile options. The plugin processes it in four stages (see @pipeline).

#fig-pipeline <pipeline>

First, if the program is to run on multiple chips, the Shardy partitioner decides how each tensor is divided among them, and the program is rewritten to describe the work of a single chip (@chips).

Second, StableHLO is lowered to TTIR, a tensor-level dialect that has no notion of tiles. Types the hardware does not support are legalized at this stage: complex numbers become pairs of floats, 64-bit floats become 32-bit, and signed 8- and 16-bit integers are held in 32 bits. FP16 and the supported FP8 formats keep their own tensor types, and operations without a native FP8 implementation compute on widened values (@numerical-types).

Third, TTIR is lowered to the TTNN dialect, in which each operation is a single call into the runtime’s operation library and, as a rule, a single program on the chip. Each tensor is assigned a layout (tiled or row-major), a placement (host or device), and a type, and each operation is assigned its arithmetic settings.

Finally, the result is serialized as a table of calls, which an interpreter in the runtime walks. No machine code is generated for the host.

The compiler defines three optimization levels, of which two are in use. At the lowest, every operation uses the highest-precision arithmetic the matrix unit offers. At the next, an optimizer queries the operation library for the layouts and types each call accepts and selects among them. SGLang-JAX compiles at the second level; torch_tpu at the lowest.

The upstream compiler already includes all four stages. libtt’s additions are described in the sections that follow.

== Matching framework semantics

A framework's test suite is largely a list of small questions about what an operation means. Does a NaN compare equal to itself? What does a scatter do with an index past the end? Which of two equal keys does a sort place first? The frameworks answer each of these, and much of our work went into making libtt give the same answers.

@tab-semantics lists the behaviors libtt matches, grouped by area, and each entry stands for one or more changes to the compiler, the operation library, or a kernel.
 
#[
#show figure: set block(breakable: true)
#figure(
  table(
    columns: (auto, 1fr),
    align: left + top,
    stroke: none,
    inset: (x: 4pt, y: 4pt),
    table.hline(),
    table.header([Area], [Framework behavior that libtt matches]),
    table.hline(stroke: 0.5pt),
 
    // Values
    [Comparison],
    [A NaN is not equal to itself, so floating-point self-comparisons are no longer folded. Total-order comparisons, which JAX's sort and `searchsorted` use, place $-0$ below $+0$ and NaNs at both ends.],
 
    [NaN and signed zero],
    [Cumulative maxima and minima propagate NaN, and `sign` of a NaN is NaN. Negation and `log1p` keep the sign of zero. Moving or repacking a 16-bit float preserves its bits, where the default path flushed subnormals, folded $-0$ into $+0$, and rewrote NaNs.],
 
    [Integer arithmetic],
    [Remainder truncates. Powers wrap modulo $2^32$, as multiplication does, and unsigned negation is modular. High-word multiplication, arithmetic right shift, leading-zero and population counts, and integer dot products are supported. Integer select, clamp, and permute stay in integer arithmetic.],

    [Floating-point types],
    [IEEE FP16 is a type distinct from BF16 from the PJRT buffer down to the kernel. `float8_e4m3fn` and `float8_e5m2` remain FP8 in buffers, compiler types, and the runtime, and layout changes, gathers, and copies preserve their bytes. Operations without a native FP8 implementation compute on widened values, and the quantization kernels return FP8 tensors directly.],

    [Narrow integer types],
    [Signed 8- and 16-bit integers are held in 32-bit integers, and integer operations on them stay in integer arithmetic.],

    [Conversion],
    [Float-to-integer conversion truncates toward zero. Integer narrowing keeps the low bits instead of saturating. A chain of explicit conversions keeps each intermediate rounding. Unsigned values keep their value: a `uint8` 255 is not $-1$. Bitcasts copy raw bits, and `reduce_precision` rounds as XLA does.],
 
    table.hline(stroke: 0.5pt),
 
    // Operations
    [Sorting],
    [Equal keys keep their order in a stable sort. NaNs sort above $+infinity$ as one tie class. Integer keys and values sort as integers or, when narrower than 32 bits, exactly through FP32. Non-strict comparators are accepted.],
 
    [Argmax, argmin, top-k],
    [Ties go to the smaller index. An initial value that is not the identity takes part in the reduction. Top-k returns its results in order.],
 
    [Reductions],
    [Integer sums, maxima, and minima are computed exactly in integer arithmetic instead of passing through FP32, which rounds above $2^(24)$. `and`, `or`, and `xor` reduce bitwise on integers and logically on booleans. Variadic reductions, which return several results at once such as a minimum and a maximum together, are supported, as are cumulative sums, maxima, and minima and general scans.],
 
    [Convolution and pooling],
    [Convolution keeps StableHLO's cross-correlation convention where the operation library reverses the weights. Negative padding crops the input, and reversed windows and input dilation are handled. A single-channel convolution is no longer taken for a depthwise one. Padding folds into a pooling only when it supplies the reduction's initial value.],
 
    [Indexing],
    [Gathers, scatters, and slices at computed positions take multi-component indices and windows. An out-of-range gather index is clamped, and an out-of-range scatter update is dropped. A scatter may add, multiply, or take the maximum or minimum, and applies duplicate updates in order. All of these lower to two new operations, `rows_gather` and `rows_scatter`, which treat the operand as a table of rows and apply the out-of-bounds rule in the kernel, replacing the runtime's embedding lookup, gather, and scatter. Only the indices and updates are rearranged, so an embedding lookup does not copy its table.],
 
    [Control flow],
    [A while loop repeatedly invokes an outlined program and updates its state. Conditionals and multi-way branches may nest, and may run inside a trace (Section 6.1).],
 
    [Complex numbers],
    [Held as pairs of floats, with multiplication, division, absolute value, equality, selection, transposition, reductions, and contractions such as dot products and convolutions.],
 
    [Empty tensors],
    [Concatenations and convolutions with empty operands fold to constants, where the row-major concatenation divided by a zero dimension. Zero-sized tensors pass through layout conversion and buffer casts.],
 
    [Random numbers],
    [StableHLO's random-bit generator runs on the device with Threefry and Philox and keeps its 64-bit state, so seeded draws repeat in eager and compiled code.],
 
    [Other operations],
    [Triangular solves expand into batched matrix products, as XLA expands them for TPUs. Batch normalization returns its training statistics. Deprecated operations that older frontends still emit are legalized.],
 
    table.hline(stroke: 0.5pt),
 
    // Interface
    [PJRT interface],
    [Buffers may be donated, borrowed from host memory with arbitrary strides, or hold 64-bit integers. Executables serialize for the frameworks' persistent caches. A custom call may return several results. Failures report their cause through the interface instead of escaping as exceptions. A process may create a new client before it destroys the old one.],
 
    table.hline(),
  ),
  caption: [Framework behavior that required changes to the upstream stack],
) <tab-semantics>
]

== Numerics <numerical-types>

=== FP16 and FP8

Supporting a numeric type requires agreement at every layer, from the PJRT buffer down to the kernel. libtt adds IEEE FP16 as a type distinct from BF16 throughout this path. Operations that only move values preserve their bit patterns, including signed zeros and subnormals, by copying the underlying bits rather than passing them through floating-point arithmetic.

The FP8 formats float8_e4m3fn and float8_e5m2 likewise remain FP8 in PJRT buffers, the compiler’s tensor types, and the runtime. These are per-element floating-point formats, distinct from the BFP8 block format described in @machine. Layout changes, gathers, copies, and other data movement preserve their bytes, and the quantization kernels return FP8 tensors directly.

Libtt recognizes PyTorch’s scaled matrix products and uses the matrix unit’s native FP8 support for supported operand pairs, with FP32 accumulation @libtt. Activation quantization runs in one device program or is fused into the preceding normalization, and scales are applied within the matrix-product kernel. These changes reduce the overhead around FP8 linear layers while preserving their numerical behavior; serving results are given in @measure.

=== FP32 and integers

Highest-precision FP32 matrix products run on a new vector-unit kernel, which retains FP32 operands and accumulation instead of the matrix unit's reduced operand precision, at a substantial cost in speed. The compiler preserves this request through optimization. PyTorch selects this kernel when highest precision is requested; its default uses the matrix unit's reduced operand precision. Other changes keep 32-bit integer collective sums and cumulative sums in integer arithmetic, avoiding the loss of exactness from conversion to floating point. These sums wrap on overflow as the framework requires.

== Fusion

A model written in a framework is a graph of small operations. An RMS norm, for example, is a square, a mean, an addition, a reciprocal square root, and two multiplications. Running these separately means starting several programs and repeatedly reading and writing intermediate results. The compiler recognizes such groups and replaces them with one kernel that does the work in a single program. This is called _fusion_. The same computation can look different in JAX and PyTorch, so the compiler recognizes both forms. Libtt adapts these patterns for activation functions, normalization, rotary embeddings, and attention preparation, preserving the original casts and rounding. The fusions remain available at the low optimization level used by `torch_tpu`. @fusions gives some examples.

#figure(
  caption: [
    Selected fusions used by libtt @libtt.
    † marks fusions added by libtt; unmarked entries adapt upstream fusions to a broad range of real-world PyTorch and JAX patterns and data types.
  ],
  table(
    columns: (1fr, 1.8fr),
    align: left,
    inset: (x: 5pt, y: 4pt),
    table.hline(stroke: 0.8pt),
    table.header([Operations], [Combined work]),
    table.hline(stroke: 0.5pt),

    [Matmul + bias or activation],
    [Apply bias, SiLU, or GELU before storing the result.],

    [Convolution + activation],
    [Apply ReLU or SiLU inside the convolution.],

    [Elementwise arithmetic + activation],
    [Combine activations with addition or multiplication.],

    [Matrix products sharing an input],
    [Use one wider product, with weights prepared once.],

    [Softmax],
    [Subtract the maximum, exponentiate, sum, and divide.],

    table.hline(stroke: 0.3pt),

    [Residual addition + RMS norm#super[†]],
    [Add the residual and normalize in one program.],

    [SwiGLU#super[†]],
    [Fuse paired projections, SiLU, and the gating multiply.],

    [Residual RMS norm + FP8 conversion#super[†]],
    [Produce FP8 values and row scales during normalization.],

    [Attention input preparation#super[†]],
    [Split heads, normalize queries and keys, and apply rotary embeddings.],

    table.hline(stroke: 0.8pt),
  ),
) <fusions>

A fused kernel must be both faster and not change the numerics. Fewer programs do not by themselves guarantee a speedup, because a kernel can still waste time reading padding or moving data between cores. Fusion must also preserve the rounding of the operations it replaces: moving arithmetic between the matrix and vector units, or removing an intermediate cast, can change the result. Several of libtt's fused kernels reproduce every rounding step, so their results match the unfused operations bit for bit.

Some improvements require no new kernels and come entirely from compiler rewrites. A matrix product that yields one number per batch becomes a multiplication followed by a sum. A program that keeps only the largest softmax probabilities and renormalizes them instead selects the largest inputs first and applies softmax to those alone. Both rewrites reduce the work that existing kernels perform.

== Extending the framework with custom calls

Some operations fall outside what StableHLO's primitives express efficiently. For these, the framework provides an extension point at runtime: a JAX program invokes a _custom call_ by name, and the compiler lowers it directly to a libtt kernel.

```
cache = jax.ffi.ffi_call(
    "tt.paged_update_cache", result_shape,
    input_output_aliases={0: 0},
)(cache, new_kv, positions, pages, share_cache=True)
```
Custom calls cover attention over a paged cache, recurrent layers, mixture-of-experts layers, and row quantization. For training, they add forward and backward attention and fused loss calls that run tt-train’s kernels (@training). Keyword arguments are passed via the call’s typed attribute dictionary. The alias allows the result to overwrite the first argument, which is how a cache is updated in place.

== Tiles and precision

The tiled storage described in @machine affects far more than matrix multiplication. A reshape, slice, or gather can force data to move between tiles, and a tensor with only a few rows is mostly padding. These costs recur throughout a model, so layout selection is a major concern for the compiler.

Libtt keeps data in layouts that neighboring operations can consume directly, which avoids unnecessary layout conversions, and it moves elementwise work to shapes that occupy fewer tiles. For wide rows, it regroups data into fuller tiles for reductions or uses kernels that skip the unused parts of tiles, reducing both memory traffic and work spent on padding @libtt. The rotary-embedding example in @training shows how a dimension of two occupies 32 columns in tiled storage, and how a rewrite avoids that padding.

These layout choices must also respect the numerical requirements in @numerical-types. Libtt uses vector-unit arithmetic and 32-bit accumulation where needed, and its optimizations preserve these choices instead of falling back on the runtime's defaults.