#import "../style.typ": cpp
#import "figures.typ": fig-core

= The Machine <machine>

A Blackhole chip has 120 small computers, which Tenstorrent calls Tensix cores, laid out in a grid @blackhole. Each has 1.5 megabytes of memory, five 32-bit RISC-V processors, a unit that multiplies matrices, a unit that does arithmetic on vectors, and connections to two networks that join it to its neighbors @metaliumguide. Beside the grid are 32 gigabytes of GDDR6 DRAM, which any core can read and write over the networks (@core). There is no cache shared between cores. The run-time system gives 110 of the cores to programs.

#fig-core <core>

A card holds one or two chips and plugs into a PCI Express slot. Most measurements in this paper were made on a TT-QuietBox 2, a workstation with four chips on two cards, in which the chips are joined by 800 Gbps Ethernet. Tenstorrent also offers datacenter servers with 32 Blackhole chips, which can be linked over the same Ethernet interconnect in various topologies. @accelerator-comparison compares Blackhole with GPUs and TPUs.

#figure(
  placement: auto,
  caption: [
    Architectural features relevant to libtt, based on vendor and compiler documentation @cudaguide @amdgpu @tpukernels @metaliumguide.
    GPU details vary by architecture; the Tenstorrent column describes Blackhole.
  ],
  table(
    columns: (0.85fr, 1.4fr, 1.4fr, 1.5fr),
    align: left,
    inset: (x: 4pt, y: 4pt),
    table.hline(stroke: 0.8pt),
    table.header([Aspect], [GPU (NVIDIA, AMD)], [TPU (Google)], [Blackhole (Tenstorrent)]),
    table.hline(stroke: 0.5pt),

    [Compute units],
    [Thread-based arithmetic and specialized matrix units],
    [Matrix, vector, and scalar units],
    [Matrix and vector units in each Tensix core],

    [Execution],
    [Hardware schedules groups of threads: warps or wavefronts],
    [Compiler schedules operations and overlaps transfers],
    [RISC-V controllers run cooperating compute and data-movement programs],

    [On-chip memory],
    [Registers, caches, and software-managed shared memory],
    [Registers and software-managed vector/scalar memory],
    [Registers and software-managed L1 SRAM in each core],

    [Data movement],
    [Loads and stores through caches; explicit transfers to shared memory],
    [Scheduled transfers between HBM and on-chip memory],
    [Explicit transfers between DRAM and cores over the on-chip network],

    [Tiling],
    [Kernel and matrix instructions determine tile shapes, tensors often row-major in main memory],
    [Typically 8 × 128 tiles, with extra packing for narrower types @tputiling],
    [Usually 32 × 32 tiles, with subtile support; tensors usually kept tiled in main memory for performance],

    [Main memory],
    [HBM or GDDR],
    [HBM],
    [GDDR6],

    table.hline(stroke: 0.3pt),

    [Software stack],
    [CUDA or ROCm, with framework backends],
    [XLA and libtpu, with framework backends],
    [TT-Metalium and TTNN; libtt is a drop-in replacement for libtpu, integrates JAX and PyTorch],

    [Compilation and replay],
    [Framework graph compilation; CUDA/HIP graphs capture and replay launches @cudaguide @hipgraphs],
    [Framework tracing and XLA compilation; reuse compiled executables @jaxjit],
    [Libtt compiles framework graphs and uses device traces to replay their commands @libtt],

    table.hline(stroke: 0.8pt),
  ),
) <accelerator-comparison>

== Tiles

Tensors are usually stored as tiles of 32 by 32 numbers. The last two dimensions are rounded up to multiples of 32, with unused entries filled by padding. For large weight matrices, this overhead is usually small. During autoregressive decoding, however, each request contributes only one token per step, so small batches often leave most of a tile's 32 rows unused. A single activation row of 4,096 numbers occupies 128 tiles, with 31 of every 32 entries devoted to padding. Conventional GPUs can operate on these short rows without requiring the same 32-row padded storage layout. Tensors can also be stored in row-major form without this padding, but the chip's tile-based matrix units cannot operate on that layout directly.

Numbers in a tile are often 16-bit floats in the BF16 format or 8-bit floats in FP8 format with explicit scales. Numbers can also be stored in _block_ formats, in which sixteen neighboring values share one exponent and keep 8 or 4 bits each (BFP8 and BFP4). The hardware also supports 32-bit float and various different integer formats.

== The five RISC-V processors

Each core has five small processors, called baby cores by Tenstorrent. They are 32-bit, in-order, single-issue RISC-V processors (RV32IM plus a few small extensions). They perform none of a model's arithmetic; they decide what data moves and what the arithmetic units do next.

Two of them move data. Either can instruct either network to copy bytes between DRAM, this core's memory, and another core's memory. Usually one runs a _reader_, which brings tiles in, and the other a _writer_, which sends results on.

The other three drive the arithmetic. The first directs the unpacker, which loads tiles from memory into the arithmetic units' registers. The second issues instructions to the arithmetic units. The third directs the packer, which stores results back to memory.

All five share the core's 1.5 megabytes, which hold their code and the buffers between them. Each buffer is a queue of tiles: the reader fills one and the unpacker empties it, and the packer fills another and the writer empties it. The stages therefore overlap: the next tile arrives while the units work on the current one.

== The two arithmetic units

A core has two arithmetic units: a matrix unit that is fast but coarse, and a vector unit that is general but slow. For each operation, libtt must choose between them and select the precision settings.

The matrix unit multiplies tiles. One instruction multiplies an 8 by 16 block by a 16 by 16 block, which is 2,048 multiplications and as many additions, and the unit issues one such instruction per cycle @ttisa. A 16 by 16 block is called a _face_, and a 32 by 32 tile consists of four faces, so multiplying two tiles takes 16 instructions. The unit also adds, subtracts, and multiplies element by element, and takes the maximum of a column.

The unit is fast because its datapath is narrow. Operands are read through 19-bit registers, which is enough for BF16 and the block formats but not for a 32-bit float, which loses the low 13 bits of its fraction on the way in @ttisa. The multipliers are narrower still, so a single pass uses only the leading bits of each operand. Using the remaining bits takes further passes, called _fidelity phases_, each as costly as the first. For BF16, two phases cover nearly all of the bits and four cover all of them.

The vector unit handles everything else, such as the nonlinear functions in activations, norms, and softmax. It operates on 32 values at a time, each a 32-bit float or integer, and it reads and writes the matrix unit's results directly, so a product can feed a function of that product without leaving the core. It is also much slower: one instruction processes 32 values, where a matrix-unit instruction processes 2,048.

== Programs

Work on the chip is expressed as kernels. A _kernel_ is a small #cpp program. In the usual arrangement, one processor runs a _reader_, which fetches tiles from DRAM or from another core into buffers in the core's memory; three run a _compute_ kernel, which unpacks tiles into the arithmetic units, operates on them, and packs the results; and one runs a _writer_, which sends the results on. The kernels on all the cores that take part, together with their arguments, make up a _program_. One operation of a model, such as a matrix product or a norm, is ordinarily one program.

Kernels are compiled on the host by a RISC-V compiler the first time they are needed, and the compiled binaries are cached on disk. The host does not start programs on the cores itself. Instead, it writes commands into a queue, and two kernels that are always running on the chip, a _prefetcher_ and a _dispatcher_, read the queue, hand each core its kernels and arguments, and signal it to start.

A sequence of commands can also be recorded on the chip and replayed later with a single command. Such a recording is called a _trace_.

== Impact on the software stack

Three characteristics of the hardware and the PyTorch and JAX workloads we run on top guide libtt's design.

_Starting a program is not free_, so libtt fuses programs. A small program, one that adds two tensors, say, takes about 5 microseconds on all 110 cores and about 2 on only the cores with work. That is small, but a language model runs a great many programs during decode: Qwen3.8-27B on four chips runs more than 1,000 for each decode step. So libtt fuses operations to do the same arithmetic in fewer programs, and launches each one only on the cores that have work. It also uses tracing to avoid host dispatch overhead when launching programs.

_A row is a bad shape for a tile_, so libtt moves only the rows that hold data. Many intermediate tensors during small-batch decoding contain only one row, or a few. Reading and writing whole tiles can therefore move up to 32 times as much data as the computation needs. Some of libtt's kernels differ from the ones they replace in touching only the rows that hold data. Sometimes we use different logical and physical tensor dimensions to reduce wasted space by padding. For example, a single large vector of size $n$ might be stored as a $(n/32, 32)$ shape tensor.

_Speed is limited by memory_, so libtt keeps the memory busy with weights and with little else. To produce a token, a dense model reads nearly every one of its weights, and in a decoding step the time of a matrix product goes to streaming them. Libtt's kernel for this product gives every core weights to read, reads on both of a core's networks at once, and spreads the reads over the banks of memory. With several chips, each holds a share of the weights and all read at once. Intermediate results stay on the chip: a fused program passes them from core to core instead of through memory.

These choices must also preserve the program’s numerical requirements. The compiler selects arithmetic units and accumulation precision accordingly, rather than relying on the fastest defaults.
