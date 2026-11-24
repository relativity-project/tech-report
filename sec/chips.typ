= Distributed Execution Across Chips <chips>

Libtt makes a program run across multiple chips. It organizes them into a logical _mesh_, lets each array be divided across one or more mesh axes, replicated across others, or held on a single chip, and adds collective operations that exchange or combine data when a computation needs values held elsewhere.

This supports many different ways of dividing work, including the following ones: _Data parallelism_ assigns different examples to different chips, usually with a copy of the model on each. _Tensor parallelism_ divides the tensors and computations within a layer @megatron. _Expert parallelism_ places the experts of a mixture-of-experts model on different chips. _Fully sharded data parallelism_ distributes parameters, gradients, and optimizer state across chips and gathers parameters when needed. _Pipeline parallelism_ assigns successive groups of layers to different chips and passes intermediate results between them. _Sequence parallelism_ divides some intermediate activations along the sequence dimension, and _context parallelism_ also distributes attention over parts of a long sequence. TorchTitan's four-chip training runs are a concrete example of fully sharded data parallelism (@training).

Libtt implements the local computation and communication that the framework's partitioning requires @libtt. The chips can be driven from a single controller or from one process per chip, as described below.

== Single controller

A JAX program states how its arrays are divided by annotating them, and the Shardy partitioner propagates the annotations through the program. With `shard_map`, the programmer can also write the per-chip computation explicitly and mix it with automatically partitioned code. Libtt supports both forms and their combination. After partitioning, the compiler rewrites every shape to the piece one chip holds, and whatever must cross between chips appears as an explicit _collective_: an all-reduce that sums, or takes the maximum or minimum of, the values the chips hold; an all-gather that concatenates them; or a send from each chip to another.

A single controller drives all the chips. Data crosses between them over their Ethernet links, routed by small kernels on each chip, and never passes through the host. No launcher or message-passing library is needed: the server starts as it does on one chip, with one additional flag.

All-reduce matters most for tensor-parallel decoding, where it combines partial results within every layer. It began as two programs and is now one: each chip sends its rows directly into spare memory on the others, and each adds what it receives. Only rows that hold data are sent. Sums that must be exact, of integers or 32-bit floats, are computed on the vector unit.

Before libtt, a row-divided embedding table forced a choice: gather the whole table onto every chip for each lookup, or keep a full copy on every chip. Libtt removes the need for either. Each chip looks the token up in its own rows and returns zeros if the row is elsewhere, and an all-reduce sums the results. Each chip now stores only its share of the table, so memory per chip falls as chips are added, and only one embedding vector per token crosses between chips, not the table.

In mixture-of-experts layers, each chip holds a subset of the experts and computes their contributions to the selected tokens, and a collective sums the contributions into the layer's output @libtt.

== One process per chip

TorchTPU runs one process per chip, as is usual on a machine full of TPUs, and each process compiles the same program with the collectives written out. For TorchTPU, libtt follows this model. Each process opens only its own chip. The processes agree on a mesh of $N$ chips in a line and find one another through a small key-value store that the first process serves over TCP, replacing the message-passing library upstream would use. Only coordination goes through the store; tensors still cross over the chips' own links.

Collectives must write to the addresses where other chips actually hold their buffers. Upstream implementations often assume these addresses match across chips, which independent processes cannot guarantee. Libtt extends the all-gather and reduce-scatter kernels used between processes to exchange buffer addresses during their existing startup handshake, so each chip writes to the address its neighbor supplies. The small direct all-reduce instead uses buffers that its program places at matching L1 addresses.