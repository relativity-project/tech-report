= Executing <running>

== Traces

A decode step can contain hundreds or thousands of small device programs, and dispatching each one from the host leaves the chip idle while it waits for work. Upstream TT-Metalium offers traces, recorded sequences of device commands that replay with a single request @ttmetal. Libtt greatly extends this mechanism so it supports the dynamic execution patterns real PyTorch and JAX workloads exhibit. Short eager operations run without trace capture, while larger programs can use replay.

_Memory._ Recorded commands refer to fixed addresses, so libtt tracks the memory each trace uses and checks that replay cannot overwrite live tensors. Libtt also supports allocations inside a trace and reserves the DRAM used by temporary tensors, so unrelated allocations rarely force a trace to be recorded again. Under memory pressure the reservations are released, and the runtime re-records any trace that needs it.

_Inputs._ Tensors already on the device stay there during replay. They are read in place when possible, or copied into fixed device buffers when their addresses change. Even inputs the compiled program expects from the host can be staged this way, so a result never has to return to the host just to feed the next compiled function.

_Branches._ Libtt adds conditional commands to the command processor, and the compiler records supported branches inside the trace. The device reads the condition from memory and skips branches that are not taken, so nested conditionals such as the token sampler run with no round trip to the host.

Every branch’s kernels must be compiled and loaded before capture, so libtt adds a preparation mode that does this without executing them. Preparing an unselected branch therefore leaves the model’s state unchanged.

== Host overhead

Traces cut the cost of dispatching device programs, but the host still prepares arguments, transfers inputs, and receives results. Libtt shrinks this work and overlaps it with device execution wherever dependencies allow @libtt.

_Caching._ Most of what is needed to launch a compiled program is the same from call to call. Libtt caches input layouts and the results of input analysis, builds sharding information only when a new runtime tensor requires it, and starts trace replay directly without constructing another program executor. Compiled programs can also be saved and loaded for `torch.compile`’s persistent cache, provided the same libtt build is used.

_Data movement._ Small transfers such as token IDs and positions are issued from the calling thread, since handing them to a thread pool can cost more than the copy itself. Larger host copies use contiguous blocks and, for supported transposes, vector instructions, and read buffers are no longer zeroed before being overwritten. Where supported, tiled results are converted to row-major on the device before transfer, which reduces both the bytes read and the CPU work.

_Overlap._ Libtt queues result transfers in program order and releases submission locks before waiting for completion, so a serving system can read finished tokens while it prepares and submits later work, with no gaps between device computations.

== Memory

Weights are large, and moving them accounts for most of a server’s startup time. Libtt supports memory-mapped checkpoints, without first copying each tensor, and can convert weights to their block format on the chip instead of on the host.