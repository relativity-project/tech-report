#import "../style.typ": cpp

= The Library <library>

== One library

Everything is in one shared library, `libtt.so`. It contains the PJRT plugin, the compiler with LLVM's MLIR libraries @mlir beneath it, StableHLO and the Shardy partitioner @shardy, the run-time system and its library of operations, and the driver that talks to the chip, all linked statically via Bazel. It exports three symbols specified by PJRT, and a linker script hides the rest. The compiler's output calls TTNN's C++ operations directly. The Python code shipped with libtt registers the plugin and adapts framework and server interfaces (@frameworks, @serving).

== The compiler

Kernels are compiled on the user's machine, so the library has to bring a compiler with it. Embedded in `libtt.so` is a compressed archive that holds a RISC-V GCC, the sources of every kernel and of the firmware, and the linker scripts. When the library is loaded it unpacks the archive into a temporary directory named after the archive's hash, and points the run-time system at it. A file lock makes this safe when several processes start at once. The compiler and runtime are compressed aggressively, so the full release build of `libtt.so` is less than 90 MB.

== Patches

Making the upstream stack support existing framework programs required substantial changes throughout the system. As of the date of writing, libtt carries nearly 400 patches across the open-source Tenstorrent software stack with over 50,000 lines of change, together with approximately 15,000 lines of additional source code. These changes span the PJRT interface, compiler, run-time system, and device kernels: they add missing functionality, correct operation semantics, extend numerics type support, and improve performance significantly. The build pins the upstream revisions and combines them with these extensions into a reproducible stack.

Many of these changes address general limitations of the underlying software and would be natural candidates for upstream inclusion. In particular, correctness fixes, numerics type support, and general-purpose compiler and kernel improvements could benefit other users of Tenstorrent’s stack. Contributing these changes upstream would broaden their availability and reduce the work of maintaining them separately. Libtt could then concentrate more of its development on framework integration and optimizations that build on those shared capabilities.

