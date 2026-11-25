= Conclusion <conclusion>

Libtt brings existing JAX and PyTorch programs to Tenstorrent through a shared compiler and runtime so that the software that people already run on JAX and PyTorch can run on Tenstorrent hardware without being rewritten. Libtt extends Tenstorrent's stack with framework compatibility, numerical support, and execution across chips, and with the adaptations that existing training and serving systems need. The results show that this holds for complete applications: TorchTitan fine-tuning, SGLang serving, and Miles reinforcement learning, from loading weights and training to serving and evaluation.

The approach works because libtt reuses existing interfaces and does not ask applications to adopt new ones. The TorchTPU backend supplies PyTorch's eager and compiled execution models, and libtt replaces the TPU-specific operations that supported applications depend on, so each new application needs only adapters. These adapters have to be kept in step with the software they support, and libtt checks them against upstream framework tests, which expose assumptions that chip-specific tests miss.

== Looking ahead

The results point to four directions that would widen what libtt can run, how efficiently it runs, and the scale it reaches.

_Framework coverage and numerics._ Today a server compiles the shapes it expects and SGLang-TorchTPU rejects anything outside that set, so supporting dimensions not fixed at compile time would let programs accept arbitrary inputs. Operation coverage would extend to eigenvalue decompositions, more complex-number arithmetic, and the remaining windowed reductions. Numerical work includes true 64-bit types (currently 32-bit except in random-number-generator state), BF16 operations that turn NaNs into infinities, FP8 subnormals, and the unsupported `float8_e8m0fnu` format.

_Memory use and execution._ Extending buffer reuse could reduce memory requirements further. Eligible donated inputs can already be released after their last use in single-device execution, but results still receive separate allocations, and inputs retained elsewhere and multi-device execution need further work. Moving loops whose termination depends on computed values onto the device would also remove the host from the critical path, since it currently drives them one iteration at a time.

_Distributed execution and hardware coverage._ Execution across machines would support deployments larger than the chips in one host. Related work includes automatic link recovery after a server is interrupted and support for processors beyond Blackhole.

_Training validation and scale._ The CPU comparisons cover a few steps on smaller models and selected layers and a scaled-down fine-tuning run (@training, @fine-tuning). Reference runs of full-size training on other hardware, and longer runs across more models and configurations, would establish convergence and scaling. Larger models also need further memory reductions or additional parallelism, such as context parallelism.

_Supporting a kernel language._ Currently Pallas is the most widely used language to write kernels for TPUs. We could try to adapt it to produce kernels for Tenstorrent accelerators, or port another kernel language like TileLang.

_Generalizing the compiler to other accelerators._ We believe the mechanisms in the compiler are broadly applicable to executing PyTorch and JAX programs on any tile-based accelerator. Writing a backend for TPUs could lead to an open-source drop-in replacement for `libtpu`. It might even make sense to write backends for AMD/Nvidia GPUs and keep the tensors tiled in GPU memory: Tensors could directly be streamed into tensor cores, and furthermore it would lead to a common software stack for tile-based accelerators and GPUs.

Contributing the generally useful compiler and runtime changes upstream would make them available to every Tenstorrent user and reduce the cost of maintaining libtt's patches.