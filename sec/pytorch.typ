= Framework Integration <frameworks>

JAX and PyTorch reach libtt through PJRT. Both use the same compiler and runtime; their frontends translate framework operations into StableHLO, and libtt supplies the device implementation described in the preceding sections.

== JAX <jax-integration>

The Python package registers libtt as JAX's `tt` platform through a plugin entry point. Setting the environment variable `JAX_PLATFORMS=tt` selects it, and supported programs use the ordinary JAX interface, including `jax.jit`. Options can be supplied for each compilation:

```
fast = jax.jit(f, compiler_options={
    "optimization_level": "O1", "enable_trace": "true"})
```

== PyTorch <pytorch>

Upstream tt-xla integrates PyTorch through PyTorch/XLA @ttxla. Its traditional lazy-tensor model records operations until a graph boundary and can require applications to manage execution boundaries. That is the wrong abstraction for libtt, whose goal is to run existing PyTorch software: ordinary eager execution and the standard `torch.compile` interface must remain the programming model.

Libtt therefore supports Pytorch workload through TorchTPU, Google's PyTorch backend for TPUs, which supports both eager execution and `torch.compile` @torchtpu. Google identifies TorchTPU as the replacement for PyTorch/XLA and recommends migrating existing workloads to it @pytorchontpu. Setting `LIBTT_TPU_COMPAT=1` enables libtt's TPU compatibility layer, which lets TorchTPU discover the Tenstorrent chips and use libtt through PJRT. Multi-chip execution follows the process model in @chips.

Software written for TPUs may also call hardware-specific kernels directly. The compatibility layer substitutes implementations when the relevant modules are imported, using Tenstorrent kernels or ordinary framework operations. This lets the supported TorchTitan and SGLang-TorchTPU paths run without editing the applications' source. The compiler and runtime changes in @compiler and @running support their mix of eager and compiled execution; @training and @serving describe the applications.
