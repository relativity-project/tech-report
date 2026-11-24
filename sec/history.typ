= Upstream Components <history>

Libtt builds on three upstream Tenstorrent projects: tt-xla supplies the PJRT plugin @ttxla, tt-mlir the compiler @ttmlir, and tt-metal the runtime, driver, and base kernel library, including the TTNN operation library @ttmetal.

TTNN's documented path for running a model is to port it: rewrite its operations against TTNN's Python API and choose tensor layouts, memory placement, and core configuration by hand @ttnnporting. This yields a separate implementation of each model. Libtt instead accepts framework programs as they are and makes those choices in the compiler and runtime, without the need for modifications or separate TTNN Python bindings (@library).

The upstream route for frameworks, tt-xla, was not sufficient for existing workloads. For JAX, missing lowerings and incorrect operation semantics were pervasive, and for PyTorch, PyTorch/XLA cannot run unmodified PyTorch code. The indexing, numerical, reduction, and control-flow changes in @compiler close the JAX gap, and @testing reports the resulting coverage. For PyTorch, libtt uses TorchTPU and its native eager and `torch.compile` execution model (@pytorch).