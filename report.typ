// Libtt technical report.
//
//   typst compile --font-path fonts report.typ report.pdf
//
// On typst.app, upload this folder as a project (the fonts in fonts/ are
// picked up automatically) and open report.typ.

#import "style.typ": memo

#show: memo.with(
  title: [Libtt — A Compiler and Runtime for Tenstorrent Processors],
  author: "Philipp Moritz (pcmoritz@relativity.cc) and Sarah Dong (sarah@relativity.cc)",
  date: datetime(year: 2026, month: 11, day: 25),
  keywords: ("PJRT", "XLA", "JAX", "PyTorch", "Tenstorrent", "compilers", "language model serving"),
  abstract: [
    Libtt brings the PyTorch and JAX ecosystem to Tenstorrent hardware. It implements PJRT, the interface through which XLA-based frameworks drive Google's TPUs, so JAX runs on it directly, PyTorch runs through TorchTPU, and software written for TPUs carries over with no changes except replacing custom kernels. Built on Tenstorrent's open-source compiler and runtime, libtt ships the entire stack, from the compiler to the multi-chip runtime, as a single shared library. It contains extensive fixes to the upstream Tenstorrent software to make existing PyTorch and JAX code work out of the box, and extensive new compiler optimizations to make it fast. It passes most of the JAX and TorchTPU test suites and runs SGLang, TorchTitan, and Miles, covering serving, fine-tuning, and reinforcement learning.
  ],
)

#include "sec/intro.typ"
#include "sec/machine.typ"
#include "sec/history.typ"
#include "sec/library.typ"
#include "sec/compiler.typ"
#include "sec/running.typ"
#include "sec/chips.typ"
#include "sec/pytorch.typ"
#include "sec/training.typ"
#include "sec/serving.typ"
#include "sec/testing.typ"
#include "sec/remarks.typ"

#bibliography("refs.bib")
