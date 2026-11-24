= Framework Compatibility <testing>

Libtt is designed to run existing JAX and PyTorch programs, and the applications built on them, without source changes. We test this against upstream framework test suites, which encode the frameworks' own definition of correct behavior, and against complete training and serving workloads. The framework interfaces are described in @frameworks; the results here measure how much existing software they run.

@jax-suite is a snapshot of suite coverage on one chip at the time of writing. Exclusions and test adaptations are listed below. The counts cover the selected suites and versions, not every feature of either framework.

#figure(
    caption: [Framework test suites on one chip, November 22, 2026: libtt 0.2.2 against upstream tt-xla 1.5.0.dev20260831. Under JAX 0.11.2 and `torch_tpu` 0.2.0, tt-xla crashes when it creates the client. Tests that crashed the process are counted as failed. PyTorch coverage is
  measured with the `torch_tpu` suite.],
    table(
      columns: 6,
      align: (left, left, right, right, right, right),
      table.hline(stroke: 0.8pt),
      table.header([Suite], [Stack], [Passed], [Failed], [Errors], [Skipped]),
      table.hline(stroke: 0.5pt),
      table.cell(rowspan: 2)[JAX 0.11.2], [libtt], [26,110], [1,663], [0], [6,737],
      [tt-xla], table.cell(colspan: 4, align: center)[crashes at start-up],
      table.hline(stroke: 0.3pt),
      table.cell(rowspan: 2)[JAX 0.7.1], [libtt], [24,075], [1,577], [0], [6,511],
      [tt-xla], [17,824], [7,816], [0], [6,523],
      table.hline(stroke: 0.3pt),
      table.cell(rowspan: 2)[`torch_tpu` 0.2.0], [libtt], [7,324], [1,677], [14], [243],
      [tt-xla], table.cell(colspan: 4, align: center)[crashes at start-up],
      table.hline(stroke: 0.8pt),
    ),
  ) <jax-suite>

== JAX

Supported programs use the ordinary JAX interface, including `jax.jit`, with libtt selected as the backend. To check this interface, the build fetches the source of the JAX version in use and runs its tests against the plugin without changing the test files. Thirteen files or directories that require facilities outside the plugin, like Pallas or CUDA, are excluded.

The suite exercises the behavior that existing programs depend on, including indexing, reductions, sorting, and control flow. It has been the main way of finding missing operations and incorrect semantics in the compiler. The changes in @compiler broaden the range of programs that compile and produce the expected results.

== PyTorch through TorchTPU

Libtt supports PyTorch eager execution and `torch.compile` through TorchTPU (@pytorch). The compatibility measurements use that backend’s test suite. Each file runs in a separate process, and operations are compared with CPU results using three samples for each operation and type. The original tests use reference files recorded on other hardware; two small patches adapt them for these comparisons, and four files are excluded.

The one-chip run includes failures from tests that require dynamic shapes or TPU-specific features. Its totals therefore reflect both unsupported behavior and differences between the test environment and the hardware the suite targets. Fixing more of these tests is an ongoing process.

== SGLang and complete applications

The two SGLang servers provide application-level compatibility checks. SGLang-JAX uses a Tenstorrent backend that calls libtt’s kernels. The supported SGLang-TorchTPU paths run without source edits: Libtt supplies adapters for TPU-specific operations. @serving describes these adaptations and the supported model families; @measure reports their performance.

Complete-model checks also exercise numerical behavior. Qwen3-8B scored 75.10 per cent on MMLU, answering 10,331 of 13,757 questions correctly @mmlu. Training comparisons use the same saved starting point on chip and CPU, with smaller models and selected layers (@training-checks). The TorchTitan fine-tuning run in @fine-tuning exercises checkpoint loading, training, export, serving, and evaluation. Together these checks provide evidence that the supported applications work through the full stack; they do not establish compatibility with every model or configuration.
