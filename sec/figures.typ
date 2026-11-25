// Line drawings, in the manner of pic.
#import "@preview/cetz:0.5.3"

#let small(body) = text(size: 9pt, body)
#let lbl(body) = text(size: 8.5pt, style: "italic", body)

// Figure: the path from a framework to the chips.
#let fig-stack = figure(
  placement: auto,
  caption: [The path of a program. Everything inside the large box is one shared library.],
  cetz.canvas(length: 1cm, {
    import cetz.draw: *
    set-style(stroke: 0.6pt, mark: (fill: black, scale: 0.8))
    let arrow(a, b) = line(a, b, mark: (end: ">"))

    // Programs and frameworks.
    rect((0.6, 9.6), (6.4, 10.5), name: "pj")
    content("pj", small[JAX programs, SGLang-JAX])
    rect((7.6, 9.6), (13.4, 10.5), name: "pt")
    content("pt", small[PyTorch programs, SGLang-TorchTPU,\ TorchTitan, Miles])
    rect((0.6, 8.1), (6.4, 9.0), name: "jax")
    content("jax", small[JAX])
    rect((7.6, 8.1), (13.4, 9.0), name: "tpu")
    content("tpu", small[PyTorch with `torch_tpu`])
    arrow("pj.south", "jax.north")
    arrow("pt.south", "tpu.north")

    // The library.
    rect((0, 1.9), (14, 6.9), name: "lib", stroke: 1pt)
    content((0.25, 6.6), anchor: "west", small[`libtt.so`])
    arrow("jax.south", (3.5, 6.9))
    arrow("tpu.south", (10.5, 6.9))
    content((7, 7.5), align(center, lbl[StableHLO programs and buffers,\ through the PJRT C interface]))

    rect((1.2, 5.2), (12.8, 6.2), name: "pjrt")
    content("pjrt", small[PJRT plugin (from tt-xla): clients, devices, buffers, executables])
    rect((1.2, 3.75), (12.8, 4.75), name: "comp")
    content("comp", small[Compiler (from tt-mlir): StableHLO → TTIR → TTNN → program image])
    rect((1.2, 2.3), (12.8, 3.3), name: "rt")
    content("rt", small[Run-time system (from tt-metal): operations, kernels, dispatch, traces])
    arrow("pjrt.south", "comp.north")
    arrow("comp.south", "rt.north")

    // The chips.
    for i in range(4) {
      let x = 1.2 + i * 3.1
      rect((x, 0), (x + 2.3, 0.8), name: "c" + str(i))
      content("c" + str(i), small[Blackhole])
      if i > 0 {
        line((x - 0.8, 0.4), (x, 0.4), stroke: (dash: "dashed", thickness: 0.6pt))
      }
    }
    arrow((2.35, 1.9), "c0.north")
    arrow((5.45, 1.9), "c1.north")
    arrow((8.55, 1.9), "c2.north")
    arrow((11.65, 1.9), "c3.north")
    content((11.85, 1.35), anchor: "west", lbl[PCIe])
    content((13.2, 0.4), anchor: "west", lbl[Ethernet])
  }),
)

// Figure: one Tensix core, and where it sits.
#let fig-core = figure(
  placement: auto,
  caption: [A Blackhole chip is a grid of cores beside a large memory. Each core is a small computer with five RISC-V processors, a vector and a matrix unit, and a small amount of local SRAM.],
  cetz.canvas(length: 1cm, {
    import cetz.draw: *
    set-style(stroke: 0.6pt, mark: (fill: black, scale: 0.8))

    // The grid, drawn as a 4 by 4 corner of it.
    for i in range(4) {
      for j in range(4) {
        let x = 0.2 + i * 1.05
        let y = 0.9 + j * 1.05
        if i < 3 { line((x + 0.7, y + 0.35), (x + 1.05, y + 0.35)) }
        if j < 3 { line((x + 0.35, y + 0.7), (x + 0.35, y + 1.05)) }
        rect((x, y), (x + 0.7, y + 0.7), fill: if i == 3 and j == 2 { luma(200) } else { white })
      }
    }
    content((2.1, 5.35), small[120 cores, joined by two networks])
    rect((0.2, -0.3), (4.05, 0.5), name: "dram")
    content("dram", small[DRAM, 32 GB])
    line((2.1, 0.5), (2.1, 0.9))

    // One core, enlarged.
    line((4.05, 3.7), (5.6, 5.0), stroke: (dash: "dotted", thickness: 0.6pt))
    line((4.05, 3.0), (5.6, -0.3), stroke: (dash: "dotted", thickness: 0.6pt))
    rect((5.6, -0.3), (14, 5.0), stroke: 1pt)
    content((5.85, 4.65), anchor: "west", small[One core])
    rect((6.0, 3.1), (9.6, 4.3), name: "dm")
    content("dm", align(center, small[2 processors:\ read and write data]))
    rect((10.0, 3.1), (13.6, 4.3), name: "tr")
    content("tr", align(center, small[3 processors:\ unpack, compute, pack]))
    rect((6.0, 1.6), (13.6, 2.7), name: "l1")
    content("l1", small[Memory, 1.5 MB: kernels, their arguments, buffers of tiles])
    rect((6.0, 0.1), (9.6, 1.2), name: "fpu")
    content("fpu", small[Matrix unit])
    rect((10.0, 0.1), (13.6, 1.2), name: "sfpu")
    content("sfpu", small[Vector unit])
    line("dm.south", (7.8, 2.7), mark: (start: ">", end: ">"))
    line("tr.south", (11.8, 2.7), mark: (start: ">", end: ">"))
    line((7.8, 1.6), "fpu.north", mark: (start: ">", end: ">"))
    line((11.8, 1.6), "sfpu.north", mark: (start: ">", end: ">"))
  }),
)

// Figure: the path from StableHLO to the chip, pass by pass.
#let fig-pipeline = figure(
  placement: auto,
  caption: [How the compiler rewrites a program, from StableHLO to the table of calls that the run-time system walks. † marks what libtt adds; the rest is extended from upstream tt-mlir.],
  cetz.canvas(length: 1cm, {
    import cetz.draw: *
    set-style(stroke: 0.6pt, mark: (fill: black, scale: 0.8))
    let arrow(a, b) = line(a, b, mark: (end: ">"))
    // One line of a box: what happens on the left, where it is described on the
    // right. Both sit in one box so that they share a baseline.
    let row(y, body, sec: none) = content(
      (0.6, y), anchor: "west",
      box(width: 12.8cm, small[#body#h(1fr)#sec]),
    )

    content((7, 14.0), small[JAX or PyTorch program, through PJRT])
    arrow((7, 13.75), (7, 13.05))

    // StableHLO.
    rect((0.3, 11.35), (13.7, 13.0), name: "shlo")
    content((0.6, 12.55), anchor: "west", lbl[StableHLO — the framework's own operations])
    row(11.9, [tensors partitioned across chips by Shardy; program rewritten to per-chip shapes], sec: [Section 7])
    arrow((7, 11.35), (7, 10.4))

    // TTIR.
    rect((0.3, 6.7), (13.7, 10.35), name: "ttir")
    content((0.6, 9.9), anchor: "west", lbl[TTIR — tensor operations, no layouts yet])
    row(9.25, [indexing lowered to rows\_gather and rows\_scatter†], sec: [Section 5.2])
    row(8.75, [types legalized (complex, f64, i8/i16); FP16 and FP8 kept native†], sec: [Section 5.3])
    row(8.25, [calls inlined, constants folded, common sub-expressions eliminated])
    row(7.75, [fused: attention prep†, residual RMSNorm†, SwiGLU†, matmul + activation], sec: [Section 5.4])
    row(7.25, [wide rows reshaped to fill whole tiles†], sec: [Section 5.6])
    arrow((7, 6.7), (7, 5.75))

    // TTNN.
    rect((0.3, 3.05), (13.7, 5.7), name: "ttnn")
    content((0.6, 5.25), anchor: "west", lbl[TTNN — one operation per run-time call])
    row(4.6, [each tensor assigned a layout (tiled or row-major), placement, and type], sec: [Section 5.6])
    row(4.1, [operations fused across calls; arithmetic settings assigned per operation])
    row(3.6, [when optimizing, layouts and types chosen from those the operation library accepts])
    arrow((7, 3.05), (7, 2.1))

    // Program image.
    rect((0.3, 0.4), (13.7, 2.05), name: "image")
    content((0.6, 1.6), anchor: "west", lbl[program image — a table of calls])
    row(0.95, [calls serialized in order, to be walked by a runtime interpreter], sec: [Section 6])
}),
)