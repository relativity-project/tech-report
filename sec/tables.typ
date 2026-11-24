// Measured results, transcribed from https://relativity.cc/models/ (read
// November 11, 2026). SGLang-JAX rows: libtt 0.1.5, SGLang-JAX 6282ec6.
// SGLang-TorchTPU rows: libtt 0.2.0, torch_tpu 0.2.0, SGLang-TorchTPU 5191118.
// All on a TT-QuietBox 2 (four Blackhole chips).

#let dash = [—]
#let heavy = 0.8pt
#let light = 0.5pt
#let group(n, body) = table.cell(colspan: n, inset: (top: 0.55em, bottom: 0.3em), emph(body))

#let tab-single = figure(
  placement: auto,
  caption: [One request, 128 generated tokens, greedy. Decoding speed in tokens per second, and time to the first token of a 215-token prompt in milliseconds. A dash means that the model does not fit on that many chips.],
  table(
    columns: 8,
    align: (left, left, right, right, right, right, right, right),
    table.hline(stroke: heavy),
    table.header(
      table.cell(rowspan: 2, align: bottom + left)[Model],
      table.cell(rowspan: 2, align: bottom + left)[Kind],
      table.cell(colspan: 3, align: center)[Tokens per second],
      table.cell(colspan: 3, align: center)[First token, ms],
      table.hline(start: 2, end: 5, stroke: 0.3pt),
      table.hline(start: 5, end: 8, stroke: 0.3pt),
      [1 chip], [2 chips], [4 chips], [1 chip], [2 chips], [4 chips],
    ),
    table.hline(stroke: light),
    group(8)[SGLang-JAX, BFP8 weights (BFP4 experts), libtt 0.1.5],
    [Qwen3-8B], [dense], [46.0], [80.0], [122.2], [115], [77], [55],
    [Qwen3-14B], [dense], [26.6], [49.0], [83.7], [182], [112], [76],
    [Qwen3-32B], [dense], dash, [22.5], [40.7], dash, [232], [148],
    [Qwen3.5-9B], [hybrid], [40.8], [67.8], [104.7], [167], [114], [83],
    [Qwen3.8-27B], [hybrid], dash, [24.9], [42.2], dash, [286], [194],
    [Qwen3-30B-A3B], [MoE], dash, [103.7], [126.5], dash, [98], [72],
    [Qwen3.5-35B-A3B], [MoE], dash, [77.2], [95.6], dash, [149], [110],
    [Qwen3.8-Flash-Next], [MoE], dash, dash, [40.0], dash, dash, [815],
    table.hline(stroke: 0.3pt),
    group(8)[SGLang-TorchTPU, FP8 weights, libtt 0.2.0],
    [Qwen3-8B-FP8], [dense], [44.0], [73.1], [112.0], [251], [166], [123],
    [Qwen3-14B-FP8], [dense], [25.7], [45.8], [74.1], [394], [237], [168],
    [Qwen3-32B-FP8], [dense], dash, [22.2], [37.8], dash, [480], [319],
    [Qwen3.5-9B], [hybrid], [37.2], [59.3], [85.7], [458], [286], [197],
    [Qwen3.8-27B-FP8], [hybrid], dash, [25.1], [41.5], dash, [701], [473],
    table.hline(stroke: heavy),
  ),
)

#let tab-batched = figure(
  placement: auto,
  caption: [Several requests at once on four chips: total tokens per second over all requests, prefill included. Each request has a 19-token prompt and generates 128 tokens, greedy.],
  table(
    columns: 5,
    align: (left, right, right, right, right),
    table.hline(stroke: heavy),
    table.header(
      table.cell(rowspan: 2, align: bottom + left)[Model],
      table.cell(colspan: 4, align: center)[Concurrent requests],
      table.hline(start: 1, end: 5, stroke: 0.3pt),
      [1], [4], [8], [16],
    ),
    table.hline(stroke: light),
    group(5)[SGLang-JAX, libtt 0.1.5],
    [Qwen3-8B], [118], [402], [723], [1,101],
    [Qwen3-14B], [82], [286], [522], [870],
    [Qwen3-32B], [40], [141], [266], [467],
    [Qwen3.5-9B], [101], [345], [578], [897],
    [Qwen3.8-27B], [41], [143], [252], [405],
    [Qwen3-30B-A3B], [121], [388], [670], [973],
    [Qwen3.5-35B-A3B], [92], [303], [490], [720],
    [Qwen3.8-Flash-Next], [37], [95], [135], [162],
    table.hline(stroke: 0.3pt),
    group(5)[SGLang-TorchTPU, libtt 0.2.0],
    [Qwen3-8B-FP8], [103], [352], [567], [741],
    [Qwen3-14B-FP8], [69], [241], [413], [599],
    [Qwen3-32B-FP8], [35], [126], [215], [333],
    [Qwen3.5-9B], [76], [238], [355], [455],
    [Qwen3.8-27B-FP8], [36], [113], [169], [230],
    table.hline(stroke: heavy),
  ),
)

#let tab-sampling = figure(
  caption: [The cost of sampling: Qwen3-8B, one request, tokens per second.],
  table(
    columns: 5,
    align: (left, right, right, right, right),
    table.hline(stroke: heavy),
    table.header(
      table.cell(rowspan: 2, align: bottom + left)[Sampling],
      table.cell(colspan: 2, align: center)[SGLang-JAX],
      table.cell(colspan: 2, align: center)[SGLang-TorchTPU],
      table.hline(start: 1, end: 3, stroke: 0.3pt),
      table.hline(start: 3, end: 5, stroke: 0.3pt),
      [1 chip], [4 chips], [1 chip], [4 chips],
    ),
    table.hline(stroke: light),
    [greedy], [46.0], [122.2], [44.0], [112.0],
    [temperature 0.7], [44.2], [108.0], [42.3], [101.7],
    [and `top_p` 0.95], [38.6], [79.9], [38.7], [83.4],
    [and `top_k` 20], [38.7], [80.1], [43.0], [105.6],
    [and both], [34.3], [63.3], [42.0], [100.0],
    table.hline(stroke: heavy),
  ),
)

#let tab-spec = figure(
  placement: auto,
  caption: [Speculative decoding, one request, tokens per second without and with the draft. MTP uses the model's own prediction layer as the draft (five draft tokens a step under SGLang-JAX, three under SGLang-TorchTPU); DFlash uses a separate draft model.],
  table(
    columns: 6,
    align: (left, right, right, right, right, right),
    table.hline(stroke: heavy),
    table.header(
      table.cell(rowspan: 2, align: bottom + left)[Request],
      table.cell(rowspan: 2, align: bottom + right)[Chips],
      table.cell(colspan: 2, align: center)[SGLang-JAX],
      table.cell(colspan: 2, align: center)[SGLang-TorchTPU],
      table.hline(start: 2, end: 4, stroke: 0.3pt),
      table.hline(start: 4, end: 6, stroke: 0.3pt),
      [without], [with], [without], [with],
    ),
    table.hline(stroke: light),
    group(6)[MTP, Qwen3.8-27B],
    [5-token prompt, greedy], [4], [42.2], [106.4], [41.5], [81.5],
    [19-token code request, greedy], [4], [42.1], [86.6], [41.5], [81.5],
    [215-token summary request, greedy], [4], [41.9], [58.4], [41.2], [71.9],
    [5-token prompt, temperature 0.7], [4], [39.1], [90.1], [39.1], [53.1],
    [5-token prompt, greedy], [2], [24.9], [65.3], [25.1], [49.6],
    [215-token summary request, greedy], [2], [24.9], [55.9], [25.0], [45.9],
    table.hline(stroke: 0.3pt),
    group(6)[DFlash, Qwen3-8B],
    [19-token code request, greedy], [1], [44.0], [126.7], [43.9], [92.4],
    [215-token summary request, greedy], [1], [43.1], [75.8], [43.0], [54.3],
    table.hline(stroke: heavy),
  ),
)

#let tab-replaced = figure(
  placement: auto,
  caption: [Serving adaptations for SGLang-TorchTPU. † Added by libtt; ‡ extended or adapted from upstream Tenstorrent.],
  table(
    columns: 2,
    align: (left, left),
    table.hline(stroke: heavy),
    table.header([Operation], [Device implementation]),
    table.hline(stroke: light),
    [Attention and cache], [Paged attention and shared updates‡; cache pool adapter†],
    [FP8 projections], [Quantization and scaled decode kernels†; larger matrix products‡],
    [Mixture of experts], [Fused BF16/FP8 expert computation‡],
    [Recurrent layers], [Decode, convolution and state-pool kernels†; chunked prefill‡],
    [Draft cache updates], [Rotary/cache-write adapter† using paged updates‡],
    [Collectives], [Native all-gather‡],
    table.hline(stroke: heavy),
  ),
)
