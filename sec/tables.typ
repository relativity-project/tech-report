// Measured results, transcribed from https://relativity.cc/models/ (read
// November 23, 2026). All rows: libtt 0.2.2. SGLang-JAX rows: SGLang-JAX
// 6282ec6. SGLang-TorchTPU rows: torch_tpu 0.2.0, SGLang-TorchTPU 5191118.
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
    group(8)[SGLang-JAX, BFP8 weights (BFP4 experts)],
    [Qwen3-8B], [dense], [47.0], [81.0], [124.4], [113], [76], [54],
    [Qwen3-14B], [dense], [27.0], [49.4], [83.3], [179], [110], [76],
    [Qwen3-32B], [dense], dash, [22.7], [40.6], dash, [229], [147],
    [Qwen3.5-9B], [hybrid], [43.5], [74.7], [117.0], [159], [108], [78],
    [Qwen3.8-27B], [hybrid], dash, [26.7], [46.1], dash, [271], [179],
    [Qwen3-30B-A3B], [MoE], dash, [104.3], [127.4], dash, [97], [69],
    [Qwen3.5-35B-A3B], [MoE], dash, [88.9], [109.2], dash, [141], [104],
    [Qwen3.8-Flash-Next], [MoE], dash, dash, [42.7], dash, dash, [810],
    table.hline(stroke: 0.3pt),
    group(8)[SGLang-TorchTPU, FP8 weights],
    [Qwen3-8B-FP8], [dense], [44.4], [75.0], [116.0], [251], [166], [122],
    [Qwen3-14B-FP8], [dense], [25.9], [46.8], [76.2], [393], [235], [167],
    [Qwen3-32B-FP8], [dense], dash, [22.6], [38.8], dash, [477], [317],
    [Qwen3.5-9B], [hybrid], [37.2], [59.9], [86.5], [451], [279], [191],
    [Qwen3.8-27B-FP8], [hybrid], dash, [25.3], [42.0], dash, [693], [457],
    [Qwen3-30B-A3B-FP8], [MoE], dash, [87.0], [97.4], dash, [196], [143],
    [Qwen3.5-35B-A3B-FP8], [MoE], dash, [73.2], [83.0], dash, [340], [239],
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
    group(5)[SGLang-JAX],
    [Qwen3-8B], [120], [439], [770], [1,139],
    [Qwen3-14B], [81], [302], [541], [902],
    [Qwen3-32B], [39], [148], [274], [444],
    [Qwen3.5-9B], [113], [392], [660], [999],
    [Qwen3.8-27B], [45], [160], [284], [448],
    [Qwen3-30B-A3B], [121], [424], [707], [1,014],
    [Qwen3.5-35B-A3B], [104], [342], [566], [820],
    [Qwen3.8-Flash-Next], [39], [100], [145], [170],
    table.hline(stroke: 0.3pt),
    group(5)[SGLang-TorchTPU],
    [Qwen3-8B-FP8], [106], [367], [580], [750],
    [Qwen3-14B-FP8], [71], [249], [422], [609],
    [Qwen3-32B-FP8], [36], [129], [219], [344],
    [Qwen3.5-9B], [77], [242], [360], [466],
    [Qwen3.8-27B-FP8], [37], [115], [172], [235],
    [Qwen3-30B-A3B-FP8], [88], [315], [500], [666],
    [Qwen3.5-35B-A3B-FP8], [72], [238], [339], [432],
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
    [greedy], [47.0], [124.4], [44.4], [116.0],
    [temperature 0.7], [45.2], [110.9], [42.7], [105.3],
    [and `top_p` 0.95], [39.9], [83.6], [39.0], [85.5],
    [and `top_k` 20], [39.9], [83.6], [43.4], [109.4],
    [and both], [35.6], [66.8], [42.4], [103.5],
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
    [5-token prompt, greedy], [4], [46.0], [119.8], [42.0], [82.3],
    [19-token code request, greedy], [4], [46.0], [102.7], [42.0], [82.2],
    [215-token summary request, greedy], [4], [45.7], [56.8], [41.8], [72.7],
    [5-token prompt, temperature 0.7], [4], [42.8], [100.7], [39.6], [61.6],
    [5-token prompt, greedy], [2], [26.7], [72.2], [25.3], [50.2],
    [215-token summary request, greedy], [2], [26.6], [38.6], [25.3], [46.5],
    table.hline(stroke: 0.3pt),
    group(6)[DFlash, Qwen3-8B],
    [19-token code request, greedy], [1], [44.6], [114.7], [44.3], [95.2],
    [215-token summary request, greedy], [1], [43.8], [58.3], [43.6], [57.0],
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
