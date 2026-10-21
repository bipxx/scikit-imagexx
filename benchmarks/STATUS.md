# Current performance verification

Reference: original scikit-image 0.26.0. 868 comparisons on Windows-11-10.0.26200-SP0; CPU: Intel64 Family 6 Model 151 Stepping 2, GenuineIntel. Individual reports record timestamps, CPU, versions, raw samples and options. All reports match the current source SHA-256:
`7aa5a4125a80126507c69cf8cb5922ba30f0409b08d626794fa5e4821ae96871`.

| Report | Cases | Worker counts |
|---|---:|---|
| [Every covered public entry](function-coverage.md) | 604 | 1, 4 |
| [Additional integer widths](conversion-widths.md) | 112 | 1, 4 |
| [Non-default exposure parameters](exposure-parameters.md) | 24 | 1 |
| [Color layouts and uint16 inputs](color-layouts.md) | 88 | 1 |
| [Perceptual color parameters](perceptual-parameters.md) | 40 | 1, 4 |

Every workload checks values and dtype against original scikit-image before timing with fallback disabled. Integer-width conversion cases also require exact values and matching identity behavior. Timings include allocation, Python dispatch, integer normalization and any needed input copying. Inputs/imports/I/O are excluded. Reference thread pools use one thread. Native worker counts are explicit.

Examples from the 1024² one-worker default-parameter comparisons:

| Operation | Input | Original ms | C++ ms | Reference / C++ |
|---|---|---:|---:|---:|
| color.rgb2yuv | uint8 | 17.5483 | 9.6370 | 1.82x |
| color.rgb2xyz | uint8 | 84.4793 | 7.5861 | 11.14x |
| filters.gaussian | uint8 | 25.5172 | 6.0452 | 4.22x |
| filters.sobel | uint8 | 28.3125 | 3.5277 | 8.03x |
| filters.threshold_otsu | uint8 | 4.3295 | 0.2756 | 15.71x |
| color.rgb2xyz | float32 | 62.2497 | 22.0475 | 2.82x |
| color.xyz2rgb | float32 | 82.8921 | 27.0797 | 3.06x |
| color.rgb2hsv | float32 | 149.2987 | 4.1397 | 36.07x |
| color.hsv2rgb | float32 | 121.8579 | 4.3510 | 28.01x |
| color.rgb2ycbcr | float64 | 14.0156 | 6.3176 | 2.22x |
| color.rgb2lab | float64 | 172.5166 | 51.0557 | 3.38x |
| color.lab2lch | float64 | 41.8614 | 18.3027 | 2.29x |

456 of 465 one-worker native processing workloads measured faster than the reference. These are workload cases, including repeated functions and parameter variants. Identity returns, copies and dtype metadata are excluded from this count and labeled separately in the full reports.

9 one-worker processing workloads remained slower:

| Operation | Input | Size | Layout | Reference / C++ |
|---|---|---:|---|---:|
| util.img_as_bool | float64 | 256² | standard | 0.64x |
| util.img_as_bool | float64 | 1024² | standard | 0.76x |
| util.img_as_float | bool | 1024² | standard | 0.95x |
| util.img_as_float64 | bool | 1024² | standard | 0.99x |
| util.img_as_bool | int8 | 1024² | standard | 0.94x |
| util.img_as_int | uint16 | 1024² | standard | 0.80x |
| util.img_as_bool | int16 | 1024² | standard | 0.94x |
| util.img_as_bool | int32 | 1024² | standard | 0.98x |
| util.img_as_bool | uint64 | 1024² | standard | 0.81x |

The color port fuses integer normalization, matrix arithmetic and sRGB transfer. Uniform floating layouts now avoid a full contiguous copy: planar input uses contiguous channel loads, other uniform strides use gathers, and scalar tails preserve the same arithmetic. General nonuniform layouts and unaligned arrays can still require copying. Simple matrix transforms remain sensitive to dispatch, shuffling and memory traffic; this run has no hardware-counter profiling to quantify their individual contribution. Further optimization remains necessary.

Measurements use ordinary system load without CPU affinity or frequency locking. Small differences near 1x can change between runs. These are measured comparisons on this machine, not universal speed guarantees. CI runs parity and benchmark smoke checks on Windows/Linux without asserting speed ratios on shared runners.

Coverage remains partial: 51 native image entries, 2 metadata helpers, 393 inventoried entries still pending. See the [inventory](../coverage/README.md), [roadmap](../coverage/ROADMAP.md) and [documented input contracts](../README.md). Reproduce all five reports, then run `python benchmarks/summarize.py` to check their source hashes and API coverage before regenerating this summary.
