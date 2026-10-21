# Current performance verification

Reference: original scikit-image 0.26.0. 1228 comparisons on Windows-11-10.0.26200-SP0; CPU: Intel64 Family 6 Model 151 Stepping 2, GenuineIntel. Individual reports record timestamps, CPU, versions, raw samples and options. All reports match the current source SHA-256:
`fc888e0fb93c2cdd5e923abfe9fb4605d4116626afd7f81957ceb4a14f87eee8`.

| Report | Cases | Worker counts |
|---|---:|---|
| [Every covered public entry](function-coverage.md) | 724 | 1, 4 |
| [Additional integer widths](conversion-widths.md) | 112 | 1, 4 |
| [Non-default exposure parameters](exposure-parameters.md) | 24 | 1 |
| [Color layouts and uint16 inputs](color-layouts.md) | 88 | 1 |
| [Perceptual color parameters](perceptual-parameters.md) | 40 | 1, 4 |
| [Morphology footprints and output reuse](morphology-parameters.md) | 240 | 1, 4 |

Every workload checks values and dtype against original scikit-image before timing with fallback disabled. Integer-width conversion cases also require exact values and matching identity behavior. Timings include allocation, Python dispatch, integer normalization and any needed input copying. Inputs/imports/I/O are excluded. Reference thread pools use one thread. Native worker counts are explicit.

Examples from the 1024² one-worker default-parameter comparisons:

| Operation | Input | Original ms | C++ ms | Reference / C++ |
|---|---|---:|---:|---:|
| morphology.erosion | uint8 | 11.1594 | 0.6082 | 18.35x |
| color.rgb2yuv | uint8 | 19.0767 | 8.8989 | 2.14x |
| color.rgb2xyz | uint8 | 108.0247 | 9.4278 | 11.46x |
| filters.gaussian | uint8 | 26.2793 | 7.0796 | 3.71x |
| filters.sobel | uint8 | 33.9390 | 4.4231 | 7.67x |
| filters.threshold_otsu | uint8 | 6.4635 | 0.3320 | 19.47x |
| morphology.opening | float32 | 9.3159 | 2.4473 | 3.81x |
| color.rgb2xyz | float32 | 58.5949 | 21.8540 | 2.68x |
| color.xyz2rgb | float32 | 74.6081 | 22.9287 | 3.25x |
| color.rgb2hsv | float32 | 155.7314 | 4.0004 | 38.93x |
| color.hsv2rgb | float32 | 134.5673 | 4.4013 | 30.57x |
| morphology.binary_dilation | float64 | 15.7422 | 5.7984 | 2.71x |
| color.rgb2ycbcr | float64 | 14.0349 | 6.0308 | 2.33x |
| color.rgb2lab | float64 | 173.3013 | 49.4836 | 3.50x |
| color.lab2lch | float64 | 36.5890 | 13.3084 | 2.75x |

629 of 645 one-worker native processing workloads measured faster than the reference. These are workload cases, including repeated functions and parameter variants. Identity returns, copies and dtype metadata are excluded from this count and labeled separately in the full reports.

16 one-worker processing workloads remained slower:

| Operation | Input | Size | Layout | Reference / C++ |
|---|---|---:|---|---:|
| color.lab2rgb | float32 | 256² | standard | 0.94x |
| util.invert | float32 | 256² | standard | 0.91x |
| util.invert | float32 | 1024² | standard | 0.83x |
| util.invert | float64 | 1024² | standard | 0.95x |
| util.img_as_bool | float64 | 1024² | standard | 0.78x |
| util.img_as_float64 | bool | 1024² | standard | 0.96x |
| util.img_as_bool | int8 | 1024² | standard | 0.97x |
| util.img_as_int | uint16 | 1024² | standard | 0.64x |
| util.img_as_bool | uint64 | 1024² | standard | 0.96x |
| util.img_as_bool | int64 | 1024² | standard | 0.82x |
| color.rgbcie2rgb | float32 | 512² | strided | 0.80x |
| morphology.dilation | bool | 512² | rectangle-31 | 0.99x |
| morphology.opening | bool | 512² | rectangle-31 | 0.82x |
| morphology.closing | bool | 512² | rectangle-31 | 0.79x |
| morphology.white_tophat | bool | 512² | rectangle-31 | 0.88x |
| morphology.black_tophat | bool | 512² | rectangle-31 | 0.77x |

The color port fuses integer normalization, matrix arithmetic and sRGB transfer. Uniform floating layouts now avoid a full contiguous copy: planar input uses contiguous channel loads, other uniform strides use gathers, and scalar tails preserve the same arithmetic. General nonuniform layouts and unaligned arrays can still require copying. Simple matrix transforms remain sensitive to dispatch, shuffling and memory traffic; this run has no hardware-counter profiling to quantify their individual contribution. Further optimization remains necessary.

Morphology reuses row mappings across border pixels and stops binary reductions when the result is known for every SIMD lane. Large rectangular footprints use linear-work prefix/suffix scans; small input types use narrower working buffers when conversion preserves extrema, while 64-bit integer scans retain double buffers for upstream rounding. Contiguous scan outputs and top-hat subtraction use SIMD. Boolean grayscale rectangles remain a bottleneck: their full scan passes can cost more than the reference on nearly uniform intermediate images.

Measurements use ordinary system load without CPU affinity or frequency locking. Small differences near 1x can change between runs. These are measured comparisons on this machine, not universal speed guarantees. CI runs parity and benchmark smoke checks on Windows/Linux without asserting speed ratios on shared runners.

Coverage remains partial: 61 native image entries, 2 metadata helpers, 387 inventoried entries still pending. See the [inventory](../coverage/README.md), [roadmap](../coverage/ROADMAP.md) and [documented input contracts](../README.md). Reproduce all six reports, then run `python benchmarks/summarize.py` to check their source hashes and API coverage before regenerating this summary.
