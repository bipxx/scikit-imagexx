# Current performance verification

Reference: original scikit-image 0.26.0. 688 comparisons on Windows-11-10.0.26200-SP0; CPU: Intel64 Family 6 Model 151 Stepping 2, GenuineIntel. Individual reports record timestamps, CPU, versions, raw samples and options. All reports match the current source SHA-256:
`20512ae2801aa16e2901fbf11ee52e8dfd84f68647a333be7297ba881f7b512c`.

| Report | Cases | Worker counts |
|---|---:|---|
| [Every covered public entry](function-coverage.md) | 496 | 1, 4 |
| [Additional integer widths](conversion-widths.md) | 112 | 1, 4 |
| [Non-default exposure parameters](exposure-parameters.md) | 24 | 1 |
| [Color layouts and uint16 inputs](color-layouts.md) | 56 | 1 |

Every workload checks values and dtype against original scikit-image before timing with fallback disabled. Integer-width conversion cases also require exact values and matching identity behavior. Timings include allocation, Python dispatch, integer normalization and any needed input copying. Inputs/imports/I/O are excluded. Reference thread pools use one thread. Native worker counts are explicit.

Examples from the 1024² one-worker default-parameter comparisons:

| Operation | Input | Original ms | C++ ms | Reference / C++ |
|---|---|---:|---:|---:|
| color.rgb2yuv | uint8 | 25.7404 | 11.9019 | 2.16x |
| color.rgb2xyz | uint8 | 173.6083 | 13.8332 | 12.55x |
| filters.gaussian | uint8 | 46.0853 | 9.5265 | 4.84x |
| filters.sobel | uint8 | 47.6817 | 4.9656 | 9.60x |
| filters.threshold_otsu | uint8 | 7.8884 | 0.4109 | 19.20x |
| color.rgb2xyz | float32 | 95.3641 | 33.9132 | 2.81x |
| color.xyz2rgb | float32 | 113.4824 | 33.3330 | 3.40x |
| color.rgb2ycbcr | float64 | 28.7920 | 11.9672 | 2.41x |

353 of 365 one-worker native processing workloads measured faster than the reference. These are workload cases, including repeated functions and parameter variants. Identity returns, copies and dtype metadata are excluded from this count and labeled separately in the full reports.

12 one-worker processing workloads remained slower:

| Operation | Input | Size | Layout | Reference / C++ |
|---|---|---:|---|---:|
| util.img_as_bool | float64 | 256² | standard | 0.69x |
| util.img_as_bool | uint8 | 1024² | standard | 0.89x |
| util.invert | float32 | 1024² | standard | 0.70x |
| util.img_as_float64 | float32 | 1024² | standard | 0.95x |
| color.rgb2gray | float64 | 1024² | standard | 0.98x |
| util.invert | float64 | 1024² | standard | 0.99x |
| util.img_as_float32 | float64 | 1024² | standard | 0.98x |
| util.img_as_bool | float64 | 1024² | standard | 0.96x |
| util.img_as_bool | uint64 | 1024² | standard | 0.92x |
| color.rgb2yuv | float32 | 512² | strided | 0.84x |
| color.yuv2rgb | float32 | 512² | strided | 0.94x |
| color.yiq2rgb | float32 | 512² | strided | 0.94x |

The color port fuses integer normalization, matrix arithmetic and sRGB transfer. Uniform floating layouts now avoid a full contiguous copy: planar input uses contiguous channel loads, other uniform strides use gathers, and scalar tails preserve the same arithmetic. General nonuniform layouts and unaligned arrays can still require copying. Simple matrix transforms remain sensitive to dispatch, shuffling and memory traffic; this run has no hardware-counter profiling to quantify their individual contribution. Further optimization remains necessary.

Measurements use ordinary system load without CPU affinity or frequency locking. Small differences near 1x can change between runs. These are measured comparisons on this machine, not universal speed guarantees. CI runs parity and benchmark smoke checks on Windows/Linux without asserting speed ratios on shared runners.

Coverage remains partial: 43 native image entries, 1 metadata helper, 402 inventoried entries still pending. See the [inventory](../coverage/README.md), [roadmap](../coverage/ROADMAP.md) and [documented input contracts](../README.md). Reproduce all four reports, then run `python benchmarks/summarize.py` to check their source hashes and API coverage before regenerating this summary.
