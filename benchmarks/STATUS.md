# Current performance verification

Reference: original scikit-image 0.26.0. 1312 comparisons on Windows-11-10.0.26200-SP0; CPU: Intel64 Family 6 Model 151 Stepping 2, GenuineIntel. Individual reports record timestamps, CPU, versions, raw samples and options. All reports match the current source SHA-256:
`ff70e599ada128399a30dd5a9bf4250a659cf08a00061f77b43f7cd0fbdb582a`.

| Report | Cases | Worker counts |
|---|---:|---|
| [Every covered public entry](function-coverage.md) | 772 | 1, 4 |
| [Additional integer widths](conversion-widths.md) | 112 | 1, 4 |
| [Non-default exposure parameters](exposure-parameters.md) | 24 | 1 |
| [Color layouts and uint16 inputs](color-layouts.md) | 104 | 1 |
| [Perceptual color parameters](perceptual-parameters.md) | 60 | 1, 4 |
| [Morphology footprints and output reuse](morphology-parameters.md) | 240 | 1, 4 |

Every workload checks values and dtype against original scikit-image before timing with fallback disabled. Integer-width conversion cases also require exact values and matching identity behavior. Timings include allocation, Python dispatch, integer normalization and any needed input copying. Inputs/imports/I/O are excluded. Reference thread pools use one thread. Native worker counts are explicit.

Examples from the 1024² one-worker default-parameter comparisons:

| Operation | Input | Original ms | C++ ms | Reference / C++ |
|---|---|---:|---:|---:|
| morphology.erosion | uint8 | 11.0035 | 0.5466 | 20.13x |
| color.rgb2yuv | uint8 | 14.2213 | 7.6354 | 1.86x |
| color.rgb2xyz | uint8 | 83.2550 | 8.1118 | 10.26x |
| filters.gaussian | uint8 | 20.4996 | 5.8011 | 3.53x |
| filters.sobel | uint8 | 24.3548 | 3.3401 | 7.29x |
| filters.threshold_otsu | uint8 | 3.4305 | 0.2557 | 13.42x |
| morphology.opening | float32 | 7.9207 | 2.4884 | 3.18x |
| color.rgb2xyz | float32 | 55.4046 | 20.6852 | 2.68x |
| color.xyz2rgb | float32 | 69.7669 | 21.3605 | 3.27x |
| color.rgb2hsv | float32 | 138.9481 | 3.7035 | 37.52x |
| color.hsv2rgb | float32 | 118.9681 | 4.2101 | 28.26x |
| color.rgb2luv | float32 | 96.9003 | 31.0433 | 3.12x |
| color.luv2rgb | float32 | 57.0977 | 25.5947 | 2.23x |
| morphology.binary_dilation | float64 | 12.3652 | 5.0761 | 2.44x |
| color.rgb2ycbcr | float64 | 19.1780 | 6.5366 | 2.93x |
| color.rgb2lab | float64 | 205.5110 | 51.7208 | 3.97x |
| color.lab2lch | float64 | 34.2272 | 13.1023 | 2.61x |
| color.luv2xyz | float64 | 86.7854 | 6.8034 | 12.76x |

Boolean 31×31 rectangles, with one worker (the previous slower morphology workloads):

| Operation | Image | Original ms | C++ ms | Reference / C++ |
|---|---|---:|---:|---:|
| morphology.erosion | 512² | 2.2864 | 0.1654 | 13.82x |
| morphology.dilation | 512² | 2.2131 | 0.1477 | 14.99x |
| morphology.opening | 512² | 4.4049 | 0.3222 | 13.67x |
| morphology.closing | 512² | 3.5593 | 0.2078 | 17.13x |
| morphology.white_tophat | 512² | 4.7825 | 0.3429 | 13.95x |
| morphology.black_tophat | 512² | 3.5764 | 0.2347 | 15.24x |

The [fixed-workload tuning experiment](morphology-tuning.md) preserves separate historical before/after C++ snapshots. Its cases are not added to this current-source total.

684 of 695 one-worker native processing workloads measured faster than the reference. These are workload cases, including repeated functions and parameter variants. Identity returns, copies and dtype metadata are excluded from this count and labeled separately in the full reports.

11 one-worker processing workloads remained slower:

| Operation | Input | Size | Layout | Reference / C++ |
|---|---|---:|---|---:|
| color.luv2rgb | float32 | 256² | standard | 0.7416x |
| util.invert | float32 | 256² | standard | 0.8485x |
| util.img_as_bool | float32 | 256² | standard | 0.9478x |
| util.img_as_float64 | float32 | 1024² | standard | 0.9526x |
| util.img_as_float32 | float64 | 1024² | standard | 0.9984x |
| util.img_as_int | uint16 | 1024² | standard | 0.7577x |
| util.img_as_bool | uint32 | 1024² | standard | 0.8680x |
| util.img_as_bool | int32 | 1024² | standard | 0.9820x |
| util.img_as_bool | uint64 | 1024² | standard | 0.9616x |
| color.lch2lab | uint16 | 512² | channel-first | 0.9452x |
| color.rgb2luv | float32 | 512² | standard | 0.8718x |

The color port fuses integer normalization, matrix arithmetic and sRGB transfer. Uniform floating layouts now avoid a full contiguous copy: planar input uses contiguous channel loads, other uniform strides use gathers, and scalar tails preserve the same arithmetic. General nonuniform layouts and unaligned arrays can still require copying. Simple matrix transforms remain sensitive to dispatch, shuffling and memory traffic; this run has no hardware-counter profiling to quantify their individual contribution. Further optimization remains necessary.

Morphology reuses row mappings across border pixels and stops binary reductions when the result is known for every SIMD lane. Large rectangular footprints use linear-work prefix/suffix scans; small input types use narrower working buffers when conversion preserves extrema, while 64-bit integer scans retain double buffers for upstream rounding. Contiguous scan outputs and top-hat subtraction use SIMD. Canonical 2D boolean rectangles now use packed words, with SIMD packing, bitwise extrema and unpacking, avoiding the previous byte-wise scan passes. Other storage and boundary constants retain the general native path.

Luv conversions fuse sRGB/XYZ stages with vector chromaticity arithmetic. Cached three-value white metadata preserves NumPy dot rounding at singular denominators. Float32 inverse cube powers retain libm rounding because sRGB amplifies small XYZ differences near black; surrounding arithmetic remains vectorized.

Measurements use ordinary system load without CPU affinity or frequency locking. Small differences near 1x can change between runs. These are measured comparisons on this machine, not universal speed guarantees. CI runs parity and benchmark smoke checks on Windows/Linux without asserting speed ratios on shared runners.

Coverage remains partial: 65 native image entries, 2 metadata helpers, 383 inventoried entries still pending. See the [inventory](../coverage/README.md), [roadmap](../coverage/ROADMAP.md) and [documented input contracts](../README.md). Reproduce all six reports, then run `python benchmarks/summarize.py` to check their source hashes and API coverage before regenerating this summary.
