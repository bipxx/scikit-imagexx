# Current performance verification

Reference: original scikit-image 0.26.0. 2026 comparisons on Windows-11-10.0.26200-SP0; CPU: Intel64 Family 6 Model 151 Stepping 2, GenuineIntel. Individual reports record timestamps, CPU, versions, raw samples and options. All reports match the current source/build SHA-256 (POSIX relative paths and LF line endings):
`c30707e1ac5b75dfa181037672afc3bf2b9ae916c6b1838cecb20123d8a21546`.

| Report | Cases | Worker counts |
|---|---:|---|
| [Every covered public entry](function-coverage.md) | 916 | 1, 4 |
| [Additional integer widths](conversion-widths.md) | 112 | 1, 4 |
| [Non-default exposure parameters](exposure-parameters.md) | 24 | 1 |
| [Color layouts and uint16 inputs](color-layouts.md) | 120 | 1 |
| [Perceptual color parameters](perceptual-parameters.md) | 60 | 1, 4 |
| [Morphology footprints and output reuse](morphology-parameters.md) | 240 | 1, 4 |
| [Stain matrices and output widths](stain-parameters.md) | 156 | 1, 4 |
| [Lab distances and color routes](color-distance.md) | 246 | 1, 4 |
| [Histogram threshold parameters](threshold-parameters.md) | 152 | 1, 4 |

Every workload checks values and dtype against original scikit-image before timing with fallback disabled. Integer-width conversion cases also require exact values and matching identity behavior. Timings include allocation, Python dispatch, integer normalization and any needed input copying. Inputs/imports/I/O are excluded. Reference thread pools use one thread. Native worker counts are explicit.

Examples from the 1024² one-worker default-parameter comparisons (color-space routing is RGB to HSV; CIE76 compares raw Lab values with a channel-reversed second input):

| Operation | Input | Original ms | C++ ms | Reference / C++ |
|---|---|---:|---:|---:|
| morphology.erosion | uint8 | 11.1188 | 0.7100 | 15.66x |
| color.rgb2yuv | uint8 | 14.5957 | 7.7302 | 1.89x |
| color.rgb2xyz | uint8 | 85.2721 | 8.3194 | 10.25x |
| color.rgb2hed | uint8 | 43.0086 | 14.4333 | 2.98x |
| filters.gaussian | uint8 | 30.6950 | 7.1837 | 4.27x |
| filters.sobel | uint8 | 25.4112 | 3.9705 | 6.40x |
| filters.threshold_otsu | uint8 | 4.0078 | 0.2676 | 14.98x |
| morphology.opening | float32 | 9.4326 | 4.1744 | 2.26x |
| color.rgb2xyz | float32 | 62.4867 | 26.3872 | 2.37x |
| color.xyz2rgb | float32 | 78.7126 | 34.6640 | 2.27x |
| color.rgb2hsv | float32 | 151.9181 | 3.9561 | 38.40x |
| color.hsv2rgb | float32 | 229.0346 | 8.8346 | 25.92x |
| color.rgb2luv | float32 | 93.3644 | 30.9512 | 3.02x |
| color.luv2rgb | float32 | 50.2761 | 4.7215 | 10.65x |
| color.separate_stains | float32 | 24.9312 | 13.1419 | 1.90x |
| color.deltaE_cie76 | float32 | 7.7125 | 1.5979 | 4.83x |
| color.convert_colorspace | float32 | 146.0925 | 3.9548 | 36.94x |
| color.deltaE_ciede94 | float32 | 107.7068 | 20.3974 | 5.28x |
| color.deltaE_ciede2000 | float32 | 175.9507 | 94.3051 | 1.87x |
| color.deltaE_cmc | float32 | 171.9148 | 42.8791 | 4.01x |
| morphology.binary_dilation | float64 | 13.0803 | 4.4997 | 2.91x |
| color.rgb2ycbcr | float64 | 16.0950 | 6.4222 | 2.51x |
| color.rgb2lab | float64 | 169.5630 | 50.5083 | 3.36x |
| color.lab2lch | float64 | 32.7503 | 13.9178 | 2.35x |
| color.luv2xyz | float64 | 82.9645 | 6.6850 | 12.41x |
| color.combine_stains | float64 | 39.4918 | 11.5901 | 3.41x |
| color.deltaE_cie76 | float64 | 23.0193 | 4.1379 | 5.56x |

Boolean 31×31 rectangles, with one worker (the previous slower morphology workloads):

| Operation | Image | Original ms | C++ ms | Reference / C++ |
|---|---|---:|---:|---:|
| morphology.erosion | 512² | 2.9033 | 0.1587 | 18.30x |
| morphology.dilation | 512² | 2.6311 | 0.1751 | 15.03x |
| morphology.opening | 512² | 4.4779 | 0.2075 | 21.58x |
| morphology.closing | 512² | 4.1523 | 0.2756 | 15.07x |
| morphology.white_tophat | 512² | 4.1455 | 0.2590 | 16.01x |
| morphology.black_tophat | 512² | 3.6571 | 0.2575 | 14.20x |

The [fixed-workload tuning experiment](morphology-tuning.md) preserves separate historical before/after C++ snapshots. Its cases are not added to this current-source total.

1034 of 1057 one-worker native processing workloads measured faster than the reference. These are workload cases, including repeated functions and parameter variants. Identity returns, copies and dtype metadata are excluded from this count and labeled separately in the full reports.

23 one-worker processing workloads remained slower:

| Operation | Input | Shape | Layout | Reference / C++ |
|---|---|---:|---|---:|
| color.lch2lab | float32 | 256² | standard | 0.9421x |
| util.invert | float64 | 256² | standard | 0.9535x |
| util.img_as_bool | float64 | 256² | standard | 0.9604x |
| util.invert | float32 | 1024² | standard | 0.9741x |
| util.img_as_float64 | float32 | 1024² | standard | 0.9373x |
| util.img_as_float32 | float64 | 1024² | standard | 0.9871x |
| util.img_as_float64 | bool | 1024² | standard | 0.9037x |
| util.img_as_int | uint16 | 1024² | standard | 0.6968x |
| util.img_as_bool | uint16 | 1024² | standard | 0.9564x |
| util.img_as_int | uint64 | 1024² | standard | 0.9695x |
| util.img_as_bool | int64 | 1024² | standard | 0.9887x |
| color.separate_stains | uint16 | 512² | strided | 0.8379x |
| color.rgb2hed | uint16 | 512² | strided | 0.9524x |
| color.separate_stains | uint16 | 512² | channel-first | 0.8931x |
| color.rgb2hed | uint16 | 512² | channel-first | 0.8145x |
| morphology.black_tophat | uint8 | 512² | rectangle-31 | 0.9871x |
| morphology.black_tophat | float32 | 512² | rectangle-31 | 0.9909x |
| filters.threshold_yen | float32/int64 | 257 | histogram-only | 0.9875x |
| filters.threshold_yen | float32/int64 | 257 | histogram-only | 0.9981x |
| filters.threshold_yen | float32/int64 | 257 | histogram-only | 0.9949x |
| filters.threshold_isodata | float32/float32 | 257 | histogram-only | 0.9905x |
| filters.threshold_yen | float32/float32 | 257 | histogram-only | 0.9618x |
| filters.threshold_yen | float32/float64 | 257 | histogram-only | 0.9654x |

The color port fuses integer normalization, matrix arithmetic and sRGB transfer. Uniform floating layouts now avoid a full contiguous copy: planar input uses contiguous channel loads, other uniform strides use gathers, and scalar tails preserve the same arithmetic. General nonuniform layouts and unaligned arrays can still require copying. Simple matrix transforms remain sensitive to dispatch, shuffling and memory traffic; this run has no hardware-counter profiling to quantify their individual contribution. Further optimization remains necessary.

Morphology reuses row mappings across border pixels and stops binary reductions when the result is known for every SIMD lane. Large rectangular footprints use linear-work prefix/suffix scans; small input types use narrower working buffers when conversion preserves extrema, while 64-bit integer scans retain double buffers for upstream rounding. Contiguous scan outputs and top-hat subtraction use SIMD. Canonical 2D boolean rectangles now use packed words, with SIMD packing, bitwise extrema and unpacking, avoiding the previous byte-wise scan passes. Other storage and boundary constants retain the general native path.

Luv conversions fuse sRGB/XYZ stages with vector chromaticity arithmetic. Cached three-value white metadata preserves NumPy dot rounding at singular denominators. Float32 inverse cube powers retain libm rounding because sRGB amplifies small XYZ differences near black; surrounding arithmetic remains vectorized. Cross-platform real-color stress tests separately bound the one-ULP luminance difference and its propagation through XYZ/sRGB; see the numerical contract.

Uniformly linear SIMD blocks skip unused sRGB power and CIE cube-root stages. The [fixed linear-color experiment](linear-color-tuning.md) retains historical before/after timings separately from this aggregate. Stain kernels fuse input normalization, logarithmic scaling, matrix arithmetic and reconstruction. Uint8 separation uses a log lookup table; floating layouts use direct loads or gathers. Float32 log values retain float32 precision before NumPy-compatible promotion, so output float64 does not imply a float64 input error budget.

CIE76 fuses raw Lab conversion, subtraction, squared components and square root without temporary image arrays. Uniform floating layouts use direct loads or gathers; integer/mixed blocks convert raw values before vector arithmetic. Aligned nonuniform CIE76 views now use per-row offsets and signed strides instead of full image copies. Weighted Lab metrics preserve dtype-specific rounding, double hue cancellation and promoted trigonometric stages. Float32 CIEDE2000 uses the documented hue-roundoff budget, while float64 retains strict comparisons. Color-space routing preserves the upstream intermediate RGB allocation and dtype rounding. Its parameter report includes all stage checks and propagated bounds for ill-conditioned HSV ratios.

The [fixed CIE76 integer-load experiment](distance-tuning.md) preserves historical before/after snapshots separately from these current-source totals.

The [fixed nonuniform-layout experiment](distance-layout-tuning.md) records the separate one-worker copy-removal comparison, including all raw samples.

Yen, Triangle and ISODATA use native histograms and selectors. Floating bin edges and centers are small NumPy metadata arrays, while image scans, bin assignment, accumulation and threshold selection run in C++. AVX2 computes floating bin indices; exact edge corrections preserve histogram parity. Yen retains the installed NumPy float32 log rounding because near ties can change the selected bin. Supplied-histogram cases measure selection without an image scan; degenerate warning and error contracts are parity-tested.

Measurements use ordinary system load without CPU affinity or frequency locking. Small differences near 1x can change between runs. These are measured comparisons on this machine, not universal speed guarantees. CI runs parity and benchmark smoke checks on Windows/Linux without asserting speed ratios on shared runners.

Coverage remains partial: 77 native image entries, 2 metadata helpers, 371 inventoried entries still pending. See the [inventory](../coverage/README.md), [roadmap](../coverage/ROADMAP.md) and [documented input contracts](../README.md). Run `python benchmarks/verify.py` to rebuild and reproduce all nine reports, including this summary. Use `--threads 1` for serial-only verification. The summary rejects mixed source hashes, missing APIs and absent one-worker comparisons.
