# Current performance verification

Reference: original scikit-image 0.26.0. 2306 comparisons on Windows-11-10.0.26200-SP0; CPU: Intel64 Family 6 Model 151 Stepping 2, GenuineIntel. Individual reports record timestamps, CPU, versions, raw samples and options. All reports match the current source/build SHA-256 (POSIX relative paths and LF line endings):
`92361a4dad2278c59ca9aa37320eb5e947b8b4cc6fd065de0a8a2077f8c8a9e8`.

| Report | Cases | Worker counts |
|---|---:|---|
| [Every covered public entry](function-coverage.md) | 940 | 1, 4 |
| [Additional integer widths](conversion-widths.md) | 112 | 1, 4 |
| [Non-default exposure parameters](exposure-parameters.md) | 24 | 1 |
| [Color layouts and uint16 inputs](color-layouts.md) | 120 | 1 |
| [Perceptual color parameters](perceptual-parameters.md) | 60 | 1, 4 |
| [Morphology footprints and output reuse](morphology-parameters.md) | 240 | 1, 4 |
| [Stain matrices and output widths](stain-parameters.md) | 156 | 1, 4 |
| [Lab distances and color routes](color-distance.md) | 246 | 1, 4 |
| [Histogram threshold parameters](threshold-parameters.md) | 152 | 1, 4 |
| [Local threshold windows and dtypes](local-threshold-parameters.md) | 256 | 1, 4 |

Every workload checks values and dtype against original scikit-image before timing with fallback disabled. Integer-width conversion cases also require exact values and matching identity behavior. Timings include allocation, Python dispatch, integer normalization and any needed input copying. Inputs/imports/I/O are excluded. Reference thread pools use one thread. Native worker counts are explicit.

Examples from the 1024² one-worker default-parameter comparisons (color-space routing is RGB to HSV; CIE76 compares raw Lab values with a channel-reversed second input):

| Operation | Input | Original ms | C++ ms | Reference / C++ |
|---|---|---:|---:|---:|
| morphology.erosion | uint8 | 10.4694 | 0.4123 | 25.40x |
| color.rgb2yuv | uint8 | 16.3391 | 8.6460 | 1.89x |
| color.rgb2xyz | uint8 | 85.4840 | 8.1586 | 10.48x |
| color.rgb2hed | uint8 | 40.0608 | 14.1888 | 2.82x |
| filters.gaussian | uint8 | 26.6318 | 7.6778 | 3.47x |
| filters.sobel | uint8 | 42.3718 | 7.3251 | 5.78x |
| filters.threshold_otsu | uint8 | 6.0468 | 0.3563 | 16.97x |
| morphology.opening | float32 | 19.9963 | 5.3267 | 3.75x |
| color.rgb2xyz | float32 | 76.5051 | 39.3045 | 1.95x |
| color.xyz2rgb | float32 | 99.7964 | 36.5962 | 2.73x |
| color.rgb2hsv | float32 | 195.3376 | 6.0737 | 32.16x |
| color.hsv2rgb | float32 | 170.2359 | 6.1017 | 27.90x |
| color.rgb2luv | float32 | 143.6439 | 52.3922 | 2.74x |
| color.luv2rgb | float32 | 71.7290 | 8.7236 | 8.22x |
| color.separate_stains | float32 | 44.1871 | 26.6223 | 1.66x |
| color.deltaE_cie76 | float32 | 10.4008 | 3.8016 | 2.74x |
| color.convert_colorspace | float32 | 219.2509 | 6.7121 | 32.67x |
| color.deltaE_ciede94 | float32 | 151.1506 | 31.2634 | 4.83x |
| color.deltaE_ciede2000 | float32 | 279.3466 | 134.2935 | 2.08x |
| color.deltaE_cmc | float32 | 259.5670 | 60.1270 | 4.32x |
| morphology.binary_dilation | float64 | 16.5691 | 6.2378 | 2.66x |
| color.rgb2ycbcr | float64 | 24.2207 | 9.4306 | 2.57x |
| color.rgb2lab | float64 | 231.9389 | 59.8401 | 3.88x |
| color.lab2lch | float64 | 41.7385 | 14.7277 | 2.83x |
| color.luv2xyz | float64 | 136.5755 | 11.1320 | 12.27x |
| color.combine_stains | float64 | 51.4063 | 12.2601 | 4.19x |
| color.deltaE_cie76 | float64 | 30.5585 | 4.6776 | 6.53x |

Boolean 31×31 rectangles, with one worker (the previous slower morphology workloads):

| Operation | Image | Original ms | C++ ms | Reference / C++ |
|---|---|---:|---:|---:|
| morphology.erosion | 512² | 2.1581 | 0.1422 | 15.17x |
| morphology.dilation | 512² | 2.1593 | 0.1516 | 14.24x |
| morphology.opening | 512² | 3.3912 | 0.2063 | 16.44x |
| morphology.closing | 512² | 3.4769 | 0.2083 | 16.69x |
| morphology.white_tophat | 512² | 3.4243 | 0.2257 | 15.17x |
| morphology.black_tophat | 512² | 3.2551 | 0.2352 | 13.84x |

The [fixed-workload tuning experiment](morphology-tuning.md) preserves separate historical before/after C++ snapshots. Its cases are not added to this current-source total.

1172 of 1197 one-worker native processing workloads measured faster than the reference. These are workload cases, including repeated functions and parameter variants. Identity returns, copies and dtype metadata are excluded from this count and labeled separately in the full reports.

25 one-worker processing workloads remained slower:

| Operation | Input | Shape | Layout | Workload | Reference / C++ |
|---|---|---:|---|---|---:|
| color.rgb2yuv | uint8 | 256² | standard | default | 0.8656x |
| color.rgb2ypbpr | uint8 | 256² | standard | default | 0.9195x |
| color.rgb2ydbdr | uint8 | 256² | standard | default | 0.9130x |
| util.img_as_bool | uint8 | 256² | standard | default | 0.8758x |
| color.rgb2gray | float64 | 256² | standard | default | 0.9115x |
| util.img_as_bool | float64 | 256² | standard | default | 0.8574x |
| util.img_as_bool | float32 | 1024² | standard | default | 0.8229x |
| util.img_as_float64 | bool | 1024² | standard | default | 0.9806x |
| util.img_as_int | uint16 | 1024² | standard | default | 0.9057x |
| util.img_as_bool | int16 | 1024² | standard | default | 0.8574x |
| util.img_as_bool | uint64 | 1024² | standard | default | 0.9812x |
| util.img_as_bool | int64 | 1024² | standard | default | 0.8300x |
| color.separate_stains | uint16 | 512² | strided | default | 0.8673x |
| color.rgb2hed | uint16 | 512² | strided | default | 0.8422x |
| color.separate_stains | uint16 | 512² | channel-first | default | 0.8250x |
| color.rgb2hed | uint16 | 512² | channel-first | default | 0.8609x |
| morphology.white_tophat | float32 | 512² | rectangle-31 | rectangle-31 | 0.9165x |
| filters.threshold_yen | float32/int64 | 257 | histogram-only | supplied-histogram | 0.6911x |
| filters.threshold_yen | float32/int64 | 257 | histogram-only | large-counts | 0.8173x |
| filters.threshold_isodata | float32/int64 | 257 | histogram-only | large-counts | 0.9304x |
| filters.threshold_yen | float32/float32 | 257 | histogram-only | supplied-histogram | 0.9636x |
| filters.threshold_isodata | float32/float32 | 257 | histogram-only | supplied-histogram | 0.7725x |
| filters.threshold_isodata | float32/float32 | 257 | histogram-only | trimmed-histogram | 0.8084x |
| filters.threshold_yen | float32/float32 | 257 | histogram-only | large-counts | 0.6961x |
| filters.threshold_isodata | float32/float32 | 257 | histogram-only | large-counts | 0.9011x |

The color port fuses integer normalization, matrix arithmetic and sRGB transfer. Uniform floating layouts now avoid a full contiguous copy: planar input uses contiguous channel loads, other uniform strides use gathers, and scalar tails preserve the same arithmetic. General nonuniform layouts and unaligned arrays can still require copying. Simple matrix transforms remain sensitive to dispatch, shuffling and memory traffic; this run has no hardware-counter profiling to quantify their individual contribution. Further optimization remains necessary.

Morphology reuses row mappings across border pixels and stops binary reductions when the result is known for every SIMD lane. Large rectangular footprints use linear-work prefix/suffix scans; small input types use narrower working buffers when conversion preserves extrema, while 64-bit integer scans retain double buffers for upstream rounding. Contiguous scan outputs and top-hat subtraction use SIMD. Canonical 2D boolean rectangles now use packed words, with SIMD packing, bitwise extrema and unpacking, avoiding the previous byte-wise scan passes. Other storage and boundary constants retain the general native path.

Luv conversions fuse sRGB/XYZ stages with vector chromaticity arithmetic. Cached three-value white metadata preserves NumPy dot rounding at singular denominators. Float32 inverse cube powers retain libm rounding because sRGB amplifies small XYZ differences near black; surrounding arithmetic remains vectorized. Cross-platform real-color stress tests separately bound the one-ULP luminance difference and its propagation through XYZ/sRGB; see the numerical contract.

Uniformly linear SIMD blocks skip unused sRGB power and CIE cube-root stages. The [fixed linear-color experiment](linear-color-tuning.md) retains historical before/after timings separately from this aggregate. Stain kernels fuse input normalization, logarithmic scaling, matrix arithmetic and reconstruction. Uint8 separation uses a log lookup table; floating layouts use direct loads or gathers. Float32 log values retain float32 precision before NumPy-compatible promotion, so output float64 does not imply a float64 input error budget.

CIE76 fuses raw Lab conversion, subtraction, squared components and square root without temporary image arrays. Uniform floating layouts use direct loads or gathers; integer/mixed blocks convert raw values before vector arithmetic. Aligned nonuniform CIE76 views now use per-row offsets and signed strides instead of full image copies. Weighted Lab metrics preserve dtype-specific rounding, double hue cancellation and promoted trigonometric stages. Float32 CIEDE2000 uses the documented hue-roundoff budget, while float64 retains strict comparisons. Color-space routing preserves the upstream intermediate RGB allocation and dtype rounding. Its parameter report includes all stage checks and propagated bounds for ill-conditioned HSV ratios.

The [fixed CIE76 integer-load experiment](distance-tuning.md) preserves historical before/after snapshots separately from these current-source totals.

The [fixed nonuniform-layout experiment](distance-layout-tuning.md) records the separate one-worker copy-removal comparison, including all raw samples.

Yen, Triangle and ISODATA use native histograms and selectors. Floating bin edges and centers are small NumPy metadata arrays, while image scans, bin assignment, accumulation and threshold selection run in C++. AVX2 computes floating bin indices; exact edge corrections preserve histogram parity. Yen retains the installed NumPy float32 log rounding because near ties can change the selected bin. Supplied-histogram cases measure selection without an image scan; degenerate warning and error contracts are parity-tested.

Niblack and Sauvola share reflected input preparation, ordered float64 integral sums and squared sums. Independent prefix columns and output pixels use AVX2 while preserving each cumulative sequence. Float32 window sums round before division; strong window/weight/range metadata retains upstream promotion. The parameter report includes all eleven input types, noncontiguous layouts, large/anisotropic windows and singleton-depth N-D volumes.

Measurements use ordinary system load without CPU affinity or frequency locking. Small differences near 1x can change between runs. These are measured comparisons on this machine, not universal speed guarantees. CI runs parity and benchmark smoke checks on Windows/Linux without asserting speed ratios on shared runners.

Coverage remains partial: 79 native image entries, 2 metadata helpers, 369 inventoried entries still pending. See the [inventory](../coverage/README.md), [roadmap](../coverage/ROADMAP.md) and [documented input contracts](../README.md). Run `python benchmarks/verify.py` to rebuild and reproduce all ten reports, including this summary. Use `--threads 1` for serial-only verification. The summary rejects mixed source hashes, missing APIs and absent one-worker comparisons.
