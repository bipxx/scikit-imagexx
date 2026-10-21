# Current performance verification

Reference: original scikit-image 0.26.0. 1838 comparisons on Windows-11-10.0.26200-SP0; CPU: Intel64 Family 6 Model 151 Stepping 2, GenuineIntel. Individual reports record timestamps, CPU, versions, raw samples and options. All reports match the current source/build SHA-256 (POSIX relative paths and LF line endings):
`69053ae1b589e1a2097c4b1350f430d40b182279b7e0a53dc821eedda2f60abc`.

| Report | Cases | Worker counts |
|---|---:|---|
| [Every covered public entry](function-coverage.md) | 880 | 1, 4 |
| [Additional integer widths](conversion-widths.md) | 112 | 1, 4 |
| [Non-default exposure parameters](exposure-parameters.md) | 24 | 1 |
| [Color layouts and uint16 inputs](color-layouts.md) | 120 | 1 |
| [Perceptual color parameters](perceptual-parameters.md) | 60 | 1, 4 |
| [Morphology footprints and output reuse](morphology-parameters.md) | 240 | 1, 4 |
| [Stain matrices and output widths](stain-parameters.md) | 156 | 1, 4 |
| [Lab distances and color routes](color-distance.md) | 246 | 1, 4 |

Every workload checks values and dtype against original scikit-image before timing with fallback disabled. Integer-width conversion cases also require exact values and matching identity behavior. Timings include allocation, Python dispatch, integer normalization and any needed input copying. Inputs/imports/I/O are excluded. Reference thread pools use one thread. Native worker counts are explicit.

Examples from the 1024² one-worker default-parameter comparisons (color-space routing is RGB to HSV; CIE76 compares raw Lab values with a channel-reversed second input):

| Operation | Input | Original ms | C++ ms | Reference / C++ |
|---|---|---:|---:|---:|
| morphology.erosion | uint8 | 10.6430 | 0.4928 | 21.60x |
| color.rgb2yuv | uint8 | 15.2794 | 7.9895 | 1.91x |
| color.rgb2xyz | uint8 | 85.2699 | 8.6785 | 9.83x |
| color.rgb2hed | uint8 | 47.2555 | 12.7162 | 3.72x |
| filters.gaussian | uint8 | 21.4467 | 5.4893 | 3.91x |
| filters.sobel | uint8 | 25.4517 | 3.4644 | 7.35x |
| filters.threshold_otsu | uint8 | 3.6485 | 0.2488 | 14.66x |
| morphology.opening | float32 | 9.1775 | 3.9644 | 2.31x |
| color.rgb2xyz | float32 | 52.9782 | 20.8150 | 2.55x |
| color.xyz2rgb | float32 | 69.6328 | 21.3667 | 3.26x |
| color.rgb2hsv | float32 | 137.3886 | 4.0601 | 33.84x |
| color.hsv2rgb | float32 | 110.4901 | 4.3286 | 25.53x |
| color.rgb2luv | float32 | 94.2153 | 32.2912 | 2.92x |
| color.luv2rgb | float32 | 44.7119 | 4.4510 | 10.05x |
| color.separate_stains | float32 | 32.3182 | 13.8111 | 2.34x |
| color.deltaE_cie76 | float32 | 7.3960 | 1.4412 | 5.13x |
| color.convert_colorspace | float32 | 136.0464 | 3.7270 | 36.50x |
| color.deltaE_ciede94 | float32 | 108.8246 | 19.8912 | 5.47x |
| color.deltaE_ciede2000 | float32 | 194.0295 | 93.7131 | 2.07x |
| color.deltaE_cmc | float32 | 180.6946 | 41.9973 | 4.30x |
| morphology.binary_dilation | float64 | 11.9384 | 4.5778 | 2.61x |
| color.rgb2ycbcr | float64 | 16.1969 | 6.2138 | 2.61x |
| color.rgb2lab | float64 | 163.1489 | 50.0198 | 3.26x |
| color.lab2lch | float64 | 31.4782 | 13.5508 | 2.32x |
| color.luv2xyz | float64 | 92.0893 | 6.8350 | 13.47x |
| color.combine_stains | float64 | 38.5724 | 11.6154 | 3.32x |
| color.deltaE_cie76 | float64 | 24.9215 | 4.3902 | 5.68x |

Boolean 31×31 rectangles, with one worker (the previous slower morphology workloads):

| Operation | Image | Original ms | C++ ms | Reference / C++ |
|---|---|---:|---:|---:|
| morphology.erosion | 512² | 2.2644 | 0.1655 | 13.69x |
| morphology.dilation | 512² | 3.5013 | 0.2637 | 13.28x |
| morphology.opening | 512² | 5.6602 | 0.3908 | 14.48x |
| morphology.closing | 512² | 5.2640 | 0.4008 | 13.13x |
| morphology.white_tophat | 512² | 5.9815 | 0.3791 | 15.78x |
| morphology.black_tophat | 512² | 5.8198 | 0.3653 | 15.93x |

The [fixed-workload tuning experiment](morphology-tuning.md) preserves separate historical before/after C++ snapshots. Its cases are not added to this current-source total.

951 of 963 one-worker native processing workloads measured faster than the reference. These are workload cases, including repeated functions and parameter variants. Identity returns, copies and dtype metadata are excluded from this count and labeled separately in the full reports.

12 one-worker processing workloads remained slower:

| Operation | Input | Size | Layout | Reference / C++ |
|---|---|---:|---|---:|
| color.lch2lab | float32 | 256² | standard | 0.9097x |
| util.img_as_bool | float64 | 256² | standard | 0.7659x |
| util.invert | float32 | 1024² | standard | 0.7852x |
| util.img_as_float64 | float32 | 1024² | standard | 0.9134x |
| util.img_as_bool | float32 | 1024² | standard | 0.8754x |
| util.img_as_bool | float64 | 1024² | standard | 0.8090x |
| util.img_as_float64 | bool | 1024² | standard | 0.9134x |
| util.img_as_bool | int8 | 1024² | standard | 0.9963x |
| util.img_as_int | uint16 | 1024² | standard | 0.7034x |
| util.img_as_bool | uint16 | 1024² | standard | 0.7786x |
| util.img_as_bool | uint32 | 1024² | standard | 0.9687x |
| util.img_as_bool | uint64 | 1024² | standard | 0.9785x |

The color port fuses integer normalization, matrix arithmetic and sRGB transfer. Uniform floating layouts now avoid a full contiguous copy: planar input uses contiguous channel loads, other uniform strides use gathers, and scalar tails preserve the same arithmetic. General nonuniform layouts and unaligned arrays can still require copying. Simple matrix transforms remain sensitive to dispatch, shuffling and memory traffic; this run has no hardware-counter profiling to quantify their individual contribution. Further optimization remains necessary.

Morphology reuses row mappings across border pixels and stops binary reductions when the result is known for every SIMD lane. Large rectangular footprints use linear-work prefix/suffix scans; small input types use narrower working buffers when conversion preserves extrema, while 64-bit integer scans retain double buffers for upstream rounding. Contiguous scan outputs and top-hat subtraction use SIMD. Canonical 2D boolean rectangles now use packed words, with SIMD packing, bitwise extrema and unpacking, avoiding the previous byte-wise scan passes. Other storage and boundary constants retain the general native path.

Luv conversions fuse sRGB/XYZ stages with vector chromaticity arithmetic. Cached three-value white metadata preserves NumPy dot rounding at singular denominators. Float32 inverse cube powers retain libm rounding because sRGB amplifies small XYZ differences near black; surrounding arithmetic remains vectorized. Cross-platform real-color stress tests separately bound the one-ULP luminance difference and its propagation through XYZ/sRGB; see the numerical contract.

Uniformly linear SIMD blocks skip unused sRGB power and CIE cube-root stages. The [fixed linear-color experiment](linear-color-tuning.md) retains historical before/after timings separately from this aggregate. Stain kernels fuse input normalization, logarithmic scaling, matrix arithmetic and reconstruction. Uint8 separation uses a log lookup table; floating layouts use direct loads or gathers. Float32 log values retain float32 precision before NumPy-compatible promotion, so output float64 does not imply a float64 input error budget.

CIE76 fuses raw Lab conversion, subtraction, squared components and square root without temporary image arrays. Uniform floating layouts use direct loads or gathers; integer/mixed blocks convert raw values before vector arithmetic. Aligned nonuniform CIE76 views now use per-row offsets and signed strides instead of full image copies. Weighted Lab metrics preserve dtype-specific rounding, double hue cancellation and promoted trigonometric stages. Float32 CIEDE2000 uses the documented hue-roundoff budget, while float64 retains strict comparisons. Color-space routing preserves the upstream intermediate RGB allocation and dtype rounding. Its parameter report includes all stage checks and propagated bounds for ill-conditioned HSV ratios.

The [fixed CIE76 integer-load experiment](distance-tuning.md) preserves historical before/after snapshots separately from these current-source totals.

The [fixed nonuniform-layout experiment](distance-layout-tuning.md) records the separate one-worker copy-removal comparison, including all raw samples.

Measurements use ordinary system load without CPU affinity or frequency locking. Small differences near 1x can change between runs. These are measured comparisons on this machine, not universal speed guarantees. CI runs parity and benchmark smoke checks on Windows/Linux without asserting speed ratios on shared runners.

Coverage remains partial: 74 native image entries, 2 metadata helpers, 374 inventoried entries still pending. See the [inventory](../coverage/README.md), [roadmap](../coverage/ROADMAP.md) and [documented input contracts](../README.md). Run `python benchmarks/verify.py` to rebuild and reproduce all eight reports, including this summary. Use `--threads 1` for serial-only verification. The summary rejects mixed source hashes, missing APIs and absent one-worker comparisons.
