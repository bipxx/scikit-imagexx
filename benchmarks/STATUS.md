# Current performance verification

Reference: original scikit-image 0.26.0. 1966 comparisons on Windows-11-10.0.26200-SP0; CPU: Intel64 Family 6 Model 151 Stepping 2, GenuineIntel. Individual reports record timestamps, CPU, versions, raw samples and options. All reports match the current source/build SHA-256 (POSIX relative paths and LF line endings):
`886c6d1439c2338af8d2407dee40287fb01e8cfa3b6dff95a80ea2a5b5686343`.

| Report | Cases | Worker counts |
|---|---:|---|
| [Every covered public entry](function-coverage.md) | 524 | 1 |
| [Additional integer widths](conversion-widths.md) | 56 | 1 |
| [Non-default exposure parameters](exposure-parameters.md) | 24 | 1 |
| [Color layouts and uint16 inputs](color-layouts.md) | 120 | 1 |
| [Perceptual color parameters](perceptual-parameters.md) | 30 | 1 |
| [Morphology footprints and output reuse](morphology-parameters.md) | 120 | 1 |
| [Stain matrices and output widths](stain-parameters.md) | 78 | 1 |
| [Lab distances and color routes](color-distance.md) | 123 | 1 |
| [Histogram threshold parameters](threshold-parameters.md) | 76 | 1 |
| [Local threshold windows and dtypes](local-threshold-parameters.md) | 128 | 1 |
| [Alpha and grayscale layouts](alpha-parameters.md) | 184 | 1 |
| [Laplace and Roberts layouts](edge-parameters.md) | 260 | 1 |
| [Farid derivatives and boundaries](farid-parameters.md) | 243 | 1 |

Every workload checks values and dtype against original scikit-image before timing with fallback disabled. Integer-width conversion cases also require exact values and matching identity behavior. Timings include allocation, Python dispatch, integer normalization and any needed input copying. Inputs/imports/I/O are excluded. Reference thread pools use one thread. Native worker counts are explicit.

Examples from the 1024² one-worker default-parameter comparisons (color-space routing is RGB to HSV; CIE76 compares raw Lab values with a channel-reversed second input):

| Operation | Input | Original ms | C++ ms | Reference / C++ |
|---|---|---:|---:|---:|
| morphology.erosion | uint8 | 5.7999 | 0.4319 | 13.43x |
| color.rgb2yuv | uint8 | 17.1529 | 8.4563 | 2.03x |
| color.rgb2xyz | uint8 | 90.3854 | 8.9799 | 10.07x |
| color.rgb2hed | uint8 | 40.8430 | 13.2706 | 3.08x |
| filters.gaussian | uint8 | 24.3341 | 5.6243 | 4.33x |
| filters.sobel | uint8 | 26.7275 | 3.6608 | 7.30x |
| filters.threshold_otsu | uint8 | 4.3096 | 0.2836 | 15.20x |
| morphology.opening | float32 | 17.1364 | 2.9269 | 5.85x |
| color.rgb2xyz | float32 | 82.2650 | 25.4686 | 3.23x |
| color.xyz2rgb | float32 | 102.1850 | 32.4437 | 3.15x |
| color.rgb2hsv | float32 | 176.9402 | 6.0140 | 29.42x |
| color.hsv2rgb | float32 | 163.4117 | 5.1028 | 32.02x |
| color.rgb2luv | float32 | 103.7191 | 32.1100 | 3.23x |
| color.luv2rgb | float32 | 61.8089 | 4.9198 | 12.56x |
| color.separate_stains | float32 | 41.9423 | 14.0314 | 2.99x |
| color.deltaE_cie76 | float32 | 13.2676 | 2.2361 | 5.93x |
| color.convert_colorspace | float32 | 204.5513 | 5.8496 | 34.97x |
| color.deltaE_ciede94 | float32 | 163.3479 | 25.8974 | 6.31x |
| color.deltaE_ciede2000 | float32 | 289.5509 | 113.1874 | 2.56x |
| color.deltaE_cmc | float32 | 281.7622 | 53.3987 | 5.28x |
| morphology.binary_dilation | float64 | 13.6562 | 5.3657 | 2.55x |
| color.rgb2ycbcr | float64 | 25.5165 | 12.9367 | 1.97x |
| color.rgb2lab | float64 | 276.1380 | 64.0547 | 4.31x |
| color.lab2lch | float64 | 51.5875 | 15.9405 | 3.24x |
| color.luv2xyz | float64 | 162.5920 | 14.2047 | 11.45x |
| color.combine_stains | float64 | 83.8262 | 22.4518 | 3.73x |
| color.deltaE_cie76 | float64 | 34.7907 | 6.2909 | 5.53x |

Boolean 31×31 rectangles, with one worker (the previous slower morphology workloads):

| Operation | Image | Original ms | C++ ms | Reference / C++ |
|---|---|---:|---:|---:|
| morphology.erosion | 512² | 2.2425 | 0.1477 | 15.19x |
| morphology.dilation | 512² | 2.2461 | 0.1569 | 14.32x |
| morphology.opening | 512² | 4.3288 | 0.3281 | 13.19x |
| morphology.closing | 512² | 3.3834 | 0.2314 | 14.62x |
| morphology.white_tophat | 512² | 3.7687 | 0.2587 | 14.57x |
| morphology.black_tophat | 512² | 3.3174 | 0.2251 | 14.74x |

The [fixed-workload tuning experiment](morphology-tuning.md) preserves separate historical before/after C++ snapshots. Its cases are not added to this current-source total.

1919 of 1938 one-worker native processing workloads measured faster than the reference. These are workload cases, including repeated functions and parameter variants. Identity returns, copies and dtype metadata are excluded from this count and labeled separately in the full reports.

19 one-worker processing workloads remained slower:

| Operation | Input | Shape | Layout | Workload | Reference / C++ |
|---|---|---:|---|---|---:|
| util.img_as_bool | float64 | 256² | standard | default | 0.7658x |
| util.invert | float32 | 1024² | standard | default | 0.7891x |
| util.img_as_bool | float32 | 1024² | standard | default | 0.6549x |
| util.invert | float64 | 1024² | standard | default | 0.9741x |
| util.img_as_bool | float64 | 1024² | standard | default | 0.9037x |
| util.img_as_bool | int8 | 1024² | standard | default | 0.9825x |
| util.img_as_int | uint16 | 1024² | standard | default | 0.7402x |
| color.separate_stains | uint16 | 512² | channel-first | default | 0.8499x |
| color.rgb2hed | uint16 | 512² | channel-first | default | 0.9283x |
| color.lch2lab | float32 | 512² | standard | four-channels | 0.9865x |
| filters.threshold_yen | float32/int64 | 257 | histogram-only | supplied-histogram | 0.6827x |
| filters.threshold_isodata | float32/int64 | 257 | histogram-only | trimmed-histogram | 0.8209x |
| filters.threshold_yen | float32/float32 | 257 | histogram-only | supplied-histogram | 0.5361x |
| filters.threshold_yen | float32/float32 | 257 | histogram-only | trimmed-histogram | 0.8229x |
| filters.threshold_isodata | float32/float32 | 257 | histogram-only | large-counts | 0.9698x |
| filters.threshold_yen | float32/float64 | 257 | histogram-only | trimmed-histogram | 0.8004x |
| color.gray2rgb | float16 | 512×512 | strided | strided | 0.8759x |
| color.gray2rgb | float16 | 512×512 | fortran | fortran | 0.7667x |
| color.gray2rgba | >f4 | 512×512 | contiguous | endian-contiguous | 0.9900x |

The color port fuses integer normalization, matrix arithmetic and sRGB transfer. Uniform floating layouts now avoid a full contiguous copy: planar input uses contiguous channel loads, other uniform strides use gathers, and scalar tails preserve the same arithmetic. General nonuniform layouts and unaligned arrays can still require copying. Simple matrix transforms remain sensitive to dispatch, shuffling and memory traffic; this run has no hardware-counter profiling to quantify their individual contribution. Further optimization remains necessary.

Morphology reuses row mappings across border pixels and stops binary reductions when the result is known for every SIMD lane. Large rectangular footprints use linear-work prefix/suffix scans; small input types use narrower working buffers when conversion preserves extrema, while 64-bit integer scans retain double buffers for upstream rounding. Contiguous scan outputs and top-hat subtraction use SIMD. Canonical 2D boolean rectangles now use packed words, with SIMD packing, bitwise extrema and unpacking, avoiding the previous byte-wise scan passes. Other storage and boundary constants retain the general native path.

Luv conversions fuse sRGB/XYZ stages with vector chromaticity arithmetic. Cached three-value white metadata preserves NumPy dot rounding at singular denominators. Float32 inverse cube powers retain libm rounding because sRGB amplifies small XYZ differences near black; surrounding arithmetic remains vectorized. Cross-platform real-color stress tests separately bound the one-ULP luminance difference and its propagation through XYZ/sRGB; see the numerical contract.

Uniformly linear SIMD blocks skip unused sRGB power and CIE cube-root stages. The [fixed linear-color experiment](linear-color-tuning.md) retains historical before/after timings separately from this aggregate. Stain kernels fuse input normalization, logarithmic scaling, matrix arithmetic and reconstruction. Uint8 separation uses a log lookup table; floating layouts use direct loads or gathers. Float32 log values retain float32 precision before NumPy-compatible promotion, so output float64 does not imply a float64 input error budget.

CIE76 fuses raw Lab conversion, subtraction, squared components and square root without temporary image arrays. Uniform floating layouts use direct loads or gathers; integer/mixed blocks convert raw values before vector arithmetic. Aligned nonuniform CIE76 views now use per-row offsets and signed strides instead of full image copies. Weighted Lab metrics preserve dtype-specific rounding, double hue cancellation and promoted trigonometric stages. Float32 CIEDE2000 uses the documented hue-roundoff budget, while float64 retains strict comparisons. Color-space routing preserves the upstream intermediate RGB allocation and dtype rounding. Its parameter report includes all stage checks and propagated bounds for ill-conditioned HSV ratios.

The [fixed CIE76 integer-load experiment](distance-tuning.md) preserves historical before/after snapshots separately from these current-source totals.

The [fixed nonuniform-layout experiment](distance-layout-tuning.md) records the separate one-worker copy-removal comparison, including all raw samples.

Yen, Triangle and ISODATA use native histograms and selectors. Floating bin edges and centers are small NumPy metadata arrays, while image scans, bin assignment, accumulation and threshold selection run in C++. AVX2 computes floating bin indices; exact edge corrections preserve histogram parity. Yen retains the installed NumPy float32 log rounding because near ties can change the selected bin. Supplied-histogram cases measure selection without an image scan; degenerate warning and error contracts are parity-tested.

Niblack and Sauvola share reflected input preparation, ordered float64 integral sums and squared sums. Independent prefix columns and output pixels use AVX2 while preserving each cumulative sequence. Float32 window sums round before division; strong window/weight/range metadata retains upstream promotion. The parameter report includes all eleven input types, noncontiguous layouts, large/anisotropic windows and singleton-depth N-D volumes.

RGBA compositing fuses integer normalization, blending and clipping with strict SIMD arithmetic. Grayscale RGB/RGBA expansion replicates raw storage and normalizes endian components directly. Both use physical stride order and partition pixels across the global worker limit; the alpha report includes planar, Fortran, negative and stepped layouts plus background/alpha options. The [fixed grayscale expansion experiment](alpha-tuning.md) retains initial, optimized and confirmation observations separately from this aggregate.

Laplace uses sparse neighbor loads in scipy coefficient order, retaining singleton-axis terms. Roberts fuses cross differences and rounded magnitude stages. Half and endian conversion, mask erosion/broadcast multiplication, layout preparation and allocation are included in the edge report. The [fixed column experiment](edge-tuning.md) retains separate baseline, optimized and confirmation snapshots. The [metadata experiment](edge-metadata-tuning.md) retains the separate validation-overhead comparison.

Farid preserves the ordered full coefficient sum while vectorizing independent output pixels. Float32 gradients round before each magnitude stage, including the dimension-typed divisor. Axes, boundaries, endian normalization, masks and input movement are included in its report. Empty axes use owned zero allocation and omit unused image copies; the [fixed empty-axis experiment](farid-empty-tuning.md) retains separate historical observations.

Measurements use ordinary system load without CPU affinity or frequency locking. Small differences near 1x can change between runs. These are measured comparisons on this machine, not universal speed guarantees. CI runs parity and benchmark smoke checks on Windows/Linux without asserting speed ratios on shared runners.

Coverage remains partial: 88 native image entries, 2 metadata helpers, 360 inventoried entries still pending. See the [inventory](../coverage/README.md), [roadmap](../coverage/ROADMAP.md) and [documented input contracts](../README.md). Run `python benchmarks/verify.py` to rebuild and reproduce all thirteen reports, including this summary. Use `--threads 1` for serial-only verification. The summary rejects mixed source hashes, missing APIs and absent one-worker comparisons.
