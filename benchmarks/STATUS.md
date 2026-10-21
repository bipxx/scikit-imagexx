# Current performance verification

Reference: original scikit-image 0.26.0. 1421 comparisons on Windows-11-10.0.26200-SP0; CPU: Intel64 Family 6 Model 151 Stepping 2, GenuineIntel. Individual reports record timestamps, CPU, versions, raw samples and options. All reports match the current source/build SHA-256 (POSIX relative paths and LF line endings):
`1ccc09d934abd18ecf5c05e6ef3736228c36462d8d7583c5340a3741347bf2a6`.

| Report | Cases | Worker counts |
|---|---:|---|
| [Every covered public entry](function-coverage.md) | 482 | 1 |
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

Every workload checks values and dtype against original scikit-image before timing with fallback disabled. Integer-width conversion cases also require exact values and matching identity behavior. Timings include allocation, Python dispatch, integer normalization and any needed input copying. Inputs/imports/I/O are excluded. Reference thread pools use one thread. Native worker counts are explicit.

Examples from the 1024² one-worker default-parameter comparisons (color-space routing is RGB to HSV; CIE76 compares raw Lab values with a channel-reversed second input):

| Operation | Input | Original ms | C++ ms | Reference / C++ |
|---|---|---:|---:|---:|
| morphology.erosion | uint8 | 10.3083 | 0.4747 | 21.71x |
| color.rgb2yuv | uint8 | 14.9315 | 8.5075 | 1.76x |
| color.rgb2xyz | uint8 | 88.2785 | 9.3626 | 9.43x |
| color.rgb2hed | uint8 | 43.6275 | 13.8162 | 3.16x |
| filters.gaussian | uint8 | 23.0988 | 5.6376 | 4.10x |
| filters.sobel | uint8 | 24.3975 | 3.4512 | 7.07x |
| filters.threshold_otsu | uint8 | 3.6657 | 0.2649 | 13.84x |
| morphology.opening | float32 | 9.1448 | 2.9090 | 3.14x |
| color.rgb2xyz | float32 | 76.9796 | 30.5659 | 2.52x |
| color.xyz2rgb | float32 | 94.5861 | 28.6596 | 3.30x |
| color.rgb2hsv | float32 | 148.6963 | 3.8553 | 38.57x |
| color.hsv2rgb | float32 | 118.6682 | 4.5081 | 26.32x |
| color.rgb2luv | float32 | 94.8766 | 32.6575 | 2.91x |
| color.luv2rgb | float32 | 50.8731 | 4.8593 | 10.47x |
| color.separate_stains | float32 | 29.2855 | 14.6014 | 2.01x |
| color.deltaE_cie76 | float32 | 6.5013 | 1.5063 | 4.32x |
| color.convert_colorspace | float32 | 140.9483 | 3.8112 | 36.98x |
| color.deltaE_ciede94 | float32 | 104.0965 | 20.2902 | 5.13x |
| color.deltaE_ciede2000 | float32 | 181.1153 | 95.8819 | 1.89x |
| color.deltaE_cmc | float32 | 176.6894 | 42.2972 | 4.18x |
| morphology.binary_dilation | float64 | 11.8734 | 4.5546 | 2.61x |
| color.rgb2ycbcr | float64 | 21.1091 | 8.9447 | 2.36x |
| color.rgb2lab | float64 | 177.0652 | 50.3413 | 3.52x |
| color.lab2lch | float64 | 35.5407 | 14.6054 | 2.43x |
| color.luv2xyz | float64 | 116.1763 | 10.4092 | 11.16x |
| color.combine_stains | float64 | 40.3435 | 12.4536 | 3.24x |
| color.deltaE_cie76 | float64 | 21.7826 | 4.0733 | 5.35x |

Boolean 31×31 rectangles, with one worker (the previous slower morphology workloads):

| Operation | Image | Original ms | C++ ms | Reference / C++ |
|---|---|---:|---:|---:|
| morphology.erosion | 512² | 2.1474 | 0.1536 | 13.98x |
| morphology.dilation | 512² | 2.1626 | 0.1434 | 15.08x |
| morphology.opening | 512² | 3.5935 | 0.2347 | 15.31x |
| morphology.closing | 512² | 4.9590 | 0.2553 | 19.42x |
| morphology.white_tophat | 512² | 3.7541 | 0.3088 | 12.16x |
| morphology.black_tophat | 512² | 3.4080 | 0.2365 | 14.41x |

The [fixed-workload tuning experiment](morphology-tuning.md) preserves separate historical before/after C++ snapshots. Its cases are not added to this current-source total.

1374 of 1393 one-worker native processing workloads measured faster than the reference. These are workload cases, including repeated functions and parameter variants. Identity returns, copies and dtype metadata are excluded from this count and labeled separately in the full reports.

19 one-worker processing workloads remained slower:

| Operation | Input | Shape | Layout | Workload | Reference / C++ |
|---|---|---:|---|---|---:|
| util.img_as_bool | float64 | 256² | standard | default | 0.4091x |
| util.img_as_float64 | float32 | 1024² | standard | default | 0.9909x |
| util.img_as_bool | float32 | 1024² | standard | default | 0.9689x |
| color.rgb2gray | float64 | 1024² | standard | default | 0.9133x |
| util.img_as_bool | float64 | 1024² | standard | default | 0.8040x |
| util.img_as_int | uint16 | 1024² | standard | default | 0.6665x |
| util.img_as_bool | int16 | 1024² | standard | default | 0.9636x |
| util.img_as_bool | int64 | 1024² | standard | default | 0.8914x |
| color.separate_stains | uint16 | 512² | strided | default | 0.7103x |
| color.rgb2hed | uint16 | 512² | strided | default | 0.8216x |
| color.separate_stains | uint16 | 512² | channel-first | default | 0.9901x |
| filters.threshold_yen | float32/int64 | 257 | histogram-only | supplied-histogram | 0.8391x |
| filters.threshold_yen | float32/int64 | 257 | histogram-only | trimmed-histogram | 0.9305x |
| filters.threshold_yen | float32/int64 | 257 | histogram-only | large-counts | 0.9470x |
| filters.threshold_yen | float32/float32 | 257 | histogram-only | supplied-histogram | 0.9483x |
| filters.threshold_isodata | float32/float32 | 257 | histogram-only | trimmed-histogram | 0.9180x |
| filters.threshold_yen | float32/float32 | 257 | histogram-only | large-counts | 0.8998x |
| filters.threshold_isodata | float32/float64 | 257 | histogram-only | trimmed-histogram | 0.6569x |
| filters.threshold_yen | float32/float64 | 257 | histogram-only | large-counts | 0.8121x |

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

Measurements use ordinary system load without CPU affinity or frequency locking. Small differences near 1x can change between runs. These are measured comparisons on this machine, not universal speed guarantees. CI runs parity and benchmark smoke checks on Windows/Linux without asserting speed ratios on shared runners.

Coverage remains partial: 81 native image entries, 2 metadata helpers, 367 inventoried entries still pending. See the [inventory](../coverage/README.md), [roadmap](../coverage/ROADMAP.md) and [documented input contracts](../README.md). Run `python benchmarks/verify.py` to rebuild and reproduce all eleven reports, including this summary. Use `--threads 1` for serial-only verification. The summary rejects mixed source hashes, missing APIs and absent one-worker comparisons.
