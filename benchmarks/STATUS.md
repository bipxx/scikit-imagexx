# Current performance verification

Reference: original scikit-image 0.26.0. 1705 comparisons on Windows-11-10.0.26200-SP0; CPU: Intel64 Family 6 Model 151 Stepping 2, GenuineIntel. Individual reports record timestamps, CPU, versions, raw samples and options. All reports match the current source/build SHA-256 (POSIX relative paths and LF line endings):
`e5c9f695c5dfd43422bafb0e59e7916e7d3a51df9df7425e65df12bc98e083b2`.

| Report | Cases | Worker counts |
|---|---:|---|
| [Every covered public entry](function-coverage.md) | 506 | 1 |
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

Every workload checks values and dtype against original scikit-image before timing with fallback disabled. Integer-width conversion cases also require exact values and matching identity behavior. Timings include allocation, Python dispatch, integer normalization and any needed input copying. Inputs/imports/I/O are excluded. Reference thread pools use one thread. Native worker counts are explicit.

Examples from the 1024² one-worker default-parameter comparisons (color-space routing is RGB to HSV; CIE76 compares raw Lab values with a channel-reversed second input):

| Operation | Input | Original ms | C++ ms | Reference / C++ |
|---|---|---:|---:|---:|
| morphology.erosion | uint8 | 10.1025 | 0.4625 | 21.84x |
| color.rgb2yuv | uint8 | 15.4361 | 8.3561 | 1.85x |
| color.rgb2xyz | uint8 | 85.6207 | 8.1730 | 10.48x |
| color.rgb2hed | uint8 | 42.3721 | 13.7455 | 3.08x |
| filters.gaussian | uint8 | 21.8625 | 5.2360 | 4.18x |
| filters.sobel | uint8 | 23.6826 | 3.1466 | 7.53x |
| filters.threshold_otsu | uint8 | 3.4186 | 0.2466 | 13.86x |
| morphology.opening | float32 | 9.7412 | 2.8060 | 3.47x |
| color.rgb2xyz | float32 | 54.6297 | 23.0229 | 2.37x |
| color.xyz2rgb | float32 | 73.4533 | 25.3937 | 2.89x |
| color.rgb2hsv | float32 | 161.4551 | 4.8827 | 33.07x |
| color.hsv2rgb | float32 | 137.6603 | 4.8389 | 28.45x |
| color.rgb2luv | float32 | 91.6968 | 31.4657 | 2.91x |
| color.luv2rgb | float32 | 49.4842 | 5.4703 | 9.05x |
| color.separate_stains | float32 | 25.7871 | 12.9746 | 1.99x |
| color.deltaE_cie76 | float32 | 8.2799 | 1.5609 | 5.30x |
| color.convert_colorspace | float32 | 149.2115 | 3.9322 | 37.95x |
| color.deltaE_ciede94 | float32 | 120.7940 | 20.4958 | 5.89x |
| color.deltaE_ciede2000 | float32 | 183.6372 | 97.9609 | 1.87x |
| color.deltaE_cmc | float32 | 216.0457 | 46.6271 | 4.63x |
| morphology.binary_dilation | float64 | 11.8769 | 5.1005 | 2.33x |
| color.rgb2ycbcr | float64 | 14.3216 | 6.6142 | 2.17x |
| color.rgb2lab | float64 | 295.6673 | 65.9720 | 4.48x |
| color.lab2lch | float64 | 40.1111 | 15.1017 | 2.66x |
| color.luv2xyz | float64 | 93.6591 | 7.2964 | 12.84x |
| color.combine_stains | float64 | 44.1093 | 13.9173 | 3.17x |
| color.deltaE_cie76 | float64 | 24.1262 | 3.9432 | 6.12x |

Boolean 31×31 rectangles, with one worker (the previous slower morphology workloads):

| Operation | Image | Original ms | C++ ms | Reference / C++ |
|---|---|---:|---:|---:|
| morphology.erosion | 512² | 2.2963 | 0.1548 | 14.84x |
| morphology.dilation | 512² | 2.1802 | 0.1488 | 14.65x |
| morphology.opening | 512² | 3.5699 | 0.2183 | 16.35x |
| morphology.closing | 512² | 3.5145 | 0.2186 | 16.08x |
| morphology.white_tophat | 512² | 3.9286 | 0.2561 | 15.34x |
| morphology.black_tophat | 512² | 4.1123 | 0.3281 | 12.53x |

The [fixed-workload tuning experiment](morphology-tuning.md) preserves separate historical before/after C++ snapshots. Its cases are not added to this current-source total.

1655 of 1677 one-worker native processing workloads measured faster than the reference. These are workload cases, including repeated functions and parameter variants. Identity returns, copies and dtype metadata are excluded from this count and labeled separately in the full reports.

22 one-worker processing workloads remained slower:

| Operation | Input | Shape | Layout | Workload | Reference / C++ |
|---|---|---:|---|---|---:|
| util.img_as_bool | uint8 | 256² | standard | default | 0.7967x |
| util.img_as_bool | float64 | 256² | standard | default | 0.5285x |
| util.img_as_bool | uint8 | 1024² | standard | default | 0.9479x |
| util.img_as_float64 | float32 | 1024² | standard | default | 0.9535x |
| util.img_as_float | bool | 1024² | standard | default | 0.9756x |
| util.img_as_int | uint16 | 1024² | standard | default | 0.6913x |
| util.img_as_bool | int16 | 1024² | standard | default | 0.9064x |
| util.img_as_int | uint64 | 1024² | standard | default | 0.9662x |
| color.separate_stains | uint16 | 512² | strided | default | 0.8851x |
| color.separate_stains | uint16 | 512² | channel-first | default | 0.9292x |
| color.rgb2hed | uint16 | 512² | channel-first | default | 0.9095x |
| morphology.black_tophat | uint8 | 512² | rectangle-31 | rectangle-31 | 0.9769x |
| morphology.black_tophat | float32 | 512² | rectangle-31 | rectangle-31 | 0.9859x |
| filters.threshold_yen | float32/int64 | 257 | histogram-only | supplied-histogram | 0.7485x |
| filters.threshold_isodata | float32/int64 | 257 | histogram-only | supplied-histogram | 0.9961x |
| filters.threshold_isodata | float32/int64 | 257 | histogram-only | trimmed-histogram | 0.6622x |
| filters.threshold_yen | float32/float32 | 257 | histogram-only | supplied-histogram | 0.9692x |
| filters.threshold_isodata | float32/float32 | 257 | histogram-only | trimmed-histogram | 0.9907x |
| filters.threshold_yen | float32/float32 | 257 | histogram-only | large-counts | 0.9439x |
| filters.threshold_yen | float32/float64 | 257 | histogram-only | supplied-histogram | 0.3850x |
| filters.roberts_pos_diag | float32 | 512×1 | single-column | single-column | 0.9803x |
| filters.roberts_pos_diag | float64 | 512×1 | single-column | single-column | 0.9126x |

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

Laplace uses sparse neighbor loads in scipy coefficient order, retaining singleton-axis terms. Roberts fuses cross differences and rounded magnitude stages. Half and endian conversion, mask erosion/broadcast multiplication, layout preparation and allocation are included in the edge report. The [fixed column experiment](edge-tuning.md) retains separate baseline, optimized and confirmation snapshots.

Measurements use ordinary system load without CPU affinity or frequency locking. Small differences near 1x can change between runs. These are measured comparisons on this machine, not universal speed guarantees. CI runs parity and benchmark smoke checks on Windows/Linux without asserting speed ratios on shared runners.

Coverage remains partial: 85 native image entries, 2 metadata helpers, 363 inventoried entries still pending. See the [inventory](../coverage/README.md), [roadmap](../coverage/ROADMAP.md) and [documented input contracts](../README.md). Run `python benchmarks/verify.py` to rebuild and reproduce all twelve reports, including this summary. Use `--threads 1` for serial-only verification. The summary rejects mixed source hashes, missing APIs and absent one-worker comparisons.
