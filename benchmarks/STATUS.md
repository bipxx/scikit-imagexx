# Last completed aggregate verification

This 16-report checkpoint predates the `transform.integrate` port. Its source hash and totals describe that checkpoint. The new port has a separate [144-workload one-worker report](integrate-parameters.md); the current registry requires a fresh 17-report aggregate.

Reference: original scikit-image 0.26.0. 2652 comparisons on Windows-11-10.0.26200-SP0; CPU: Intel64 Family 6 Model 151 Stepping 2, GenuineIntel. Individual reports record timestamps, CPU, versions, raw samples and options. These checkpoint reports match source/build SHA-256 (POSIX relative paths and LF line endings):
`1e7ff448ca59f88a2c2d523dd537bfcdf3191d5ebc0306de2d43a497fb6db79c`.

| Report | Cases | Worker counts |
|---|---:|---|
| [Every covered public entry](function-coverage.md) | 542 | 1 |
| [Additional integer widths](conversion-widths.md) | 56 | 1 |
| [Non-default exposure parameters](exposure-parameters.md) | 24 | 1 |
| [Color layouts and uint16 inputs](color-layouts.md) | 120 | 1 |
| [Perceptual color parameters](perceptual-parameters.md) | 30 | 1 |
| [Morphology footprints and output reuse](morphology-parameters.md) | 120 | 1 |
| [Stain matrices and output widths](stain-parameters.md) | 78 | 1 |
| [Lab distances and color routes](color-distance.md) | 123 | 1 |
| [Histogram threshold parameters](threshold-parameters.md) | 76 | 1 |
| [Local threshold windows and dtypes](local-threshold-parameters.md) | 128 | 1 |
| [Alpha and grayscale layouts](alpha-parameters.md) | 208 | 1 |
| [Laplace and Roberts layouts](edge-parameters.md) | 260 | 1 |
| [Farid derivatives and boundaries](farid-parameters.md) | 243 | 1 |
| [Every covered entry with dtype metadata](metadata-parameters.md) | 271 | 1 |
| [Gaussian families and output precision](gaussian-parameters.md) | 225 | 1 |
| [Integral-image dtypes and layouts](integral-parameters.md) | 148 | 1 |

Every workload checks values and dtype against original scikit-image before timing with fallback disabled. Integer-width conversion cases also require exact values and matching identity behavior. Timings include allocation, Python dispatch, integer normalization and any needed input copying. Inputs/imports/I/O are excluded. Reference thread pools use one thread. Native worker counts are explicit.

Examples from the 1024² one-worker default-parameter comparisons (color-space routing is RGB to HSV; CIE76 compares raw Lab values with a channel-reversed second input):

| Operation | Input | Original ms | C++ ms | Reference / C++ |
|---|---|---:|---:|---:|
| morphology.erosion | uint8 | 10.3385 | 0.4798 | 21.55x |
| color.rgb2yuv | uint8 | 17.4836 | 8.9619 | 1.95x |
| color.rgb2xyz | uint8 | 107.8433 | 9.3582 | 11.52x |
| color.rgb2hed | uint8 | 45.6847 | 12.9229 | 3.54x |
| filters.gaussian | uint8 | 27.4894 | 9.0125 | 3.05x |
| filters.sobel | uint8 | 31.4608 | 3.6720 | 8.57x |
| filters.threshold_otsu | uint8 | 4.8783 | 0.2626 | 18.58x |
| morphology.opening | float32 | 9.7115 | 4.4171 | 2.20x |
| color.rgb2xyz | float32 | 58.3478 | 21.5091 | 2.71x |
| color.xyz2rgb | float32 | 75.7217 | 21.7008 | 3.49x |
| color.rgb2hsv | float32 | 157.4054 | 4.7412 | 33.20x |
| color.hsv2rgb | float32 | 148.5068 | 4.8591 | 30.56x |
| color.rgb2luv | float32 | 103.0294 | 32.8128 | 3.14x |
| color.luv2rgb | float32 | 56.4282 | 4.3509 | 12.97x |
| color.separate_stains | float32 | 30.3423 | 12.9005 | 2.35x |
| color.deltaE_cie76 | float32 | 8.8091 | 1.7040 | 5.17x |
| color.convert_colorspace | float32 | 153.1557 | 3.7468 | 40.88x |
| color.deltaE_ciede94 | float32 | 126.0486 | 20.3898 | 6.18x |
| color.deltaE_ciede2000 | float32 | 203.8473 | 98.5172 | 2.07x |
| color.deltaE_cmc | float32 | 311.3172 | 50.8294 | 6.12x |
| morphology.binary_dilation | float64 | 11.7392 | 4.7765 | 2.46x |
| color.rgb2ycbcr | float64 | 15.8591 | 7.4201 | 2.14x |
| color.rgb2lab | float64 | 201.6438 | 53.3562 | 3.78x |
| color.lab2lch | float64 | 38.8072 | 13.3797 | 2.90x |
| color.luv2xyz | float64 | 120.0924 | 8.2799 | 14.50x |
| color.combine_stains | float64 | 46.0909 | 12.7355 | 3.62x |
| color.deltaE_cie76 | float64 | 29.3414 | 4.9139 | 5.97x |

Boolean 31×31 rectangles, with one worker (the previous slower morphology workloads):

| Operation | Image | Original ms | C++ ms | Reference / C++ |
|---|---|---:|---:|---:|
| morphology.erosion | 512² | 2.4025 | 0.1696 | 14.16x |
| morphology.dilation | 512² | 2.2356 | 0.1568 | 14.26x |
| morphology.opening | 512² | 3.4919 | 0.2080 | 16.79x |
| morphology.closing | 512² | 3.4575 | 0.2182 | 15.85x |
| morphology.white_tophat | 512² | 3.5405 | 0.2468 | 14.34x |
| morphology.black_tophat | 512² | 3.8840 | 0.2381 | 16.32x |

The [fixed-workload tuning experiment](morphology-tuning.md) preserves separate historical before/after C++ snapshots. Its cases are not added to this current-source total.

2588 of 2613 one-worker native processing workloads measured faster than the reference. These are workload cases, including repeated functions and parameter variants. Identity returns, copies and dtype metadata are excluded from this count and labeled separately in the full reports.

25 one-worker processing workloads remained slower:

| Operation | Input | Shape | Layout | Workload | Reference / C++ |
|---|---|---:|---|---|---:|
| util.invert | float32 | 256² | standard | default | 0.9434x |
| util.img_as_bool | float64 | 256² | standard | default | 0.5995x |
| util.invert | float64 | 1024² | standard | default | 0.9688x |
| util.img_as_int | uint16 | 1024² | standard | default | 0.7163x |
| util.img_as_bool | int64 | 1024² | standard | default | 0.9546x |
| color.separate_stains | uint16 | 512² | channel-first | default | 0.9728x |
| color.rgb2hed | uint16 | 512² | channel-first | default | 0.9194x |
| morphology.closing | float32 | 512² | rectangle-31 | rectangle-31 | 0.9155x |
| filters.threshold_yen | float32/int64 | 257 | histogram-only | supplied-histogram | 0.8260x |
| filters.threshold_yen | float32/float32 | 257 | histogram-only | supplied-histogram | 0.9642x |
| filters.threshold_isodata | float32/float32 | 257 | histogram-only | supplied-histogram | 0.9831x |
| filters.threshold_isodata | float32/float32 | 257 | histogram-only | trimmed-histogram | 0.9985x |
| filters.threshold_yen | float32/float32 | 257 | histogram-only | large-counts | 0.9567x |
| filters.threshold_yen | float32/float64 | 257 | histogram-only | supplied-histogram | 0.9420x |
| filters.threshold_isodata | float32/float64 | 257 | histogram-only | large-counts | 0.9292x |
| color.gray2rgb | uint32 | 512×512 | reversed | reversed | 0.9823x |
| filters.farid_v | float64 | 1×512 | single-row | single-row | 0.9978x |
| util.invert | float32 | 512² | dtype-metadata | default | 0.9876x |
| util.img_as_bool | float32 | 512² | dtype-metadata | default | 0.8783x |
| util.img_as_bool | float64 | 512² | dtype-metadata | default | 0.9473x |
| filters.difference_of_gaussians | float64 | 512×1 | single-column | single-column | 0.8885x |
| transform.integral_image | uint32 | 512×1 | single-column | single-column | 0.9562x |
| transform.integral_image | uint64 | 512×1 | single-column | single-column | 0.6352x |
| transform.integral_image | int64 | 512×1 | single-column | single-column | 0.9557x |
| transform.integral_image | float16 | 512×1 | single-column | single-column | 0.9798x |

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

Gaussian-family workloads cover N-D and strided images, channels, all five boundaries, typed sigma, endian conversion and float32/float64 output buffers. Native convolution retains SciPy coefficient order and double accumulation; float32 results round after each axis. Difference of Gaussians reuses its first blur buffer, and unsharp masking fuses the subtraction, scale, addition and clipping stages. The [fixed interior-span experiment](gaussian-tuning.md) retains baseline, optimized and independent confirmation snapshots.

Integral images preserve ordered floating prefixes, modular integer overflow and physical output layout. The [dispatch experiment](integral-dispatch-tuning.md) and [paired integer-prefix experiment](integral-integer-tuning.md) and [native-call experiment](integral-binding-tuning.md) retain independent confirmation samples and rejected variants. All current integral workloads check exact values, dtype tags, strides and warnings before timing.

Measurements use ordinary system load without CPU affinity or frequency locking. Small differences near 1x can change between runs. These are measured comparisons on this machine, not universal speed guarantees. CI runs parity and benchmark smoke checks on Windows/Linux without asserting speed ratios on shared runners.

Coverage remains partial: 91 native image entries, 2 metadata helpers, 357 inventoried entries still pending. See the [inventory](../coverage/README.md), [roadmap](../coverage/ROADMAP.md) and [documented input contracts](../README.md). Run `python benchmarks/verify.py` to rebuild and reproduce all sixteen reports, including this summary. Use `--threads 1` for serial-only verification. The summary rejects mixed source hashes, missing APIs and absent one-worker comparisons.
