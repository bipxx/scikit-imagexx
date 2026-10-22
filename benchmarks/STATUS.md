# Last completed whole-library performance checkpoint

These 19 reports were refreshed at `0dcaf50`, covering the 96 reviewed entries before the minimum-threshold port. That port changes the source fingerprint; its dedicated measurements remain separate until the next complete refresh.

Reference: original scikit-image 0.26.0. 3069 comparisons on Windows-11-10.0.26200-SP0; CPU: Intel64 Family 6 Model 151 Stepping 2, GenuineIntel. Individual reports record timestamps, CPU, versions, raw samples and options. All reports match the recorded checkpoint source/build SHA-256 (POSIX relative paths and LF line endings):
`acb2ae0434191c9257218d9191ea4e23d9fba4e2a701d233147951ec4aeb87a0`.

| Report | Cases | Worker counts |
|---|---:|---|
| [Every covered public entry](function-coverage.md) | 560 | 1 |
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
| [Every covered entry with dtype metadata](metadata-parameters.md) | 280 | 1 |
| [Gaussian families and output precision](gaussian-parameters.md) | 225 | 1 |
| [Integral-image dtypes and layouts](integral-parameters.md) | 148 | 1 |
| [Integral-window batches and layouts](integrate-parameters.md) | 144 | 1 |
| [Raw local-block means and layouts](downscale-parameters.md) | 120 | 1 |
| [Low-contrast percentiles and layouts](contrast-parameters.md) | 126 | 1 |

Every workload checks values and dtype against original scikit-image before timing with fallback disabled. Integer-width conversion cases also require exact values and matching identity behavior. Timings include allocation, Python dispatch, integer normalization and any needed input copying. Inputs/imports/I/O are excluded. Reference thread pools use one thread. Native worker counts are explicit.

Examples from the 1024² one-worker default-parameter comparisons (color-space routing is RGB to HSV; CIE76 compares raw Lab values with a channel-reversed second input):

| Operation | Input | Original ms | C++ ms | Reference / C++ |
|---|---|---:|---:|---:|
| morphology.erosion | uint8 | 10.1402 | 0.4725 | 21.46x |
| color.rgb2yuv | uint8 | 17.0204 | 8.0631 | 2.11x |
| color.rgb2xyz | uint8 | 85.0076 | 7.9191 | 10.73x |
| color.rgb2hed | uint8 | 39.6427 | 13.0513 | 3.04x |
| filters.gaussian | uint8 | 24.5456 | 7.6705 | 3.20x |
| filters.sobel | uint8 | 24.6058 | 3.3192 | 7.41x |
| filters.threshold_otsu | uint8 | 3.7253 | 0.2548 | 14.62x |
| morphology.opening | float32 | 9.3310 | 4.2781 | 2.18x |
| color.rgb2xyz | float32 | 58.6486 | 22.2901 | 2.63x |
| color.xyz2rgb | float32 | 73.6355 | 21.8590 | 3.37x |
| color.rgb2hsv | float32 | 151.5615 | 3.9115 | 38.75x |
| color.hsv2rgb | float32 | 156.6214 | 4.4774 | 34.98x |
| color.rgb2luv | float32 | 103.4417 | 31.2897 | 3.31x |
| color.luv2rgb | float32 | 48.0599 | 4.4646 | 10.76x |
| color.separate_stains | float32 | 35.8793 | 15.7692 | 2.28x |
| color.deltaE_cie76 | float32 | 8.5992 | 2.1069 | 4.08x |
| color.convert_colorspace | float32 | 150.6866 | 4.2518 | 35.44x |
| color.deltaE_ciede94 | float32 | 134.3298 | 20.4427 | 6.57x |
| color.deltaE_ciede2000 | float32 | 198.7542 | 97.2833 | 2.04x |
| color.deltaE_cmc | float32 | 197.7662 | 43.0687 | 4.59x |
| morphology.binary_dilation | float64 | 11.8995 | 5.4248 | 2.19x |
| color.rgb2ycbcr | float64 | 36.4561 | 15.3319 | 2.38x |
| color.rgb2lab | float64 | 201.2339 | 51.3928 | 3.92x |
| color.lab2lch | float64 | 36.3434 | 12.9538 | 2.81x |
| color.luv2xyz | float64 | 103.6785 | 7.3689 | 14.07x |
| color.combine_stains | float64 | 48.1906 | 11.7568 | 4.10x |
| color.deltaE_cie76 | float64 | 30.0181 | 4.9745 | 6.03x |

Boolean 31×31 rectangles, with one worker (the previous slower morphology workloads):

| Operation | Image | Original ms | C++ ms | Reference / C++ |
|---|---|---:|---:|---:|
| morphology.erosion | 512² | 2.2191 | 0.1549 | 14.32x |
| morphology.dilation | 512² | 2.2357 | 0.1548 | 14.45x |
| morphology.opening | 512² | 3.4245 | 0.2076 | 16.50x |
| morphology.closing | 512² | 3.4450 | 0.2149 | 16.03x |
| morphology.white_tophat | 512² | 3.3625 | 0.2193 | 15.33x |
| morphology.black_tophat | 512² | 3.3569 | 0.2659 | 12.62x |

The [fixed-workload tuning experiment](morphology-tuning.md) preserves separate historical before/after C++ snapshots. Its cases are not added to this current-source total.

3015 of 3030 one-worker native processing workloads measured faster than the reference. These are workload cases, including repeated functions and parameter variants. Identity returns, copies and dtype metadata are excluded from this count and labeled separately in the full reports.

15 one-worker processing workloads remained slower:

| Operation | Input | Shape | Layout | Workload | Reference / C++ |
|---|---|---:|---|---|---:|
| util.invert | float32 | 256² | standard | default | 0.9415x |
| util.img_as_bool | float64 | 256² | standard | default | 0.6793x |
| util.img_as_float64 | float32 | 1024² | standard | default | 0.9449x |
| util.img_as_bool | float64 | 1024² | standard | default | 0.8349x |
| util.img_as_int | uint16 | 1024² | standard | default | 0.7959x |
| util.img_as_bool | int64 | 1024² | standard | default | 0.8559x |
| color.separate_stains | uint16 | 512² | strided | default | 0.8766x |
| color.rgb2hed | uint16 | 512² | strided | default | 0.8333x |
| color.separate_stains | uint16 | 512² | channel-first | default | 0.8284x |
| color.rgb2hed | uint16 | 512² | channel-first | default | 0.9501x |
| filters.threshold_yen | float32/int64 | 257 | histogram-only | supplied-histogram | 0.9728x |
| filters.threshold_isodata | float32/float32 | 257 | histogram-only | supplied-histogram | 0.9771x |
| filters.threshold_yen | float32/float32 | 257 | histogram-only | trimmed-histogram | 0.9735x |
| filters.threshold_yen | float32/float64 | 257 | histogram-only | supplied-histogram | 0.9891x |
| util.img_as_bool | float32 | 512² | dtype-metadata | default | 0.9924x |

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

Integral-window queries read only selected corners through the original byte strides. They preserve raw scalar-to-double conversion and ordered corner additions; SIMD operates across independent windows. The report includes single-window overhead and large batches without charging either backend for table construction. The [coordinate experiment](integrate-metadata-tuning.md) retains paired baseline/current and independent confirmation measurements, including adverse old/new results.

Measurements use ordinary system load without CPU affinity or frequency locking. Small differences near 1x can change between runs. These are measured comparisons on this machine, not universal speed guarantees. CI runs parity and benchmark smoke checks on Windows/Linux without asserting speed ratios on shared runners.

Coverage remains partial: 94 native image entries, 2 metadata helpers, 359 inventoried entries still pending. See the [inventory](../coverage/README.md), [roadmap](../coverage/ROADMAP.md) and [documented input contracts](../README.md). Run `python benchmarks/verify.py` to rebuild and reproduce all 19 reports, including this summary. Use `--threads 1` for serial-only verification. The summary rejects mixed source hashes, missing APIs and absent one-worker comparisons.
