# Current performance verification

Reference: original scikit-image 0.26.0. 2805 comparisons on Windows-11-10.0.26200-SP0; CPU: Intel64 Family 6 Model 151 Stepping 2, GenuineIntel. Individual reports record timestamps, CPU, versions, raw samples and options. All reports match the current source/build SHA-256 (POSIX relative paths and LF line endings):
`9df0360bf25a89ce495437806fbbcb310b07ba39e67137a983e9780194ee310a`.

| Report | Cases | Worker counts |
|---|---:|---|
| [Every covered public entry](function-coverage.md) | 548 | 1 |
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
| [Every covered entry with dtype metadata](metadata-parameters.md) | 274 | 1 |
| [Gaussian families and output precision](gaussian-parameters.md) | 225 | 1 |
| [Integral-image dtypes and layouts](integral-parameters.md) | 148 | 1 |
| [Integral-window batches and layouts](integrate-parameters.md) | 144 | 1 |

Every workload checks values and dtype against original scikit-image before timing with fallback disabled. Integer-width conversion cases also require exact values and matching identity behavior. Timings include allocation, Python dispatch, integer normalization and any needed input copying. Inputs/imports/I/O are excluded. Reference thread pools use one thread. Native worker counts are explicit.

Examples from the 1024² one-worker default-parameter comparisons (color-space routing is RGB to HSV; CIE76 compares raw Lab values with a channel-reversed second input):

| Operation | Input | Original ms | C++ ms | Reference / C++ |
|---|---|---:|---:|---:|
| morphology.erosion | uint8 | 10.1291 | 0.4821 | 21.01x |
| color.rgb2yuv | uint8 | 16.7671 | 7.7787 | 2.16x |
| color.rgb2xyz | uint8 | 94.1477 | 8.5523 | 11.01x |
| color.rgb2hed | uint8 | 40.4777 | 12.6257 | 3.21x |
| filters.gaussian | uint8 | 25.2425 | 8.3944 | 3.01x |
| filters.sobel | uint8 | 23.2664 | 3.2881 | 7.08x |
| filters.threshold_otsu | uint8 | 4.3299 | 0.2719 | 15.93x |
| morphology.opening | float32 | 9.6659 | 3.0549 | 3.16x |
| color.rgb2xyz | float32 | 54.9161 | 20.5848 | 2.67x |
| color.xyz2rgb | float32 | 73.8705 | 21.9296 | 3.37x |
| color.rgb2hsv | float32 | 150.8013 | 4.2770 | 35.26x |
| color.hsv2rgb | float32 | 142.7686 | 4.5785 | 31.18x |
| color.rgb2luv | float32 | 103.3608 | 31.4688 | 3.28x |
| color.luv2rgb | float32 | 59.5531 | 4.8775 | 12.21x |
| color.separate_stains | float32 | 29.7606 | 12.9103 | 2.31x |
| color.deltaE_cie76 | float32 | 9.1974 | 1.7406 | 5.28x |
| color.convert_colorspace | float32 | 160.6595 | 4.3535 | 36.90x |
| color.deltaE_ciede94 | float32 | 124.2461 | 20.5166 | 6.06x |
| color.deltaE_ciede2000 | float32 | 233.3598 | 100.0019 | 2.33x |
| color.deltaE_cmc | float32 | 189.5905 | 42.8823 | 4.42x |
| morphology.binary_dilation | float64 | 12.2940 | 5.4966 | 2.24x |
| color.rgb2ycbcr | float64 | 16.7684 | 6.7717 | 2.48x |
| color.rgb2lab | float64 | 195.9385 | 52.2171 | 3.75x |
| color.lab2lch | float64 | 36.2458 | 12.8903 | 2.81x |
| color.luv2xyz | float64 | 108.5471 | 7.1723 | 15.13x |
| color.combine_stains | float64 | 46.5171 | 13.8335 | 3.36x |
| color.deltaE_cie76 | float64 | 27.4868 | 4.4372 | 6.19x |

Boolean 31×31 rectangles, with one worker (the previous slower morphology workloads):

| Operation | Image | Original ms | C++ ms | Reference / C++ |
|---|---|---:|---:|---:|
| morphology.erosion | 512² | 2.2451 | 0.1564 | 14.36x |
| morphology.dilation | 512² | 2.2623 | 0.1578 | 14.34x |
| morphology.opening | 512² | 3.5974 | 0.2268 | 15.86x |
| morphology.closing | 512² | 3.4676 | 0.2246 | 15.44x |
| morphology.white_tophat | 512² | 3.5218 | 0.2422 | 14.54x |
| morphology.black_tophat | 512² | 3.5674 | 0.2440 | 14.62x |

The [fixed-workload tuning experiment](morphology-tuning.md) preserves separate historical before/after C++ snapshots. Its cases are not added to this current-source total.

2741 of 2766 one-worker native processing workloads measured faster than the reference. These are workload cases, including repeated functions and parameter variants. Identity returns, copies and dtype metadata are excluded from this count and labeled separately in the full reports.

25 one-worker processing workloads remained slower:

| Operation | Input | Shape | Layout | Workload | Reference / C++ |
|---|---|---:|---|---|---:|
| util.invert | float32 | 256² | standard | default | 0.9520x |
| util.img_as_bool | float64 | 256² | standard | default | 0.6852x |
| util.img_as_bool | float64 | 1024² | standard | default | 0.9688x |
| util.img_as_float64 | bool | 1024² | standard | default | 0.9419x |
| util.img_as_int | uint16 | 1024² | standard | default | 0.7608x |
| util.img_as_bool | uint32 | 1024² | standard | default | 0.9301x |
| util.img_as_bool | uint64 | 1024² | standard | default | 0.6934x |
| color.separate_stains | uint16 | 512² | strided | default | 0.9786x |
| color.rgb2hed | uint16 | 512² | strided | default | 0.9592x |
| color.separate_stains | uint16 | 512² | channel-first | default | 0.8513x |
| color.rgb2hed | uint16 | 512² | channel-first | default | 0.8676x |
| morphology.white_tophat | float32 | 512² | rectangle-31 | rectangle-31 | 0.9965x |
| morphology.black_tophat | float32 | 512² | rectangle-31 | rectangle-31 | 0.9782x |
| filters.threshold_yen | float32/int64 | 257 | histogram-only | trimmed-histogram | 0.9514x |
| filters.threshold_yen | float32/int64 | 257 | histogram-only | large-counts | 0.7640x |
| color.gray2rgb | bool | 512×512 | reversed | reversed | 0.6229x |
| color.gray2rgb | uint16 | 512×512 | strided | strided | 0.8139x |
| util.img_as_bool | float32 | 512² | dtype-metadata | default | 0.9493x |
| morphology.black_tophat | float64 | 512² | dtype-metadata | default | 0.9239x |
| color.rgb2gray | float64 | 512² | dtype-metadata | default | 0.9817x |
| util.img_as_bool | float64 | 512² | dtype-metadata | default | 0.9103x |
| filters.gaussian | int64 | 512×512 | fortran | fortran | 0.7967x |
| transform.integral_image | uint64 | 512×1 | single-column | single-column | 0.9114x |
| transform.integral_image | int64 | 512×1 | single-column | single-column | 0.9225x |
| transform.integral_image | float16 | 512×1 | single-column | single-column | 0.8320x |

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

Coverage remains partial: 92 native image entries, 2 metadata helpers, 356 inventoried entries still pending. See the [inventory](../coverage/README.md), [roadmap](../coverage/ROADMAP.md) and [documented input contracts](../README.md). Run `python benchmarks/verify.py` to rebuild and reproduce all 17 reports, including this summary. Use `--threads 1` for serial-only verification. The summary rejects mixed source hashes, missing APIs and absent one-worker comparisons.
