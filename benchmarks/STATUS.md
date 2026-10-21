# Last completed whole-library performance checkpoint

These 17 reports were refreshed at `9b285ad`, before the local-mean port. They cover the 94 reviewed entries at that checkpoint. The new port changes the source fingerprint and requires a fresh aggregate after validation; its experiments are recorded separately.

Reference: original scikit-image 0.26.0. 2805 comparisons on Windows-11-10.0.26200-SP0; CPU: Intel64 Family 6 Model 151 Stepping 2, GenuineIntel. Individual reports record timestamps, CPU, versions, raw samples and options. All reports match the recorded checkpoint source/build SHA-256 (POSIX relative paths and LF line endings):
`23b011e17f94169a964984643fa8d2f6c59cd34a19aa9fb205a4211709a31539`.

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
| morphology.erosion | uint8 | 9.7751 | 0.5159 | 18.95x |
| color.rgb2yuv | uint8 | 20.5654 | 9.8354 | 2.09x |
| color.rgb2xyz | uint8 | 126.8090 | 9.6564 | 13.13x |
| color.rgb2hed | uint8 | 56.5077 | 14.3947 | 3.93x |
| filters.gaussian | uint8 | 34.6495 | 13.4194 | 2.58x |
| filters.sobel | uint8 | 32.0011 | 3.9081 | 8.19x |
| filters.threshold_otsu | uint8 | 4.7264 | 0.3187 | 14.83x |
| morphology.opening | float32 | 9.5049 | 3.1440 | 3.02x |
| color.rgb2xyz | float32 | 63.4347 | 26.8065 | 2.37x |
| color.xyz2rgb | float32 | 100.0884 | 31.9422 | 3.13x |
| color.rgb2hsv | float32 | 220.2321 | 6.2925 | 35.00x |
| color.hsv2rgb | float32 | 198.4013 | 6.3063 | 31.46x |
| color.rgb2luv | float32 | 139.1283 | 40.6713 | 3.42x |
| color.luv2rgb | float32 | 79.4612 | 7.0720 | 11.24x |
| color.separate_stains | float32 | 40.0711 | 17.8424 | 2.25x |
| color.deltaE_cie76 | float32 | 10.0028 | 2.7292 | 3.67x |
| color.convert_colorspace | float32 | 192.4244 | 5.1162 | 37.61x |
| color.deltaE_ciede94 | float32 | 143.2583 | 23.5950 | 6.07x |
| color.deltaE_ciede2000 | float32 | 274.7464 | 111.2199 | 2.47x |
| color.deltaE_cmc | float32 | 285.1100 | 59.8621 | 4.76x |
| morphology.binary_dilation | float64 | 13.3824 | 6.3126 | 2.12x |
| color.rgb2ycbcr | float64 | 23.1222 | 9.0900 | 2.54x |
| color.rgb2lab | float64 | 297.9290 | 69.4228 | 4.29x |
| color.lab2lch | float64 | 44.0657 | 16.9753 | 2.60x |
| color.luv2xyz | float64 | 114.4267 | 9.3380 | 12.25x |
| color.combine_stains | float64 | 53.8737 | 15.7378 | 3.42x |
| color.deltaE_cie76 | float64 | 30.5496 | 5.7856 | 5.28x |

Boolean 31×31 rectangles, with one worker (the previous slower morphology workloads):

| Operation | Image | Original ms | C++ ms | Reference / C++ |
|---|---|---:|---:|---:|
| morphology.erosion | 512² | 2.6317 | 0.1903 | 13.83x |
| morphology.dilation | 512² | 2.4668 | 0.1942 | 12.70x |
| morphology.opening | 512² | 5.4763 | 0.3608 | 15.18x |
| morphology.closing | 512² | 5.1798 | 0.3619 | 14.31x |
| morphology.white_tophat | 512² | 4.9855 | 0.4123 | 12.09x |
| morphology.black_tophat | 512² | 5.9195 | 0.4435 | 13.35x |

The [fixed-workload tuning experiment](morphology-tuning.md) preserves separate historical before/after C++ snapshots. Its cases are not added to this current-source total.

2736 of 2766 one-worker native processing workloads measured faster than the reference. These are workload cases, including repeated functions and parameter variants. Identity returns, copies and dtype metadata are excluded from this count and labeled separately in the full reports.

30 one-worker processing workloads remained slower:

| Operation | Input | Shape | Layout | Workload | Reference / C++ |
|---|---|---:|---|---|---:|
| util.invert | float32 | 256² | standard | default | 0.7062x |
| util.img_as_bool | float32 | 256² | standard | default | 0.4813x |
| util.img_as_bool | float64 | 256² | standard | default | 0.6587x |
| util.invert | float32 | 1024² | standard | default | 0.9372x |
| util.invert | float64 | 1024² | standard | default | 0.9513x |
| util.img_as_bool | float64 | 1024² | standard | default | 0.5480x |
| util.img_as_float | bool | 1024² | standard | default | 0.9072x |
| util.img_as_bool | int8 | 1024² | standard | default | 0.9984x |
| util.img_as_int | uint16 | 1024² | standard | default | 0.7617x |
| util.img_as_bool | int64 | 1024² | standard | default | 0.8725x |
| morphology.white_tophat | float32 | 512² | rectangle-31 | rectangle-31 | 0.8861x |
| morphology.black_tophat | float32 | 512² | rectangle-31 | rectangle-31 | 0.9114x |
| morphology.closing | float64 | 512² | rectangle-31 | rectangle-31 | 0.9534x |
| filters.threshold_yen | float32/int64 | 257 | histogram-only | supplied-histogram | 0.9914x |
| filters.threshold_isodata | float32/int64 | 257 | histogram-only | supplied-histogram | 0.8888x |
| filters.threshold_yen | float32/int64 | 257 | histogram-only | trimmed-histogram | 0.9040x |
| filters.threshold_isodata | float32/int64 | 257 | histogram-only | trimmed-histogram | 0.9622x |
| filters.threshold_yen | float32/int64 | 257 | histogram-only | large-counts | 0.8136x |
| filters.threshold_isodata | float32/float32 | 257 | histogram-only | supplied-histogram | 0.9538x |
| filters.threshold_yen | float32/float32 | 257 | histogram-only | large-counts | 0.7352x |
| filters.threshold_yen | float32/float64 | 257 | histogram-only | supplied-histogram | 0.8505x |
| filters.threshold_isodata | float32/float64 | 257 | histogram-only | supplied-histogram | 0.9364x |
| color.gray2rgb | int16 | 512×512 | strided | strided | 0.8713x |
| color.rgb2ypbpr | float32 | 512² | dtype-metadata | default | 0.9538x |
| util.invert | float32 | 512² | dtype-metadata | default | 0.7692x |
| util.img_as_float64 | float32 | 512² | dtype-metadata | default | 0.9297x |
| util.img_as_bool | float32 | 512² | dtype-metadata | default | 0.8232x |
| util.invert | float64 | 512² | dtype-metadata | default | 0.9825x |
| filters.difference_of_gaussians | float32 | 512×1 | single-column | single-column | 0.9635x |
| filters.difference_of_gaussians | float64 | 512×1 | single-column | single-column | 0.9273x |

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

Coverage remains partial: 92 native image entries, 2 metadata helpers, 361 inventoried entries still pending. See the [inventory](../coverage/README.md), [roadmap](../coverage/ROADMAP.md) and [documented input contracts](../README.md). Run `python benchmarks/verify.py` to rebuild and reproduce all 17 reports, including this summary. Use `--threads 1` for serial-only verification. The summary rejects mixed source hashes, missing APIs and absent one-worker comparisons.
