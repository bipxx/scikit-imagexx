# Current performance verification

Reference: original scikit-image 0.26.0. 2495 comparisons on Windows-11-10.0.26200-SP0; CPU: Intel64 Family 6 Model 151 Stepping 2, GenuineIntel. Individual reports record timestamps, CPU, versions, raw samples and options. All reports match the current source/build SHA-256 (POSIX relative paths and LF line endings):
`423a63332ff25215f86c978a99f1fb6c9e52ea071e7a1cbe9ca31b1365afb94b`.

| Report | Cases | Worker counts |
|---|---:|---|
| [Every covered public entry](function-coverage.md) | 536 | 1 |
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
| [Every covered entry with dtype metadata](metadata-parameters.md) | 268 | 1 |
| [Gaussian families and output precision](gaussian-parameters.md) | 225 | 1 |

Every workload checks values and dtype against original scikit-image before timing with fallback disabled. Integer-width conversion cases also require exact values and matching identity behavior. Timings include allocation, Python dispatch, integer normalization and any needed input copying. Inputs/imports/I/O are excluded. Reference thread pools use one thread. Native worker counts are explicit.

Examples from the 1024² one-worker default-parameter comparisons (color-space routing is RGB to HSV; CIE76 compares raw Lab values with a channel-reversed second input):

| Operation | Input | Original ms | C++ ms | Reference / C++ |
|---|---|---:|---:|---:|
| morphology.erosion | uint8 | 11.4702 | 0.7270 | 15.78x |
| color.rgb2yuv | uint8 | 23.0225 | 10.7120 | 2.15x |
| color.rgb2xyz | uint8 | 162.8747 | 12.2747 | 13.27x |
| color.rgb2hed | uint8 | 66.7280 | 16.3940 | 4.07x |
| filters.gaussian | uint8 | 38.4374 | 14.5674 | 2.64x |
| filters.sobel | uint8 | 42.9994 | 4.8587 | 8.85x |
| filters.threshold_otsu | uint8 | 7.3178 | 0.3406 | 21.48x |
| morphology.opening | float32 | 19.5442 | 5.0564 | 3.87x |
| color.rgb2xyz | float32 | 97.7498 | 28.3147 | 3.45x |
| color.xyz2rgb | float32 | 121.8268 | 32.7251 | 3.72x |
| color.rgb2hsv | float32 | 300.9083 | 9.1383 | 32.93x |
| color.hsv2rgb | float32 | 347.6694 | 11.9402 | 29.12x |
| color.rgb2luv | float32 | 274.4324 | 48.0359 | 5.71x |
| color.luv2rgb | float32 | 179.5398 | 20.6890 | 8.68x |
| color.separate_stains | float32 | 73.1264 | 28.6597 | 2.55x |
| color.deltaE_cie76 | float32 | 31.7164 | 7.3069 | 4.34x |
| color.convert_colorspace | float32 | 338.4239 | 15.1622 | 22.32x |
| color.deltaE_ciede94 | float32 | 342.9856 | 35.3727 | 9.70x |
| color.deltaE_ciede2000 | float32 | 578.7090 | 155.8728 | 3.71x |
| color.deltaE_cmc | float32 | 530.7122 | 74.2998 | 7.14x |
| morphology.binary_dilation | float64 | 15.9711 | 6.8090 | 2.35x |
| color.rgb2ycbcr | float64 | 29.4851 | 11.2082 | 2.63x |
| color.rgb2lab | float64 | 266.1978 | 69.4151 | 3.83x |
| color.lab2lch | float64 | 48.8200 | 16.1023 | 3.03x |
| color.luv2xyz | float64 | 131.1770 | 9.5621 | 13.72x |
| color.combine_stains | float64 | 57.8566 | 15.6825 | 3.69x |
| color.deltaE_cie76 | float64 | 31.5897 | 6.5523 | 4.82x |

Boolean 31×31 rectangles, with one worker (the previous slower morphology workloads):

| Operation | Image | Original ms | C++ ms | Reference / C++ |
|---|---|---:|---:|---:|
| morphology.erosion | 512² | 2.2426 | 0.1707 | 13.13x |
| morphology.dilation | 512² | 2.3878 | 0.1670 | 14.30x |
| morphology.opening | 512² | 3.7670 | 0.2500 | 15.07x |
| morphology.closing | 512² | 3.7106 | 0.2572 | 14.43x |
| morphology.white_tophat | 512² | 4.1356 | 0.2972 | 13.91x |
| morphology.black_tophat | 512² | 4.6483 | 0.3718 | 12.50x |

The [fixed-workload tuning experiment](morphology-tuning.md) preserves separate historical before/after C++ snapshots. Its cases are not added to this current-source total.

2427 of 2456 one-worker native processing workloads measured faster than the reference. These are workload cases, including repeated functions and parameter variants. Identity returns, copies and dtype metadata are excluded from this count and labeled separately in the full reports.

29 one-worker processing workloads remained slower:

| Operation | Input | Shape | Layout | Workload | Reference / C++ |
|---|---|---:|---|---|---:|
| util.invert | float32 | 256² | standard | default | 0.9728x |
| util.img_as_bool | float64 | 256² | standard | default | 0.6125x |
| util.img_as_bool | uint8 | 1024² | standard | default | 0.9243x |
| util.invert | float32 | 1024² | standard | default | 0.4350x |
| util.img_as_float64 | float32 | 1024² | standard | default | 0.9531x |
| util.img_as_float | bool | 1024² | standard | default | 0.9405x |
| util.img_as_int | uint16 | 1024² | standard | default | 0.5760x |
| util.img_as_bool | uint16 | 1024² | standard | default | 0.7151x |
| util.img_as_ubyte | uint64 | 1024² | standard | default | 0.9861x |
| util.img_as_bool | uint64 | 1024² | standard | default | 0.6629x |
| color.rgb2hed | uint16 | 512² | strided | default | 0.8506x |
| color.separate_stains | uint16 | 512² | channel-first | default | 0.8227x |
| color.rgb2hed | uint16 | 512² | channel-first | default | 0.8467x |
| morphology.opening | uint8 | 512² | rectangle-31 | rectangle-31 | 0.9925x |
| morphology.white_tophat | float32 | 512² | rectangle-31 | rectangle-31 | 0.9777x |
| morphology.white_tophat | float64 | 512² | rectangle-31 | rectangle-31 | 0.9992x |
| filters.threshold_isodata | float32/float32 | 257 | histogram-only | supplied-histogram | 0.9013x |
| filters.threshold_yen | float32/float64 | 257 | histogram-only | supplied-histogram | 0.9321x |
| filters.threshold_isodata | float32/float64 | 257 | histogram-only | supplied-histogram | 0.8536x |
| filters.threshold_isodata | float32/float64 | 257 | histogram-only | trimmed-histogram | 0.8986x |
| filters.farid_v | float32 | 1×512 | single-row | single-row | 0.3360x |
| filters.farid | float64 | 512×512 | strided | strided | 0.8923x |
| filters.farid | float32 | 512×512 | contiguous | axis=() | 0.7221x |
| util.img_as_bool | float32 | 512² | dtype-metadata | default | 0.9999x |
| util.invert | float64 | 512² | dtype-metadata | default | 0.9425x |
| filters.difference_of_gaussians | uint32 | 512×512 | fortran | fortran | 0.5746x |
| filters.difference_of_gaussians | uint8 | 512×1 | single-column | single-column | 0.8125x |
| filters.difference_of_gaussians | float32 | 512×1 | single-column | single-column | 0.9368x |
| filters.difference_of_gaussians | float64 | 1×512 | single-row | single-row | 0.9976x |

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

Measurements use ordinary system load without CPU affinity or frequency locking. Small differences near 1x can change between runs. These are measured comparisons on this machine, not universal speed guarantees. CI runs parity and benchmark smoke checks on Windows/Linux without asserting speed ratios on shared runners.

Coverage remains partial: 90 native image entries, 2 metadata helpers, 358 inventoried entries still pending. See the [inventory](../coverage/README.md), [roadmap](../coverage/ROADMAP.md) and [documented input contracts](../README.md). Run `python benchmarks/verify.py` to rebuild and reproduce all fifteen reports, including this summary. Use `--threads 1` for serial-only verification. The summary rejects mixed source hashes, missing APIs and absent one-worker comparisons.
