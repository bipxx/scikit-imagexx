# Current performance verification

Reference: original scikit-image 0.26.0. 2252 comparisons on Windows-11-10.0.26200-SP0; CPU: Intel64 Family 6 Model 151 Stepping 2, GenuineIntel. Individual reports record timestamps, CPU, versions, raw samples and options. All reports match the current source/build SHA-256 (POSIX relative paths and LF line endings):
`96f542f04192c2d5c62723a80cca30713bd076f96d8afa6a5935afc6c4c24edb`.

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
| [Alpha and grayscale layouts](alpha-parameters.md) | 208 | 1 |
| [Laplace and Roberts layouts](edge-parameters.md) | 260 | 1 |
| [Farid derivatives and boundaries](farid-parameters.md) | 243 | 1 |
| [Every covered entry with dtype metadata](metadata-parameters.md) | 262 | 1 |

Every workload checks values and dtype against original scikit-image before timing with fallback disabled. Integer-width conversion cases also require exact values and matching identity behavior. Timings include allocation, Python dispatch, integer normalization and any needed input copying. Inputs/imports/I/O are excluded. Reference thread pools use one thread. Native worker counts are explicit.

Examples from the 1024² one-worker default-parameter comparisons (color-space routing is RGB to HSV; CIE76 compares raw Lab values with a channel-reversed second input):

| Operation | Input | Original ms | C++ ms | Reference / C++ |
|---|---|---:|---:|---:|
| morphology.erosion | uint8 | 10.2172 | 0.4965 | 20.58x |
| color.rgb2yuv | uint8 | 15.2751 | 7.8178 | 1.95x |
| color.rgb2xyz | uint8 | 96.1257 | 9.6071 | 10.01x |
| color.rgb2hed | uint8 | 53.2776 | 13.8912 | 3.84x |
| filters.gaussian | uint8 | 27.7059 | 5.6874 | 4.87x |
| filters.sobel | uint8 | 29.2850 | 3.3998 | 8.61x |
| filters.threshold_otsu | uint8 | 4.0720 | 0.2738 | 14.87x |
| morphology.opening | float32 | 8.8258 | 2.9886 | 2.95x |
| color.rgb2xyz | float32 | 59.2840 | 21.7887 | 2.72x |
| color.xyz2rgb | float32 | 76.9060 | 21.9079 | 3.51x |
| color.rgb2hsv | float32 | 153.6735 | 4.1910 | 36.67x |
| color.hsv2rgb | float32 | 155.7656 | 5.0133 | 31.07x |
| color.rgb2luv | float32 | 113.9963 | 31.7356 | 3.59x |
| color.luv2rgb | float32 | 65.0051 | 5.6718 | 11.46x |
| color.separate_stains | float32 | 32.0296 | 12.8709 | 2.49x |
| color.deltaE_cie76 | float32 | 12.7621 | 1.8393 | 6.94x |
| color.convert_colorspace | float32 | 209.9960 | 7.1688 | 29.29x |
| color.deltaE_ciede94 | float32 | 153.3195 | 22.0744 | 6.95x |
| color.deltaE_ciede2000 | float32 | 267.0704 | 101.8211 | 2.62x |
| color.deltaE_cmc | float32 | 415.5031 | 63.9440 | 6.50x |
| morphology.binary_dilation | float64 | 12.6095 | 6.2238 | 2.03x |
| color.rgb2ycbcr | float64 | 17.7696 | 6.3024 | 2.82x |
| color.rgb2lab | float64 | 218.8060 | 53.0567 | 4.12x |
| color.lab2lch | float64 | 38.5944 | 13.4208 | 2.88x |
| color.luv2xyz | float64 | 104.5170 | 6.5203 | 16.03x |
| color.combine_stains | float64 | 44.9583 | 12.2013 | 3.68x |
| color.deltaE_cie76 | float64 | 35.0073 | 5.5338 | 6.33x |

Boolean 31×31 rectangles, with one worker (the previous slower morphology workloads):

| Operation | Image | Original ms | C++ ms | Reference / C++ |
|---|---|---:|---:|---:|
| morphology.erosion | 512² | 2.3291 | 0.1689 | 13.79x |
| morphology.dilation | 512² | 2.1698 | 0.1597 | 13.59x |
| morphology.opening | 512² | 3.7620 | 0.2673 | 14.07x |
| morphology.closing | 512² | 4.0665 | 0.3100 | 13.12x |
| morphology.white_tophat | 512² | 4.2125 | 0.3000 | 14.04x |
| morphology.black_tophat | 512² | 4.2966 | 0.2918 | 14.72x |

The [fixed-workload tuning experiment](morphology-tuning.md) preserves separate historical before/after C++ snapshots. Its cases are not added to this current-source total.

2183 of 2213 one-worker native processing workloads measured faster than the reference. These are workload cases, including repeated functions and parameter variants. Identity returns, copies and dtype metadata are excluded from this count and labeled separately in the full reports.

30 one-worker processing workloads remained slower:

| Operation | Input | Shape | Layout | Workload | Reference / C++ |
|---|---|---:|---|---|---:|
| color.rgb2ypbpr | uint8 | 256² | standard | default | 0.9678x |
| filters.threshold_mean | float32 | 256² | standard | default | 0.8220x |
| util.invert | float32 | 256² | standard | default | 0.7059x |
| util.img_as_bool | float64 | 256² | standard | default | 0.5716x |
| util.img_as_bool | uint8 | 1024² | standard | default | 0.8208x |
| util.img_as_float64 | float32 | 1024² | standard | default | 0.8920x |
| util.img_as_bool | float32 | 1024² | standard | default | 0.9335x |
| util.img_as_float32 | float64 | 1024² | standard | default | 0.8257x |
| util.img_as_float | bool | 1024² | standard | default | 0.9756x |
| util.img_as_float64 | bool | 1024² | standard | default | 0.9431x |
| util.img_as_int | uint16 | 1024² | standard | default | 0.7859x |
| util.img_as_bool | uint16 | 1024² | standard | default | 0.8447x |
| util.img_as_float | uint32 | 1024² | standard | default | 0.9327x |
| util.img_as_int | uint64 | 1024² | standard | default | 0.7590x |
| color.lch2lab | uint16 | 512² | strided | default | 0.7970x |
| color.separate_stains | uint16 | 512² | channel-first | default | 0.8433x |
| color.rgb2hed | uint16 | 512² | channel-first | default | 0.8535x |
| morphology.dilation | float32 | 512² | rectangle-31 | rectangle-31 | 0.9406x |
| morphology.closing | float32 | 512² | rectangle-31 | rectangle-31 | 0.9652x |
| morphology.white_tophat | float32 | 512² | rectangle-31 | rectangle-31 | 0.9276x |
| morphology.black_tophat | float64 | 512² | rectangle-31 | rectangle-31 | 0.8668x |
| filters.threshold_yen | float32/int64 | 257 | histogram-only | trimmed-histogram | 0.4511x |
| filters.threshold_isodata | float32/float32 | 257 | histogram-only | trimmed-histogram | 0.7516x |
| filters.threshold_yen | float32/float32 | 257 | histogram-only | large-counts | 0.9509x |
| filters.threshold_yen | float32/float64 | 257 | histogram-only | large-counts | 0.8986x |
| color.gray2rgb | int16 | 512×512 | fortran | fortran | 0.9767x |
| filters.farid | float32 | 512×512 | contiguous | axis=() | 0.8403x |
| util.invert | float32 | 512² | dtype-metadata | default | 0.9606x |
| util.invert | float64 | 512² | dtype-metadata | default | 0.8587x |
| util.img_as_bool | float64 | 512² | dtype-metadata | default | 0.9146x |

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

Coverage remains partial: 88 native image entries, 2 metadata helpers, 360 inventoried entries still pending. See the [inventory](../coverage/README.md), [roadmap](../coverage/ROADMAP.md) and [documented input contracts](../README.md). Run `python benchmarks/verify.py` to rebuild and reproduce all fourteen reports, including this summary. Use `--threads 1` for serial-only verification. The summary rejects mixed source hashes, missing APIs and absent one-worker comparisons.
