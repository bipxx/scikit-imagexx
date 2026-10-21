# Current performance verification

Reference: original scikit-image 0.26.0. 1640 comparisons on Windows-11-10.0.26200-SP0; CPU: Intel64 Family 6 Model 151 Stepping 2, GenuineIntel. Individual reports record timestamps, CPU, versions, raw samples and options. All reports match the current source SHA-256:
`718948c5461338d7bca310bbf7837e98f653860d4e9300e13474707175c570f4`.

| Report | Cases | Worker counts |
|---|---:|---|
| [Every covered public entry](function-coverage.md) | 844 | 1, 4 |
| [Additional integer widths](conversion-widths.md) | 112 | 1, 4 |
| [Non-default exposure parameters](exposure-parameters.md) | 24 | 1 |
| [Color layouts and uint16 inputs](color-layouts.md) | 120 | 1 |
| [Perceptual color parameters](perceptual-parameters.md) | 60 | 1, 4 |
| [Morphology footprints and output reuse](morphology-parameters.md) | 240 | 1, 4 |
| [Stain matrices and output widths](stain-parameters.md) | 156 | 1, 4 |
| [Lab distances and color routes](color-distance.md) | 84 | 1, 4 |

Every workload checks values and dtype against original scikit-image before timing with fallback disabled. Integer-width conversion cases also require exact values and matching identity behavior. Timings include allocation, Python dispatch, integer normalization and any needed input copying. Inputs/imports/I/O are excluded. Reference thread pools use one thread. Native worker counts are explicit.

Examples from the 1024² one-worker default-parameter comparisons (color-space routing is RGB to HSV; CIE76 compares raw Lab values with a channel-reversed second input):

| Operation | Input | Original ms | C++ ms | Reference / C++ |
|---|---|---:|---:|---:|
| morphology.erosion | uint8 | 10.3965 | 0.4389 | 23.69x |
| color.rgb2yuv | uint8 | 15.1237 | 8.1384 | 1.86x |
| color.rgb2xyz | uint8 | 83.9495 | 8.3001 | 10.11x |
| color.rgb2hed | uint8 | 39.5715 | 12.9589 | 3.05x |
| filters.gaussian | uint8 | 26.8482 | 6.3948 | 4.20x |
| filters.sobel | uint8 | 24.2495 | 3.3152 | 7.31x |
| filters.threshold_otsu | uint8 | 3.3267 | 0.2486 | 13.38x |
| morphology.opening | float32 | 8.3598 | 2.4389 | 3.43x |
| color.rgb2xyz | float32 | 51.5964 | 20.5693 | 2.51x |
| color.xyz2rgb | float32 | 74.9917 | 26.6154 | 2.82x |
| color.rgb2hsv | float32 | 157.0306 | 4.2483 | 36.96x |
| color.hsv2rgb | float32 | 121.3944 | 4.4396 | 27.34x |
| color.rgb2luv | float32 | 91.7892 | 30.9030 | 2.97x |
| color.luv2rgb | float32 | 53.2397 | 5.1532 | 10.33x |
| color.separate_stains | float32 | 25.9088 | 14.0390 | 1.85x |
| color.deltaE_cie76 | float32 | 6.9083 | 1.8734 | 3.69x |
| color.convert_colorspace | float32 | 142.5505 | 3.8107 | 37.41x |
| morphology.binary_dilation | float64 | 12.3542 | 5.0460 | 2.45x |
| color.rgb2ycbcr | float64 | 13.3560 | 5.9927 | 2.23x |
| color.rgb2lab | float64 | 167.8351 | 51.0019 | 3.29x |
| color.lab2lch | float64 | 34.0398 | 13.8053 | 2.47x |
| color.luv2xyz | float64 | 90.2978 | 8.1688 | 11.05x |
| color.combine_stains | float64 | 39.7044 | 11.9444 | 3.32x |
| color.deltaE_cie76 | float64 | 25.5270 | 4.0526 | 6.30x |

Boolean 31×31 rectangles, with one worker (the previous slower morphology workloads):

| Operation | Image | Original ms | C++ ms | Reference / C++ |
|---|---|---:|---:|---:|
| morphology.erosion | 512² | 2.2686 | 0.1435 | 15.81x |
| morphology.dilation | 512² | 3.0805 | 0.1731 | 17.79x |
| morphology.opening | 512² | 3.8183 | 0.3559 | 10.73x |
| morphology.closing | 512² | 3.4742 | 0.2211 | 15.71x |
| morphology.white_tophat | 512² | 3.7759 | 0.2630 | 14.35x |
| morphology.black_tophat | 512² | 4.0530 | 0.3013 | 13.45x |

The [fixed-workload tuning experiment](morphology-tuning.md) preserves separate historical before/after C++ snapshots. Its cases are not added to this current-source total.

845 of 864 one-worker native processing workloads measured faster than the reference. These are workload cases, including repeated functions and parameter variants. Identity returns, copies and dtype metadata are excluded from this count and labeled separately in the full reports.

19 one-worker processing workloads remained slower:

| Operation | Input | Size | Layout | Reference / C++ |
|---|---|---:|---|---:|
| util.img_as_bool | uint8 | 256² | standard | 0.6891x |
| util.invert | float32 | 256² | standard | 0.8152x |
| util.img_as_bool | float32 | 256² | standard | 0.8592x |
| util.img_as_bool | float64 | 256² | standard | 0.5999x |
| util.invert | float32 | 1024² | standard | 0.9442x |
| util.img_as_float64 | float32 | 1024² | standard | 0.9638x |
| color.rgb2gray | float64 | 1024² | standard | 0.9776x |
| util.img_as_float64 | bool | 1024² | standard | 0.9659x |
| util.img_as_bool | int8 | 1024² | standard | 0.8795x |
| util.img_as_bool | uint16 | 1024² | standard | 0.8378x |
| util.img_as_bool | int16 | 1024² | standard | 0.8245x |
| util.img_as_bool | int32 | 1024² | standard | 0.9641x |
| util.img_as_bool | uint64 | 1024² | standard | 0.8958x |
| color.separate_stains | uint16 | 512² | strided | 0.8385x |
| color.rgb2hed | uint16 | 512² | strided | 0.8830x |
| color.separate_stains | uint16 | 512² | channel-first | 0.7684x |
| color.rgb2hed | uint16 | 512² | channel-first | 0.9816x |
| color.lch2lab | float32 | 512² | standard | 0.9510x |
| color.deltaE_cie76 | float32 | 512² | strided | 0.6103x |

The color port fuses integer normalization, matrix arithmetic and sRGB transfer. Uniform floating layouts now avoid a full contiguous copy: planar input uses contiguous channel loads, other uniform strides use gathers, and scalar tails preserve the same arithmetic. General nonuniform layouts and unaligned arrays can still require copying. Simple matrix transforms remain sensitive to dispatch, shuffling and memory traffic; this run has no hardware-counter profiling to quantify their individual contribution. Further optimization remains necessary.

Morphology reuses row mappings across border pixels and stops binary reductions when the result is known for every SIMD lane. Large rectangular footprints use linear-work prefix/suffix scans; small input types use narrower working buffers when conversion preserves extrema, while 64-bit integer scans retain double buffers for upstream rounding. Contiguous scan outputs and top-hat subtraction use SIMD. Canonical 2D boolean rectangles now use packed words, with SIMD packing, bitwise extrema and unpacking, avoiding the previous byte-wise scan passes. Other storage and boundary constants retain the general native path.

Luv conversions fuse sRGB/XYZ stages with vector chromaticity arithmetic. Cached three-value white metadata preserves NumPy dot rounding at singular denominators. Float32 inverse cube powers retain libm rounding because sRGB amplifies small XYZ differences near black; surrounding arithmetic remains vectorized. Cross-platform real-color stress tests separately bound the one-ULP luminance difference and its propagation through XYZ/sRGB; see the numerical contract.

Uniformly linear SIMD blocks skip unused sRGB power and CIE cube-root stages. The [fixed linear-color experiment](linear-color-tuning.md) retains historical before/after timings separately from this aggregate. Stain kernels fuse input normalization, logarithmic scaling, matrix arithmetic and reconstruction. Uint8 separation uses a log lookup table; floating layouts use direct loads or gathers. Float32 log values retain float32 precision before NumPy-compatible promotion, so output float64 does not imply a float64 input error budget.

CIE76 fuses raw Lab conversion, subtraction, squared components and square root without temporary image arrays. Uniform floating layouts use direct loads or gathers; integer/mixed blocks convert raw values before vector arithmetic. Color-space routing preserves the upstream intermediate RGB allocation and dtype rounding. Its parameter report includes all stage checks and propagated bounds for ill-conditioned HSV ratios.

The [fixed CIE76 integer-load experiment](distance-tuning.md) preserves historical before/after snapshots separately from these current-source totals.

Measurements use ordinary system load without CPU affinity or frequency locking. Small differences near 1x can change between runs. These are measured comparisons on this machine, not universal speed guarantees. CI runs parity and benchmark smoke checks on Windows/Linux without asserting speed ratios on shared runners.

Coverage remains partial: 71 native image entries, 2 metadata helpers, 377 inventoried entries still pending. See the [inventory](../coverage/README.md), [roadmap](../coverage/ROADMAP.md) and [documented input contracts](../README.md). Run `python benchmarks/verify.py` to rebuild and reproduce all eight reports, including this summary. Use `--threads 1` for serial-only verification. The summary rejects mixed source hashes, missing APIs and absent one-worker comparisons.
