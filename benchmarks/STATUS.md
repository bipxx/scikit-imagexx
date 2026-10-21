# Current performance verification

Reference: original scikit-image 0.26.0. 1532 comparisons on Windows-11-10.0.26200-SP0; CPU: Intel64 Family 6 Model 151 Stepping 2, GenuineIntel. Individual reports record timestamps, CPU, versions, raw samples and options. All reports match the current source SHA-256:
`edca58d36b30dc36ee173aaa4c7b8048df9c247434451766c95e8a28507c8b32`.

| Report | Cases | Worker counts |
|---|---:|---|
| [Every covered public entry](function-coverage.md) | 820 | 1, 4 |
| [Additional integer widths](conversion-widths.md) | 112 | 1, 4 |
| [Non-default exposure parameters](exposure-parameters.md) | 24 | 1 |
| [Color layouts and uint16 inputs](color-layouts.md) | 120 | 1 |
| [Perceptual color parameters](perceptual-parameters.md) | 60 | 1, 4 |
| [Morphology footprints and output reuse](morphology-parameters.md) | 240 | 1, 4 |
| [Stain matrices and output widths](stain-parameters.md) | 156 | 1, 4 |

Every workload checks values and dtype against original scikit-image before timing with fallback disabled. Integer-width conversion cases also require exact values and matching identity behavior. Timings include allocation, Python dispatch, integer normalization and any needed input copying. Inputs/imports/I/O are excluded. Reference thread pools use one thread. Native worker counts are explicit.

Examples from the 1024² one-worker default-parameter comparisons:

| Operation | Input | Original ms | C++ ms | Reference / C++ |
|---|---|---:|---:|---:|
| morphology.erosion | uint8 | 10.4561 | 0.4777 | 21.89x |
| color.rgb2yuv | uint8 | 14.7929 | 8.3046 | 1.78x |
| color.rgb2xyz | uint8 | 81.3631 | 8.1686 | 9.96x |
| color.rgb2hed | uint8 | 39.6842 | 12.6615 | 3.13x |
| filters.gaussian | uint8 | 25.8962 | 5.4735 | 4.73x |
| filters.sobel | uint8 | 26.0192 | 3.3667 | 7.73x |
| filters.threshold_otsu | uint8 | 3.6166 | 0.2913 | 12.42x |
| morphology.opening | float32 | 9.1738 | 3.2464 | 2.83x |
| color.rgb2xyz | float32 | 51.2585 | 20.5697 | 2.49x |
| color.xyz2rgb | float32 | 70.3887 | 21.0628 | 3.34x |
| color.rgb2hsv | float32 | 176.1978 | 5.8644 | 30.05x |
| color.hsv2rgb | float32 | 120.3327 | 4.2897 | 28.05x |
| color.rgb2luv | float32 | 92.4519 | 31.3952 | 2.94x |
| color.luv2rgb | float32 | 42.8274 | 4.8691 | 8.80x |
| color.separate_stains | float32 | 33.0959 | 13.7061 | 2.41x |
| morphology.binary_dilation | float64 | 11.6680 | 4.5259 | 2.58x |
| color.rgb2ycbcr | float64 | 13.0502 | 5.7702 | 2.26x |
| color.rgb2lab | float64 | 175.5883 | 52.9188 | 3.32x |
| color.lab2lch | float64 | 40.2455 | 12.9311 | 3.11x |
| color.luv2xyz | float64 | 86.1342 | 7.0948 | 12.14x |
| color.combine_stains | float64 | 39.8348 | 12.3748 | 3.22x |

Boolean 31×31 rectangles, with one worker (the previous slower morphology workloads):

| Operation | Image | Original ms | C++ ms | Reference / C++ |
|---|---|---:|---:|---:|
| morphology.erosion | 512² | 2.1966 | 0.1561 | 14.07x |
| morphology.dilation | 512² | 2.3842 | 0.1639 | 14.55x |
| morphology.opening | 512² | 3.8970 | 0.2519 | 15.47x |
| morphology.closing | 512² | 4.4205 | 0.2612 | 16.92x |
| morphology.white_tophat | 512² | 3.4979 | 0.2377 | 14.71x |
| morphology.black_tophat | 512² | 3.5013 | 0.2343 | 14.94x |

The [fixed-workload tuning experiment](morphology-tuning.md) preserves separate historical before/after C++ snapshots. Its cases are not added to this current-source total.

795 of 813 one-worker native processing workloads measured faster than the reference. These are workload cases, including repeated functions and parameter variants. Identity returns, copies and dtype metadata are excluded from this count and labeled separately in the full reports.

18 one-worker processing workloads remained slower:

| Operation | Input | Size | Layout | Reference / C++ |
|---|---|---:|---|---:|
| color.rgb2ydbdr | uint8 | 256² | standard | 0.7365x |
| util.img_as_bool | uint8 | 256² | standard | 0.7665x |
| util.img_as_bool | float32 | 256² | standard | 0.9851x |
| color.rgb2gray | float64 | 256² | standard | 0.9615x |
| util.img_as_bool | float64 | 256² | standard | 0.7329x |
| util.invert | float32 | 1024² | standard | 0.8463x |
| util.img_as_float64 | float32 | 1024² | standard | 0.9477x |
| util.img_as_bool | float32 | 1024² | standard | 0.9149x |
| util.img_as_bool | float64 | 1024² | standard | 0.8268x |
| util.img_as_float64 | bool | 1024² | standard | 0.9518x |
| util.img_as_int | uint16 | 1024² | standard | 0.6919x |
| util.img_as_bool | uint32 | 1024² | standard | 0.8140x |
| util.img_as_bool | int32 | 1024² | standard | 0.7065x |
| util.img_as_bool | uint64 | 1024² | standard | 0.8607x |
| color.rgb2hed | uint16 | 512² | strided | 0.9074x |
| color.separate_stains | uint16 | 512² | channel-first | 0.8045x |
| color.rgb2hed | uint16 | 512² | channel-first | 0.8784x |
| morphology.black_tophat | float32 | 512² | rectangle-31 | 0.9900x |

The color port fuses integer normalization, matrix arithmetic and sRGB transfer. Uniform floating layouts now avoid a full contiguous copy: planar input uses contiguous channel loads, other uniform strides use gathers, and scalar tails preserve the same arithmetic. General nonuniform layouts and unaligned arrays can still require copying. Simple matrix transforms remain sensitive to dispatch, shuffling and memory traffic; this run has no hardware-counter profiling to quantify their individual contribution. Further optimization remains necessary.

Morphology reuses row mappings across border pixels and stops binary reductions when the result is known for every SIMD lane. Large rectangular footprints use linear-work prefix/suffix scans; small input types use narrower working buffers when conversion preserves extrema, while 64-bit integer scans retain double buffers for upstream rounding. Contiguous scan outputs and top-hat subtraction use SIMD. Canonical 2D boolean rectangles now use packed words, with SIMD packing, bitwise extrema and unpacking, avoiding the previous byte-wise scan passes. Other storage and boundary constants retain the general native path.

Luv conversions fuse sRGB/XYZ stages with vector chromaticity arithmetic. Cached three-value white metadata preserves NumPy dot rounding at singular denominators. Float32 inverse cube powers retain libm rounding because sRGB amplifies small XYZ differences near black; surrounding arithmetic remains vectorized. Cross-platform real-color stress tests separately bound the one-ULP luminance difference and its propagation through XYZ/sRGB; see the numerical contract.

Uniformly linear SIMD blocks skip unused sRGB power and CIE cube-root stages. The [fixed linear-color experiment](linear-color-tuning.md) retains historical before/after timings separately from this aggregate. Stain kernels fuse input normalization, logarithmic scaling, matrix arithmetic and reconstruction. Uint8 separation uses a log lookup table; floating layouts use direct loads or gathers. Float32 log values retain float32 precision before NumPy-compatible promotion, so output float64 does not imply a float64 input error budget.

Measurements use ordinary system load without CPU affinity or frequency locking. Small differences near 1x can change between runs. These are measured comparisons on this machine, not universal speed guarantees. CI runs parity and benchmark smoke checks on Windows/Linux without asserting speed ratios on shared runners.

Coverage remains partial: 69 native image entries, 2 metadata helpers, 379 inventoried entries still pending. See the [inventory](../coverage/README.md), [roadmap](../coverage/ROADMAP.md) and [documented input contracts](../README.md). Run `python benchmarks/verify.py` to rebuild and reproduce all seven reports, including this summary. Use `--threads 1` for serial-only verification. The summary rejects mixed source hashes, missing APIs and absent one-worker comparisons.
