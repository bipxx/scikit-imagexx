# Last completed whole-library performance checkpoint

Retained padded `feature.hessian_matrix`, measured source `f24bc39`:
[full CI39895498410](https://github.com/bipxx/scikit-imagexx/actions/runs/39895498410)
passed Windows, Linux and minimum NumPy; isolated
[measurement39895498481](https://github.com/bipxx/scikit-imagexx/actions/runs/39895498481)
passed both OSes. One-worker geometric means for finite/Gaussian modes are
**4.553658× / 4.657071× Linux** and **3.787322× / 4.577836× Windows**.
All 240 one-worker cases are faster on each OS, including the twelve previously
slower small-volume Windows cases. All 960 timed rows have zero numerical
difference. The same-source [shape control39895501195](https://github.com/bipxx/scikit-imagexx/actions/runs/39895501195)
also passes all 120 one-worker cases per OS. Native coverage remains
**116 + 2 metadata APIs**, with **337 callable names** and 29 classes/284 static
member paths pending. [Raw samples, validation and trial history](hessian-initial.md)
are preserved.

Retained `feature.shape_index`, measured source `fc32f93`:
[full CI39891161029](https://github.com/bipxx/scikit-imagexx/actions/runs/39891161029)
passed Windows, Linux and minimum NumPy; isolated
[measurement39891161346](https://github.com/bipxx/scikit-imagexx/actions/runs/39891161346)
passed both OSes. One-worker geometric means are **3.787195× Linux / 2.689617×
Windows**, all 120 cases faster per OS; minima 2.321666× / 1.714554×.
All 480 timed rows pass NaN-mask and numerical parity. Maximum finite absolute
errors are 1.1920929e-7 float32 and 2.2204461e-16 float64. Native coverage is
**116 + 2 metadata APIs**; **337 callable names** and 29 classes/284 static
member paths remain pending. [Raw samples and scope](shape-index-initial.md)
are preserved. The subsequent padded Hessian retention is recorded above.

Retained `feature.hessian_matrix`, measured source `5f010a6`:
[full CI39890585539](https://github.com/bipxx/scikit-imagexx/actions/runs/39890585539)
passed all three jobs; [measurement39890585030](https://github.com/bipxx/scikit-imagexx/actions/runs/39890585030)
passed both OSes. One-worker geometric means for finite/Gaussian modes are
**4.600456× / 2.493615× Linux** and **3.486546× / 2.070721× Windows**.
All 240 Linux cases and all 120 Windows finite cases are faster. Windows
Gaussian derivatives are faster in 108/120; twelve small 3-D cases remain
slower in this historical trial; the retained padded-line follow-up above addresses them. All 960 timed rows have
zero numerical difference. Native coverage is **115 + 2 metadata APIs**;
**338 callable names** and 29 classes/284 static member paths remain pending.
[Raw samples, validation and failed trials](hessian-initial.md) are preserved.

Retained `feature.corner_kitchen_rosenfeld`, measured source `ee733e2`:
[full CI39883029028](https://github.com/bipxx/scikit-imagexx/actions/runs/39883029028)
passed Windows, Linux and minimum NumPy; isolated
[measurement39883035493](https://github.com/bipxx/scikit-imagexx/actions/runs/39883035493)
passed both OSes. One-worker geometric means are **7.043285× Linux / 6.175798×
Windows**, all 120 cases faster per OS; minima 5.295321× / 3.294634×.
All 480 timed rows have zero numerical difference. Dedicated/native/portable/
SDK/sanitizer validation passes; [raw samples and scope](kitchen-initial.md)
are preserved. Native coverage is **114 + 2 metadata APIs**; **339 callable
names** and 29 classes/284 static member paths remain pending.

Retained `feature.corner_harris`, `corner_shi_tomasi` and `corner_foerstner`,
measured source `0513a44`: [full CI39881327501](https://github.com/bipxx/scikit-imagexx/actions/runs/39881327501)
passed Windows, Linux and minimum NumPy. Isolated [measurement39881327150](https://github.com/bipxx/scikit-imagexx/actions/runs/39881327150)
passed both OSes. One-worker geometric means by API are **3.225440× / 3.486305× /
3.590115× Linux** and **2.333236× / 2.377552× / 2.850550× Windows**.
All 264 one-worker cases are faster per OS; all 1,056 timed rows have zero
numerical difference. Dedicated/native/SDK/sanitizer validation passes;
[raw samples and scope](corner-responses-initial.md) are preserved.
Native coverage is **113 + 2 metadata APIs**; **340 callable names** and
29 classes/284 static member paths remain pending.

Retained `feature.structure_tensor`, measured source `ccab3b0`:
[full CI39877936041](https://github.com/bipxx/scikit-imagexx/actions/runs/39877936041)
passed Windows, Linux and minimum NumPy. Isolated
[measurement39877936033](https://github.com/bipxx/scikit-imagexx/actions/runs/39877936033)
passed both OSes. One-worker geometric means **2.984798× Linux / 2.263855×
Windows**, all 120 cases faster per OS; minima 1.771831×/1.502563×.
All 480 timed rows have zero numerical difference. Dedicated/native/installed
SDK/sanitizer checks pass; [raw samples and scope](structure-tensor-initial.md)
are preserved. Native image coverage is now **110 + 2 metadata APIs**, with
**343 callable names** and 29 classes/284 static member paths pending.


Retained symmetric image eigenvalues (`feature.hessian_matrix_eigvals` and
`feature.structure_tensor_eigenvalues`): measured runtime `33a0e57`, test-only
portable correction `1e1b21b`, full CI [39874772356](https://github.com/bipxx/scikit-imagexx/actions/runs/39874772356)
passed all three jobs. Isolated [measurement39872350846](https://github.com/bipxx/scikit-imagexx/actions/runs/39872350846)
passed both OSes. One-worker geometric means per API: Linux **6.374628× /
6.349021×**, Windows **5.176761× / 4.725646×**. All 60 cases faster per API/OS;
slowest corrected case 1.425715×. All 480 timed rows pass parity; float32 exact,
maximum float64 absolute error 1.066e-14. Initial and intermediate source
failures and raw samples remain in the [report](symmetric-eigen-initial.md).
This increment brings native image coverage to **109 + 2 metadata APIs**;
**344 callable names** and 29 classes/284 static member paths remain pending.


The [Gabor image filter](gabor-image-initial.md) is retained after full CI.
One-worker geometric means are 8.977x Linux /9.006x Windows over scikit-image
0.26.0; all 72 cases per OS are faster, including every real dtype. Integer and
boolean result bytes and float32 outputs are exact in this corpus. Shared
kernel controls also pass, at 3.443x/3.648x over upstream across 36 cases per OS.

The [Gabor kernel port](gabor-initial.md) is retained on main after full CI.
Corrected-source one-worker geometric means are 3.054x on Linux and 3.582x on
Windows over scikit-image 0.26.0; all 36 cases per OS are faster. Both exact
packages, the original trial and every raw sample remain available.

The [CLAHE experiments](clahe-initial.md) retain the initial native port and the
validated float-buffer reuse optimization. The latest paired run measures
one-worker geometric means of 9.786x on Linux and 14.073x on Windows over
scikit-image 0.26.0. Same-run improvement over the initial native package is
1.011x/1.013x across 40 fixed cases, with adverse controls retained.

The [multi-Otsu experiments](multiotsu-initial.md) retain one-worker geometric
means of 3.770x/5.444x over upstream after optimizing two-class thresholds.
Both ports are merged on main after full source CI.

The [median experiments](median-initial.md) preserve initial and paired trials,
including severe large-window regressions and their subsequent improvement.
The port is retained on main. The integrated paired run measures 6.999x/4.331x
over upstream on Linux/Windows; signed-zero 1-D and some tiny Windows calls
remain slower. These scoped measurements do not replace the older whole-library
checkpoint below or prove performance for every possible call.

The [retained sparse-correlation port](sparse.md) adds exact ordered products,
native row padding and optional global workers. One-worker geometric means are
1.938x over scikit-image on Linux and 2.258x on Windows; every one-worker case in
this corpus beats upstream. Against the exact initial native package, means
improve 1.105x/1.053x, with slower reflect/strided controls retained explicitly.
Initial experiments, corrected dtype/warning behavior, exact packages and all
raw samples are preserved. Scoped SIMD/scalar, minimum-runtime and native
sanitizer checks pass. Full sparse CI passes all three jobs: Linux 54,441 tests
and Windows 54,439 plus two skips in each SIMD mode, with native normal/dynamic/
portable checks passing 2/2 on both OSes.

The [retained row-span helper](hysteresis-spans.md) improves the paired one-worker
corpus 1.150x on Linux and 1.136x on Windows against the exact initial package.
Current geometric means are 2.104x/1.828x over scikit-image. The earlier inline
trial and every residual control regression remain preserved. Exact-source
scoped and minimum-runtime checks pass. Full helper CI passes all three jobs:
Linux 50,853 tests and Windows 50,851 plus two skips in each SIMD mode,
with native normal/dynamic/portable checks passing 2/2 on both OSes.

The [hysteresis parity checkpoint](hysteresis.md) adds native broadcast thresholds,
SIMD masks and optional parallel connectivity. Retained-source one-worker geometric
means are 1.73x/1.63x over scikit-image on Linux/Windows, with slower random,
strided and Fortran cases retained. Exact final-source timings and scoped SIMD,
scalar, minimum-NumPy and sanitizer checks are recorded. Full retained compatibility
CI passes all three jobs, including Linux 50,845 and Windows 50,843 plus two skips
in each SIMD mode, and normal/dynamic/portable native checks 2/2 on both OSes.

The [packed wide-range rank checkpoint](rank-packed.md) reduces sort-record
traffic and removes the full group-start array. Against the exact table package,
one-worker wide-range cases improve 1.56x on Linux and 1.23x on Windows; the
whole fixed corpus improves 1.075x/1.064x. The report preserves the revised
first trial and every control regression. General uint64/float64 sorting still
trails upstream in some cases. Scoped SIMD/scalar, minimum-runtime, portable
and sanitizer checks pass. All three follow-up CI jobs completed successfully;
Linux passes 48,657 and Windows 48,655 plus two skips in each SIMD mode,
with native normal/dynamic/portable checks passing 2/2 on both OSes.
The parent run status still lags its completed jobs.

The [rank lookup-table checkpoint](rank-tables.md) avoids sorting every pixel
for bounded-cardinality data. Against the retained uniform package, the fixed
one-worker corpus improves 1.91x on Linux and 2.10x on Windows; current geometric
means are 5.67x/3.43x over scikit-image. Dense SIMD and bounded sparse tables
retain exact ranks and representative bytes. Wide-range sorting remains an
outstanding bottleneck, and every adverse sample is retained in the report.
Full lookup-table CI passes all three jobs, including Linux 48,585 tests and
Windows 48,583 plus two skips in each SIMD mode, and normal/dynamic/portable
native checks passing 2/2 on both OSes.

The [rank-order port](rank-order.md) and [retained uniform fast path](rank-uniform.md)
add exact ranks, original-value bytes and reconstruction tests for real dtypes,
layouts, label-width boundaries and memmap results. The paired one-worker corpus
improves 1.91x on Linux and 1.58x on Windows against the initial native package;
current geometric means are 2.55x/1.67x over scikit-image. Constant images improve
strongly, while floating sorts, layout normalization and several controls remain
slower. Reports retain all cases and raw samples. Full uniform-checkpoint CI
passes all three jobs, with Linux 48,453 tests and Windows 48,451 plus two skips
in each SIMD mode. Scoped SIMD/scalar, minimum-runtime and native sanitizer
checks pass; the newer lookup-table full CI also passes all three jobs.

The [streaming adaptive-mean checkpoint](adaptive-stream.md) removes first-axis
packing and scattering. Paired one-worker mean workloads improve 25% on Linux
and 24% on Windows against the exact initial native package. Current mean
geometric means are 1.39x/1.11x original/current (Linux/Windows); every measured
mean case improves against that native baseline. Tiny-call and layout costs
remain, and the report retains all slower cases and Gaussian/median controls.

The [adaptive-threshold initial checkpoint](adaptive.md) adds native Gaussian,
mean and median `threshold_local` paths with 252 measured Windows/Linux cases.
One-worker Gaussian geometric means are 2.61x/1.68x original/current and median
5.96x/5.46x (Linux/Windows). Mean is 1.28x on Linux and 0.78x on Windows, so its
Windows throughput, layout costs and tiny-call overhead remain optimization
targets. All adverse cases and exact initial packages are retained. This is a
source-specific partial port, not a whole-library refresh.
The [overflow-policy correction and exact paired follow-up](adaptive-final.md)
records current one-worker speedups of 2.79x/1.69x Gaussian, 1.34x/0.79x mean,
and 6.20x/5.55x median (Linux/Windows), with every per-case regression retained.

The [Li constant-image fast path](li-uniform.md) is retained after paired
one-worker geometric-mean gains of 34% on Linux and 17% on Windows. It records
the remaining per-case regressions and all raw samples against the exact initial
package. These are source-specific results, not a whole-library refresh.

The [singleton-stride CIEDE2000 checkpoint](distance-singleton.md) follows with
392 focused cases passing on four NumPy/runtime replicas and paired one-worker
speedups of 2.26x over scikit-image on Linux and 2.47x on Windows. It records the
2-3% cost against the initial C++ package. Full CI run 39767948656 passes all
three jobs: Linux 44,419 tests and Windows 44,417 plus two skips in each SIMD
mode, with normal/dynamic/portable native checks passing 2/2 on both OSes.

The newer [Li threshold checkpoint](li.md) records source-specific Windows/Linux
one-worker and four-worker results, including regressions. The subsequent
[runtime angle-layout correction](distance-layout.md) addresses the eight Linux
CIEDE2000 failures recorded at that checkpoint and retains paired measurements
against the exact previous package. The first correction passes all gates in
CI run 39763182640. A later AVX-512/NumPy 1.24 runner exposed two length-128
antipodal failures. Two diagnostic runners prove that upstream itself returns
multiple arrays for identical inputs as temporary allocation changes; the
antipodal gate now requires one actually observed whole upstream array with
unchanged numerical budgets. Its subsequent singleton-stride gates pass. Both
paired performance jobs pass.
The whole-library reports below remain historical.

These 21 reports were measured at `984141f`, before the conversion layout optimization, legacy color-distance corrections and histogram-matching port/tuning. The [conversion experiment](conversion-packet-tuning.md), [scalar-distance report](distance-scalar.md), [CIEDE2000 report](ciede2000.md), [histogram-matching port](match-histograms.md) and [rank-storage tuning](match-memory.md) record their separate source-specific measurements; the whole-library source set remains historical until the next full refresh.

Reference: original scikit-image 0.26.0. 3227 comparisons on Windows-11-10.0.26200-SP0; CPU: Intel64 Family 6 Model 151 Stepping 2, GenuineIntel. Individual reports record timestamps, CPU, versions, raw samples and options. All reports match the checkpoint source/build SHA-256 (POSIX relative paths and LF line endings):
`711c18c3ad26fb69639ca61ddbd477dc67ccccb306f0b54f08b2f5b1c685f7fe`.

| Report | Cases | Worker counts |
|---|---:|---|
| [Every covered public entry](function-coverage.md) | 572 | 1 |
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
| [Every covered entry with dtype metadata](metadata-parameters.md) | 286 | 1 |
| [Gaussian families and output precision](gaussian-parameters.md) | 225 | 1 |
| [Integral-image dtypes and layouts](integral-parameters.md) | 148 | 1 |
| [Integral-window batches and layouts](integrate-parameters.md) | 144 | 1 |
| [Raw local-block means and layouts](downscale-parameters.md) | 120 | 1 |
| [Low-contrast percentiles and layouts](contrast-parameters.md) | 126 | 1 |
| [Minimum threshold smoothing and convergence](minimum-parameters.md) | 39 | 1 |
| [Label ranks, overlays and regional averages](label-color-parameters.md) | 101 | 1 |

Every workload checks values and dtype against original scikit-image before timing with fallback disabled. Integer-width conversion cases also require exact values and matching identity behavior. Timings include allocation, Python dispatch, integer normalization and any needed input copying. Inputs/imports/I/O are excluded. Reference thread pools use one thread. Native worker counts are explicit.

Examples from the 1024² one-worker default-parameter comparisons (color-space routing is RGB to HSV; CIE76 compares raw Lab values with a channel-reversed second input):

| Operation | Input | Original ms | C++ ms | Reference / C++ |
|---|---|---:|---:|---:|
| morphology.erosion | uint8 | 11.2898 | 0.6186 | 18.25x |
| color.rgb2yuv | uint8 | 16.6304 | 8.7286 | 1.91x |
| color.rgb2xyz | uint8 | 94.3592 | 8.9784 | 10.51x |
| color.rgb2hed | uint8 | 42.2120 | 12.3069 | 3.43x |
| filters.gaussian | uint8 | 22.3002 | 8.3198 | 2.68x |
| filters.sobel | uint8 | 36.8091 | 4.0323 | 9.13x |
| filters.threshold_otsu | uint8 | 3.9226 | 0.2691 | 14.58x |
| morphology.opening | float32 | 12.0966 | 4.7824 | 2.53x |
| color.rgb2xyz | float32 | 56.0160 | 22.7184 | 2.47x |
| color.xyz2rgb | float32 | 79.3003 | 24.6809 | 3.21x |
| color.rgb2hsv | float32 | 167.5777 | 5.2223 | 32.09x |
| color.hsv2rgb | float32 | 120.9493 | 4.4720 | 27.05x |
| color.rgb2luv | float32 | 95.8037 | 32.8901 | 2.91x |
| color.luv2rgb | float32 | 49.5442 | 4.2125 | 11.76x |
| color.separate_stains | float32 | 29.7085 | 12.8916 | 2.30x |
| color.deltaE_cie76 | float32 | 8.1693 | 1.6462 | 4.96x |
| color.convert_colorspace | float32 | 149.9372 | 3.9863 | 37.61x |
| color.deltaE_ciede94 | float32 | 111.5435 | 20.4267 | 5.46x |
| color.deltaE_ciede2000 | float32 | 205.1118 | 99.3039 | 2.07x |
| color.deltaE_cmc | float32 | 256.5354 | 49.3461 | 5.20x |
| morphology.binary_dilation | float64 | 12.7392 | 4.9029 | 2.60x |
| color.rgb2ycbcr | float64 | 14.2688 | 5.9957 | 2.38x |
| color.rgb2lab | float64 | 201.1017 | 53.4861 | 3.76x |
| color.lab2lch | float64 | 45.6124 | 15.6008 | 2.92x |
| color.luv2xyz | float64 | 96.6012 | 10.8206 | 8.93x |
| color.combine_stains | float64 | 44.5408 | 12.4011 | 3.59x |
| color.deltaE_cie76 | float64 | 24.8272 | 4.0699 | 6.10x |

Boolean 31×31 rectangles, with one worker (the previous slower morphology workloads):

| Operation | Image | Original ms | C++ ms | Reference / C++ |
|---|---|---:|---:|---:|
| morphology.erosion | 512² | 2.3189 | 0.1749 | 13.26x |
| morphology.dilation | 512² | 2.3914 | 0.1559 | 15.33x |
| morphology.opening | 512² | 4.0061 | 0.2484 | 16.12x |
| morphology.closing | 512² | 3.9194 | 0.3272 | 11.98x |
| morphology.white_tophat | 512² | 4.0038 | 0.3066 | 13.06x |
| morphology.black_tophat | 512² | 3.9299 | 0.3521 | 11.16x |

The [fixed-workload tuning experiment](morphology-tuning.md) preserves separate historical before/after C++ snapshots. Its cases are not added to this current-source total.

3157 of 3188 one-worker native processing workloads measured faster than the reference. These are workload cases, including repeated functions and parameter variants. Identity returns, copies and dtype metadata are excluded from this count and labeled separately in the full reports.

31 one-worker processing workloads remained slower:

| Operation | Input | Shape | Layout | Workload | Reference / C++ |
|---|---|---:|---|---|---:|
| util.img_as_bool | float32 | 256² | standard | default | 0.9608x |
| util.img_as_bool | float64 | 256² | standard | default | 0.7073x |
| util.invert | float64 | 1024² | standard | default | 0.9917x |
| util.img_as_bool | float64 | 1024² | standard | default | 0.8321x |
| util.img_as_float | bool | 1024² | standard | default | 0.9076x |
| util.img_as_float64 | bool | 1024² | standard | default | 0.9583x |
| util.img_as_int | uint16 | 1024² | standard | default | 0.5630x |
| util.img_as_bool | int16 | 1024² | standard | default | 0.9310x |
| util.img_as_bool | uint32 | 1024² | standard | default | 0.8529x |
| util.img_as_bool | uint64 | 1024² | standard | default | 0.7728x |
| util.img_as_bool | int64 | 1024² | standard | default | 0.9145x |
| color.rgb2ypbpr | uint16 | 512² | strided | default | 0.7068x |
| morphology.black_tophat | float32 | 512² | rectangle-31 | rectangle-31 | 0.8487x |
| filters.threshold_yen | float32/int64 | 257 | histogram-only | supplied-histogram | 0.9059x |
| filters.threshold_isodata | float32/int64 | 257 | histogram-only | supplied-histogram | 0.7982x |
| filters.threshold_yen | float32/int64 | 257 | histogram-only | trimmed-histogram | 0.8101x |
| filters.threshold_yen | float32/int64 | 257 | histogram-only | large-counts | 0.5751x |
| filters.threshold_isodata | float32/int64 | 257 | histogram-only | large-counts | 0.9514x |
| filters.threshold_yen | float32/float32 | 257 | histogram-only | supplied-histogram | 0.5368x |
| filters.threshold_isodata | float32/float32 | 257 | histogram-only | supplied-histogram | 0.9842x |
| filters.threshold_yen | float32/float32 | 257 | histogram-only | trimmed-histogram | 0.9387x |
| filters.threshold_isodata | float32/float32 | 257 | histogram-only | trimmed-histogram | 0.7897x |
| filters.threshold_yen | float32/float32 | 257 | histogram-only | large-counts | 0.8682x |
| filters.threshold_isodata | float32/float32 | 257 | histogram-only | large-counts | 0.8591x |
| util.invert | float64 | 512² | dtype-metadata | default | 0.9887x |
| util.img_as_bool | float64 | 512² | dtype-metadata | default | 0.8376x |
| filters.gaussian | int64 | 512×512 | fortran | fortran | 0.8788x |
| filters.gaussian | float32 | 512×1 | single-column | single-column | 0.9023x |
| filters.difference_of_gaussians | float64 | 512×1 | single-column | single-column | 0.8651x |
| transform.integral_image | uint16 | 262144 | line | line | 0.9654x |
| transform.integral_image | float64 | 512×1 | single-column | single-column | 0.8124x |

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

This aggregate checkpoint covers 96 native image entries and 2 metadata helpers; it predates the later dedicated reports above. Current implementation and pending counts are in the generated inventory. See the [inventory](../coverage/README.md), [roadmap](../coverage/ROADMAP.md) and [documented input contracts](../README.md). Run `python benchmarks/verify.py` to rebuild and reproduce all 21 reports, including this summary. Use `--threads 1` for serial-only verification. The summary rejects mixed source hashes, missing APIs and absent one-worker comparisons.
