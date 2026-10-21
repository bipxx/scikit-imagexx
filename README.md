# scikit-imagexx

An incremental C++17 rewrite of scikit-image algorithms, with NumPy
bindings, numerical parity checks, explicit AVX2 SIMD, OpenMP parallelism, and
reproducible benchmarks. The native port is in progress. Python compatibility
namespaces let applications swap backends while unsupported calls use explicitly
reported upstream fallback. Fallback is compatibility, not a native rewrite.

Every native public function has measured comparisons against original scikit-image,
including **one-worker** runs. Allocation, conversion and Python dispatch are
included. See the [current verification report](benchmarks/STATUS.md) for measured
speedups, raw reports and remaining regressions on an i7-12700K.

## Supported parity surface

The reference is scikit-image **0.26.0**. There are currently **93 partially native
image functions and two metadata helpers**. The
[API inventory](coverage/README.md) tracks 455 public callable names across 18
modules, including four deprecated binary morphology functions omitted from upstream's
export list; 360 remain pending native implementation and dedicated parity tests.
Static discovery also records 284 member paths across 29 exported classes;
their numerical/state parity and runtime attributes remain pending.
This is not full scikit-image coverage.

| Function | Supported inputs and options | Numerical contract |
|---|---|---|
| `rgb2gray` | 3D RGB; uint8, float32, float64; arbitrary channel axis | Same luminance weights; uint8 normalized to float64 |
| `gray2rgb`, `gray2rgba` | Arbitrary dimensions, including scalars and empty arrays; fixed object-free storage, including float16 and complex; arbitrary output channel axis; RGBA default alpha for real image dtypes, explicit scalar or same-dtype broadcast alpha for additional storage | Exact replication and NumPy stack dtype/endian normalization; alpha casting warnings and broadcast errors preserved |
| `rgba2rgb` | Boolean; signed/unsigned 8/16/32/64-bit integers; float32/64; arbitrary dimensions with four channels and any channel axis; RGB background | Fused normalization and alpha compositing; float32 output for float32 input, float64 otherwise; clipping preserves signed zero and background NaN |
| `rgb2yuv`/`yuv2rgb`, `rgb2yiq`/`yiq2rgb`, `rgb2ypbpr`/`ypbpr2rgb`, `rgb2ycbcr`/`ycbcr2rgb`, `rgb2ydbdr`/`ydbdr2rgb`, `rgb2rgbcie`/`rgbcie2rgb` | Boolean; signed/unsigned 8/16/32/64-bit integers; float32/64; arbitrary dimensions with three channels; channel axis, including empty arrays | Upstream matrices, integer normalization and dtype-dependent YCbCr subtraction/errors; float32 output for float32 input, float64 otherwise |
| `rgb2xyz`, `xyz2rgb` | Same color input types and dimensions | Piecewise sRGB transfer, XYZ matrix, inverse clipping; uint8 decode LUT |
| `rgb2hsv`, `hsv2rgb` | Same color dtypes, arbitrary dimensions and channel axes; finite inputs and RGB-to-HSV intermediates; inverse hue sector must fit int32 | Upstream tie order, grayscale cases, signed zero and uint8-wrapped hue sectors |
| `xyz2lab`, `lab2xyz`, `rgb2lab`, `lab2rgb` | Same color dtypes/dimensions; all eight illuminants and three observers | Fused XYZ/Lab/sRGB stages; upstream negative-Z warning counts and single-pixel float32 promotion |
| `lab2lch`, `lch2lab` | Same color dtypes; three or more channels; extra channels preserved after normalization | Native hypot/atan2 and sin/cos; IEEE/special ranges use scalar libm |
| `xyz2luv`, `luv2xyz` | Same color dtypes/dimensions; three channels; all eight illuminants and three observers | Native Luv curves and epsilon-regularized denominators; dtype-specific white metadata preserves upstream dot rounding |
| `rgb2luv`, `luv2rgb` | Same color dtypes/dimensions; three channels; fixed D65/2 reference, matching upstream signatures | Fused sRGB/XYZ/Luv stages; vector arithmetic and transfer, with float32 inverse cube powers preserving libm rounding |
| `separate_stains`, `combine_stains`, `rgb2hed`, `hed2rgb` | Same color input dtypes/dimensions; three input channels; unbatched real numeric 3×K matrices for general functions; standard HED presets | Native log/exp and matrix stages; output dtype follows NumPy promotion; 11 stain-family matrix pairs available as metadata |
| `deltaE_cie76` | Boolean; signed/unsigned 8/16/32/64-bit integers; float32/64; array-like inputs; broadcastable pixel shapes; arbitrary channel axes; first three channels used, extras ignored | Fused raw Lab difference/square/sum/sqrt; no integer intensity normalization; float32 only when both inputs are float32, float64 otherwise; scalar return preserved |
| `deltaE_ciede94`, `deltaE_ciede2000`, `deltaE_cmc` | Same raw Lab input dtypes, broadcasting and extra-channel handling; finite Python int/float or NumPy float32 scalar weights; upstream channel-axis semantics | Fused weighted metrics; double hue-cancellation stage; dtype-specific intermediate rounding and promoted trigonometry; CMC returns float64 for two float32 scalar colors |
| `convert_colorspace` | All nine upstream spaces and all 81 routes; input contracts of the participating native transforms | Same route through RGB and intermediate rounding/allocation; RGB-to-RGB returns the original object without validation or processing |
| `xyz_tristimulus_values` | All upstream white-point combinations and NumPy output dtypes | Metadata helper; no image kernel |
| `gaussian` | N-D real arrays, scalars and empty images; all 12 real dtypes; endian/strided storage; scalar/per-axis sigma, channels, all five boundary modes and grid aliases, preserve-range and floating output buffers | Ordered separable convolution with double accumulation, typed-sigma kernel rounding and output precision after every axis; preserves output identity and casting warnings |
| `difference_of_gaussians` | Same real image storage; scalar/per-axis low/high sigma, channels and boundary modes | Two native blurs with upstream normalization, output dtype and subtraction stages |
| `unsharp_mask` | Finite real image working ranges; scalar/per-axis radius, finite scalar amount, preserve-range and channel axes | Native blur and fused sharpening with upstream scalar promotion, float32 rounding and clipping; exceptional arithmetic and upstream channel loops that leave output regions unwritten require explicit fallback |
| `sobel`, `prewitt`, `scharr` | 2D grayscale; same dtypes; reflected borders; magnitude or axis 0/1 | Same normalization; uint8 normalized to float64 |
| `sobel_h/v`, `prewitt_h/v`, `scharr_h/v` | 2D grayscale; same dtypes; reflected borders | Signed directional gradients |
| `farid`, `farid_h`, `farid_v` | All 12 real image dtypes, endian/native storage, arrays/memmaps and eroded numeric masks; generic N-D images up to eight dimensions, ordered/repeated/negative axes, all five boundary modes and grid aliases; directional nonempty 2D images | Ordered full convolution with float64 accumulation, float32 gradient rounding before magnitude stages, upstream dimension normalization and signed zeros; optional global workers and runtime AVX2 |
| `laplace` | One or more dimensions, including empty; all 12 real image dtypes (including float16), native/endian storage, arrays/memmaps; integer `ksize` ≥3; eroded numeric masks and mask broadcasting | Sparse reflected Laplacian in upstream coefficient order, including singleton axes; float64 convolution accumulation; float16/32 output is float32, otherwise float64 |
| `roberts`, `roberts_pos_diag`, `roberts_neg_diag` | Nonempty 2D arrays; same real dtypes/storage and masks | Fused cross differences with upstream gradient rounding, magnitude/sqrt(2) stages, signed zero and mask semantics |
| `threshold_otsu` | 2D uint8 grayscale | Exact scalar threshold; integer histogram, first maximum wins ties |
| `threshold_yen`, `threshold_triangle`, `threshold_isodata` | ND uint8/float32/float64 arrays; integer-bin and floating-bin histograms; supplied count/bin-center metadata for Yen and ISODATA; `return_all` for ISODATA | Exact threshold and dtype; SIMD floating bin assignment, global optional workers, upstream first-maximum and fixed-point rules |
| `threshold_niblack`, `threshold_sauvola` | Nonempty N-D bool, signed/unsigned 8/16/32/64-bit integers, float32/64; odd scalar or per-axis windows; finite scalar weights and nonzero Sauvola range | Reflected padding, float64 integral images in upstream axis order, dtype-specific statistics and parameter promotion; SIMD prefix columns and output math; optional global workers |
| `threshold_mean` | 2D grayscale; same dtypes | Double accumulation; output scalar dtype follows upstream; numerical tolerance applies |
| `invert` | 2D grayscale; same dtypes; `signed_float` | Dtype-preserving inversion, including signed floating zero |
| `adjust_gamma`, `adjust_log`, `adjust_sigmoid` | Nonempty arrays of any dimension; uint8, float32, float64; finite nonnegative gain; log gain ≤1 for uint8 | Same dtype; uint8 LUTs; floating expression rounding tested against upstream |
| `rescale_intensity` | Nonempty arrays of any dimension; same dtypes; `'image'`, `'dtype'`, or finite range pairs | Native min/max and SIMD scaling; output dtype follows input unless a pair requests floating output |
| `histogram`, `cumulative_distribution` | Nonempty uint8 arrays of any dimension; image/dtype histogram range; normalization; no channel axis | Exact integer counts/bin centers; float64 probabilities/CDF |
| `equalize_hist` | Nonempty uint8 arrays of any dimension; mask=None | Float64 output using native histogram and CDF lookup |
| `img_as_float`, `img_as_float32`, `img_as_float64`, `img_as_ubyte`, `img_as_uint`, `img_as_int`, `img_as_bool` | Boolean; signed/unsigned 8/16/32/64-bit integers; float32/64; arbitrary dimensions; `force_copy` | Exact values, dtype, identity, warnings and copy behavior; float16 identity supported |
| `dtype_limits` | Boolean, integer, float16/32/64; `clip_negative` | Metadata only; no pixel processing |
| `erosion`, `dilation`, `opening`, `closing`, `white_tophat`, `black_tophat` | ND arrays, including empty; bool, signed/unsigned 8/16/32/64-bit integers, float32/64; all eight modes, constant fill, flat footprints/sequences; same-dtype output buffers, overlapping/strided views | Exact values on the documented domain; upstream footprint mirroring/centering, integer buffering and wrapping; grayscale NaN inputs/constant fill require fallback |
| `binary_erosion`, `binary_dilation`, `binary_opening`, `binary_closing` | Same input dtypes/dimensions; nonzero truth, including NaN; `ignore`/`max`/`min` modes; footprints/sequences, including convergence iterations; any supported numeric output dtype | Exact bool/numeric 0/1 values; original deprecation warnings retained |
| `transform.integral_image` | N-D real arrays, including empty/strided/endian storage; all 12 real image dtypes; default 64-bit accumulation, explicit int64/uint64 for integer inputs and float64 for all real inputs, explicit float32 for float32 input; scalar identity | Exact ordered cumulative sums, signed/unsigned modular overflow, signed zero, output physical layout and dtype tags; floating exceptional arithmetic uses explicit fallback |
| `transform.integrate` | N-D real integral tables, including scalars and arbitrary signed/zero strides, memmaps, endian/unaligned storage and float16; integer window coordinates, negative indexing and broadcasting | Exact upstream corner order, raw scalar-to-float64 conversion and signed zero; only queried corners are read; sampled nonfinite values or overflow use explicit fallback; corner count must fit the native index width |
| `transform.downscale_local_mean` | N-D real arrays, memmaps and NumPy scalars; all 12 real image dtypes; positive integer factors; scalar constant padding; upstream ignores `clip` | Raw-value means, independent output storage, dtype tags and physical layout; integer/bool output float64, half output with float32 accumulation; scale/depth reduction tolerance; exceptional floating ranges and callback policies use explicit fallback |

Histogram thresholds use native image scans and selectors. Floating bin edges and
centers use small NumPy metadata arrays to preserve version-specific rounding.
Yen also selects a C++ float32 logarithm implementation using fixed metadata
probes; its first maximum can change when log rounding differs by one ULP.
Supplied histograms support finite nonnegative counts whose sum fits float32,
with int64/float32/float64 centers. Invalid, nonfinite or unsupported histogram
metadata, integer widths other than uint8, float16/endian arrays, oversized bin
counts and unknown NumPy log providers currently require explicit fallback.
Floating images with a range wider than their dtype can represent also require
fallback. Degenerate constant/empty cases preserve upstream results, warnings
and errors; ISODATA can legitimately have no fixed point and raise IndexError.
The current arithmetic-warning checks cover ordinary finite and degenerate
histograms; extreme underflow/overflow warning modes need broader validation.

Farid preserves the reference's full coefficient summation order, including its
negative-axis weighting and repeated-axis behavior. Empty axis sequences produce
zeros without consuming boundary options. Nonfinite images, oversized full-kernel
plans, unsupported metadata/classes and extreme constant-fill values use explicit
fallback. Floating magnitude calls with nondefault NumPy underflow policies also
use fallback; directional convolution retains SciPy's warning behavior. Scalar
signed-integer inputs preserve the reference conversion error. Generic Farid
retains dtype metadata on native float32/64 outputs; directional variants follow
SciPy and remove those tags.

Spatial/exposure float32 outputs are checked with `rtol=atol=2e-6`, float64 with
`rtol=atol=2e-13`; dtype conversion values are checked exactly. Spatial filters
retain floating input dtype. Gaussian's intermediate
image retains input floating precision. Noncontiguous arrays are generally copied;
new color transforms directly read uniformly strided floating arrays, including
planar channel-first, reversed, and broadcast inputs. Input
arrays are modified only when explicitly reused as morphology output. Public wrappers preserve upstream signatures and
forward unsupported dtypes/options with a `FallbackWarning`; disable fallback
to require native execution. Low-level `_core` functions reject unsupported
inputs. Native Gaussian radius is capped at 100000.

Color matrix float32 parity includes an absolute dot-product roundoff bound of
`4 * eps * max_row_sum(abs(matrix)) * max(abs(input))` in addition to the base
tolerance. BLAS fusion can differ near cancellation, especially for out-of-range
YCbCr values. Float64 color comparisons use the base tolerance. Color float16,
non-native byte order, and ndarray subclasses use explicit upstream fallback.
Forward Lab/Luv conversions use `atol=2.5e-4` for float32 and `2e-12` for float64,
with the same relative tolerance. The 500× opponent-channel scaling amplifies
cbrt/matrix roundoff; four float32 epsilon units at that scale are about 2.4e-4.
Luv also scales chromaticity differences by `13*L`. Its cached three-value
white-point dot product uses NumPy metadata arithmetic to preserve singular
denominators; all pixel processing remains in C++. Float32 inverse powers use
platform libm. NumPy's dispatched power can differ by one ULP, which Luv/XYZ
cancellation and the near-black sRGB slope amplify. Inverse Luv stress tests and
parameter benchmarks require luminance within one ULP, exact remaining XYZ arithmetic given
that luminance, and the base RGB tolerance against the reconstructed XYZ.
Direct RGB comparisons must also satisfy the propagated bound
`12.92 * (abs(native_XYZ - reference_XYZ) @ abs(RGB_from_XYZ.T))` plus the base
tolerance. Other inverse workloads retain the base tolerance.
Stain error budgets follow input precision and matrix coefficient magnitude.
Float32 separation rounds the logarithm in float32 before scaling, even when
NumPy promotes the result to float64. Standard HED default cases use the base
input-dtype tolerance; parameter reports record the matrix-scaled budget.
Stain float16/endian inputs, ndarray subclasses, complex matrices and batched or
vector matrices currently require explicit upstream fallback.
CIE76 uses the base dtype tolerance. Float16, complex, object and non-native-endian
distance inputs currently require fallback. Array subclasses are stripped by
`np.asarray`, matching upstream. Integer Lab values are cast without intensity
normalization. Color-space routes retain the numerical contracts of each stage.
Weighted distances use the input computation precision for their error budget.
CIEDE2000 float32 uses `rtol=2e-6, atol=2.5e-4`: NumPy's AVX-512 math path can
differ from platform libm by two float32 ULPs in atan2 and one in power. The
chroma-scaled hue term amplifies those differences near small distances. Tests
also compare float32 results with a float64 reference away from the formula's
hue discontinuity. Float64 input comparisons retain `rtol=atol=2e-13`.
CMC's scalar float64 return retains float32 Lab intermediates for float32 inputs
and their error budget. The final promoted accumulation and square root remain
float64. Parameter arrays, NumPy float64 weight scalars and nonfinite weights
currently require explicit upstream fallback. The tests include all 34 of
[Sharma's CIEDE2000 reference pairs](https://hajim.rochester.edu/ece/sites/gsharma/ciede2000/dataNprograms/ciede2000testdata.txt).
For ill-conditioned matrix-to-HSV routes, tests verify matrix accuracy and HSV
against the same intermediate RGB, then propagate the measured RGB perturbation
through the max/min, saturation ratio and periodic hue. Near-zero RGB value or
chroma can amplify small matrix differences; the parameter JSON records the
stage contract instead of claiming a uniform final-output tolerance.
HSV nonfinite values, nonfinite RGB-to-HSV intermediates, and inverse hue sectors
outside int32 currently require explicit upstream fallback.

N-dimensional Gaussian/edge filtering, integer Gaussian/edge filters beyond uint8, masks, negative/sequence edge axes,
anisotropic sigma, other boundary modes, floating-point Otsu histograms, and
Gaussian output buffers currently require upstream fallback. Exposure histogram
functions on floating images, channel-wise histograms, masked equalization,
nonfinite transform parameters, negative gain, named rescale output dtypes,
and empty exposure arrays also currently use fallback. Dtype conversions support
empty arrays where upstream does, preserve upstream scalar behavior, and reproduce
data-dependent downcast warnings. Float16 conversions beyond identity and
non-native-endian conversions beyond identity/copy still require fallback.
Floating exposure and color arithmetic is
numerically checked; full NumPy floating-point warning/`errstate` semantics
remain pending. Each C++ entry documents its buffer layout; caller-supplied
lengths and strides must describe valid storage.
Morphology supports caller output buffers; grayscale output dtype changes, float16,
non-native byte order, ndarray subclasses and scalar inputs currently require fallback.
Sparse-footprint NaN behavior in original SciPy depends on traversal order, so
grayscale NaN calls are explicitly excluded from native coverage. Large rectangles
use separable linear-work extrema scans; canonical 2D boolean rectangles use
packed-bit reductions with SIMD packing/unpacking. Noncanonical bool bytes and
other cases retain the general native path. Small neighborhoods use SIMD stencils.
Scan recurrences, exceptional/border handling and rounding-sensitive powers retain scalar code paths.

## Swap Python backends

```python
# Replace `import skimage as ski` or `from skimage import filters, color`:
import skimagexx as ski
from skimagexx import filters, color

ski.set_num_threads(1)           # serial by default; SIMD remains active
ski.set_fallback_enabled(False)  # require C++; unsupported calls raise
gray = color.rgb2gray(rgb)
edges = filters.scharr(gray)
mask = edges > filters.threshold_mean(edges)
```

For compatibility during migration, install `pip install ".[compat]"` and leave
fallback enabled (the default). Unsupported calls warn and execute original
scikit-image. `filters`, `color`, `exposure`, `morphology`, `transform`, and `util` expose native wrappers.
Unported operations in these modules and processing namespaces such as `measure`
expose upstream-only objects, reported with warnings. Classes and nested modules
are original upstream objects, not accelerated implementations. This does not
patch `sys.modules`, change already imported scikit-image symbols, or guarantee
every nested import path is interchangeable. Use the module alias/import change
shown above. Unknown options retain upstream behavior through fallback.

All 455 inventoried names have availability/signature checks. These checks do
not invoke the algorithms and do not establish numerical parity for pending ports.
All 95 covered wrapper signatures have per-function parity tests. Tests also run
the same pipeline with `skimage` and `skimagexx`, verify unsupported-call warnings,
and assert that strict mode never silently falls back. Adding a native function
requires updating the registry, per-function tests, and benchmark coverage.

## Build and use

Python 3.10+, a C++17 compiler, and OpenMP are required for the default build.
On Windows install Visual Studio's Desktop development with C++ workload.
On Linux install a compiler with OpenMP; Apple Clang needs an OpenMP runtime
or a build with `SKIMAGEXX_OPENMP=OFF`.

```powershell
git clone https://github.com/bipxx/scikit-imagexx.git
cd scikit-imagexx
python -m venv .venv
.\.venv\Scripts\python -m pip install ".[test]"
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python benchmarks/run.py --output benchmarks/results.json
```

On Linux/macOS use `.venv/bin/python` instead of `.venv\Scripts\python`.

```python
import numpy as np
import skimagexx as xx

xx.set_num_threads(4)
rgb = np.random.default_rng(0).random((1024, 1024, 3), dtype=np.float32)
gray = xx.rgb2gray(rgb)
blurred = xx.gaussian(gray, sigma=1.0)
edges = xx.sobel(blurred)
threshold = xx.threshold_otsu((gray * 255).astype(np.uint8))
```

Standalone C++ build and tests, without Python dependencies:

```sh
cmake -S . -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native --config Release
ctest --test-dir build/native -C Release --output-on-failure
```

Include `skimagexx/image.hpp` and link `skimagexx::skimagexx` after
`add_subdirectory(...)`. The public functions return owning `std::vector`s.
For example, `skimagexx::sobel(image.data(), height, width)` accepts float or
double pixels. RGB conversion additionally accepts uint8 directly. For repeated
work, `rgb2gray_into`, `gaussian_into`, and `sobel_into` write to caller-owned
output without allocating or zero-initializing it. The `_into` Gaussian/Sobel
overloads also accept uint8 input with double output. Buffers must be contiguous
and must not overlap; output needs height × width elements. See
`tests/test_native.cpp` for a complete runnable example.

`skimagexx/convert.hpp` exposes dtype conversions into caller-owned storage:

```cpp
#include <skimagexx/convert.hpp>
// source contains count uint16 pixels; destination has count float elements.
auto status = skimagexx::convert_into(source, destination, count,
    skimagexx::PixelType::uint16, skimagexx::PixelType::float32);
```

`ConversionInfo` reports data-dependent downcasting and floating exceptions so
bindings can reproduce warnings without rescanning pixels. Identity and copy
decisions belong to the caller; input and output buffers must not overlap.

`skimagexx/color.hpp` exposes `color_transform_into(input, output, pixel_count,
input_type, transform)` for contiguous three-channel color buffers. It accepts
the conversion pixel types above and writes float32 for float32 input, float64
otherwise. `color_transform_strided_into<T>` accepts floating input with signed
pixel/channel strides measured in elements. Both use the global worker setting
and write caller-owned output without allocation.
`skimagexx/perceptual.hpp` provides `perceptual_into` with explicit white point,
channel count and element strides. Its result reports negative-Z counts and
unsupported HSV inputs. For interleaved extra-channel buffers, set pixel stride
to the channel count. The one-pixel promotion helper preserves NumPy's special
float32 inverse-Lab behavior.

`skimagexx/distance.hpp` provides `delta_e76_into` for two raw Lab buffers,
independent input dtypes and signed pixel/channel strides. It writes float32
only when both inputs are float32, float64 otherwise. Inputs are not normalized;
broadcasting is represented with zero strides. It uses the same process-wide
worker limit as the other kernels.
`delta_e76_rows_into` accepts per-row element offsets, preserving nonuniform
ND views without a temporary image copy. `delta_e_into` provides CIE94,
CIEDE2000 and CMC with scalar weights and explicit intermediate-promotion flags.

## CPU execution

| Algorithm | SIMD work | Parallel work |
|---|---|---|
| RGB to gray | Contiguous AVX2 loads/shuffles; fused uint8 normalization | Independent pixel ranges |
| Color matrices / XYZ | AVX2 matrix arithmetic, interleaved double path, vector sRGB transfer; uint8 decode LUT; bounded normalization blocks | Independent pixel ranges |
| HSV / Lab / LCh | AVX2 masked arithmetic, vector cube roots, range-reduced atan2/sin/cos; libm for special ranges | Independent pixel ranges; local warning counts reduced once per worker |
| Gaussian | Register-resident AVX2 double accumulation; fused uint8 conversion | Independent rows in each pass |
| Sobel | AVX2 fused derivatives and square root | Independent rows |
| Yen / Triangle / ISODATA | AVX2 min/max and floating bin indices, with exact edge correction; uint8 uses the existing histogram kernel | Worker-local histograms and deterministic integer-count merge; 256-bin selector stays serial |
| Otsu | AVX2 constant-image detection and variance scoring | Per-thread eight-lane histograms, then reduction |
| Prewitt/Scharr and directional edges | AVX2 fused gradient arithmetic | Independent rows |
| Gray to RGB / invert | AVX2 replication / arithmetic | Independent pixel ranges |
| CIE76 distance | AVX2 raw-value differences, squares and square root; contiguous loads or gathers for floating inputs | Independent broadcasted pixel pairs |
| CIE94 / CIEDE2000 / CMC | SIMD scaling, reductions, roots, powers and promoted trig/exp; rounding-sensitive atan2 and hue-cancellation hypot use libm | Independent broadcasted pixel pairs |
| Color-space routing | Uses SIMD kernels of the participating transforms, preserving intermediate RGB | Uses the same global limit in each transform |
| Mean threshold | AVX2 double-precision reduction | Per-worker partial sums |
| Exposure correction | uint8 SIMD LUT mapping; AVX2 float/double log/exp/power and simple gamma paths | Independent pixel ranges |
| Intensity rescaling | AVX2 min/max and scaling; uint8 LUT | Per-worker ranges, then pixel ranges |
| Histograms / equalization | Eight-lane counting; AVX2 CDF lookup | Per-worker histograms, then independent pixel ranges |
| Dtype conversion | AVX2 normalization, rounding, integer packing, boolean comparisons and validation maxima; portable loops for 32/64-bit integer-to-float paths | Per-worker ranges and validation extrema |

Histogram counting uses scalar independent lanes, since AVX2 has no efficient
conflict-safe histogram scatter. Prefix sums, borders, tails, and small kernels
also include scalar work. SIMD is applied where suitable; not every instruction
is vectorized.

AVX2 code lives in a separate translation unit. Runtime CPU/OS checks keep it
off unsupported machines. Portable scalar code is always available. There are
no global `-march=native`, AVX2, or fast-math requirements. Disable the compiled
SIMD backend with `-DSKIMAGEXX_AVX2=OFF`, or disable dispatch for a process by
setting `SKIMAGEXX_DISABLE_SIMD=1` **before its first kernel call**.

Floating exposure transforms use IEEE-754 range reduction and reduced-interval
polynomials in the AVX2 translation unit. Standard-library math handles special
values, subnormals, and extreme exponent ranges. These paths retain the same
float32/float64 tolerances as the parity suite; they are not bitwise libm
replacements. Without AVX2, correction uses scalar standard-library math.

Parallel execution is **optional and off by default**. All native kernels share
one process-wide worker limit, initially **1**. Set it once at application startup:

```python
xx.set_num_threads(4)  # enable up to four workers globally
xx.set_num_threads(1)  # disable parallel execution globally
```

The C++ equivalents are `skimagexx::set_num_threads(4)` and
`skimagexx::set_num_threads(1)`. Read the setting with `get_num_threads()` in
either API. SIMD remains enabled regardless of worker count. When the limit is
greater than one, images with at least 262144 pixels can use OpenMP; smaller
images run serially. The limit is shared across calling threads, and each
algorithm snapshots it at entry, so updates apply to subsequent calls. It is
not a per-call context manager. Calls release Python's GIL. When calling from your own worker
pool, setting one C++ thread per caller avoids oversubscription.
OpenMP is optional via `-DSKIMAGEXX_OPENMP=OFF`; those builds accept one worker.

Custom dtype metadata is parity-tested across every covered entry point,
including mixed-dtype Lab operands and scalar/broadcast colors. Descriptor
propagation follows the upstream operation order; stack expansion removes tags
while the applicable color/gradient operations retain them. Pixel arithmetic
remains native. Grayscale expansion also handles mixed-endian structured records
whose normalized field offsets and item size remain unchanged, including nested
fields and subarrays; field byte permutation is vectorized with optional workers.

## Benchmarks and validation

[Verification summary and remaining slower workloads](benchmarks/STATUS.md)
records current comparisons against original scikit-image. Each covered API
has one-worker measurements; recipes also support optional parallel execution.

[Current per-function performance](benchmarks/function-coverage.md) compares
**every native public function** with original scikit-image. Its raw JSON records
input dtype/size, worker count, error, samples, and source hash. Fallback is
disabled during timing and parity verification. These results include Python
dispatch and allocation costs. Ratios below one are retained as regressions;
CI checks benchmark coverage and execution, not hardware-dependent speed ratios.

To rebuild and reproduce all seventeen reports with their default workloads:

```sh
python -m pip install -r benchmarks/requirements.txt
python benchmarks/verify.py
```

Use `python benchmarks/verify.py --threads 1` for serial-only comparisons.
Both commands check parity before timing and regenerate the summary only after
all reports pass coverage and source-provenance checks. Individual suites can
also be run after rebuilding/installing the extension:

```sh
python tools/inventory.py
python benchmarks/coverage.py --sizes 256 1024 --threads 1 4
python benchmarks/conversions.py --size 1024 --threads 1 4
python benchmarks/color.py --size 512 --threads 1
python benchmarks/exposure.py --size 1024 --threads 1
python benchmarks/perceptual.py --size 512 --threads 1 4
python benchmarks/stains.py --size 512 --threads 1 4
python benchmarks/distance.py --size 512 --threads 1 4
python benchmarks/morphology.py --size 512 --threads 1 4
python benchmarks/gaussian.py --size 512 --threads 1 4
python benchmarks/integral.py --size 512 --threads 1 4
python benchmarks/integrate.py --size 512 --threads 1 4
python benchmarks/downscale.py --size 513 --threads 1 4
python benchmarks/summarize.py
```

[Non-default exposure parameters](benchmarks/exposure-parameters.md) separately
cover general gamma powers, inverse log, inverse sigmoid, and explicit rescale
ranges. This prevents the default gamma=1 case from standing in for all gamma
performance. The last completed whole-library checkpoint covers 95 entries in 554
serial default cases and 277 dtype-tagged cases. All 18 reports share one source
fingerprint: 2,934 one-thread comparisons, including 2,868 of 2,895 native image
workloads faster than original scikit-image. The 27 slower cases remain listed.
[Additional conversion widths](benchmarks/conversion-widths.md) cover 56 one-worker cases
across boolean and 8/16/32/64-bit integer inputs. Reports label identity returns,
copies, and metadata separately from native image processing. The Python suite
runs per-function reference checks in each SIMD mode on Windows and Linux; inventory checks alone do not
establish numerical parity for pending ports.
[Color layouts and integer inputs](benchmarks/color-layouts.md) add 120 one-worker
comparisons with strided and contiguous channel-first inputs, including copy costs.
[Perceptual parameters](benchmarks/perceptual-parameters.md) add 60 comparisons
with alternate white points, wrapped hues, extra channels, negative-Z warnings,
and Luv low-lightness/midtone branches.
[Stain parameters](benchmarks/stain-parameters.md) add 156 comparisons across all
11 standard matrix families, three input dtypes and custom output widths.
[Lab distance and color routes](benchmarks/color-distance.md) add 246 comparisons
over broadcast/strided/planar inputs, extra channels, mixed dtypes, representative
color routes, all four Lab distance metrics, scalar pairs, non-default weights,
out-of-gamut HSV and identity calls. Identity rows are labeled separately.
The [CIE76 integer-load experiment](benchmarks/distance-tuning.md) records the
fixed five-layout optimization from individual conversions to packed SIMD loads.
The [nonuniform-layout experiment](benchmarks/distance-layout-tuning.md) records
the measured change from full input copies to row-strided C++ processing.
The [linear-color experiment](benchmarks/linear-color-tuning.md) retains a separate
16-case before/after comparison for skipping unused nonlinear SIMD work.
[Morphology footprints and output reuse](benchmarks/morphology-parameters.md) add
240 comparisons across ten APIs, four dtypes, rectangles, disks and even/asymmetric
footprints. Every case checks exact output values, dtype and warning parity.
The [boolean rectangle tuning experiment](benchmarks/morphology-tuning.md) retains
before/after C++ snapshots and 18 fixed one-worker workloads; it is historical
evidence, separate from the current-source aggregate.

[Earlier single-thread results and same-run before/after comparison](benchmarks/optimized-single-thread.md)
and [Otsu input distributions](benchmarks/histogram-results.md) measure the
earlier optimized kernels. The [initial multithreaded results](benchmarks/results.md)
and [earlier single-thread results](benchmarks/single-thread.md) are historical
snapshots, with their own source hashes. The benchmark checks parity before timing,
warms both libraries, alternates their timing order, uses medians of seven
samples, and includes output allocation/destruction and input conversion.
Seeded synthetic inputs cover 256², 1024², and 2048² pixels, three dtypes,
and C++ worker counts of 1, 4, and 8. Gaussian timing uses sigma=1; it does not
characterize every sigma. Reference BLAS/OpenMP pools are limited to one thread
and C++ counts are explicit, so serial speedups and parallel gains can be
inspected separately. A 20 ms cooldown outside timing reduces worker-pool
interference between batches, and `OMP_WAIT_POLICY=PASSIVE` requests passive
waiting where the runtime supports it. Input creation, imports, and I/O are excluded.

Results depend on CPU, compiler, memory bandwidth, allocator, system load, and
image shape. Values below 1× are retained as regressions. The JSON contains raw
timings, loop counts, versions, source fingerprint, and the invocation. The current fingerprint uses POSIX relative
paths, LF line endings, and includes CMake/Python build configuration. Rebuild after
source changes before rerunning; the hash describes source files, not binary
provenance. `benchmarks/requirements.txt` pins this run's numerical/build tools.
The initial multithreaded run predates the change to serial-by-default execution;
the benchmark explicitly sets every worker count, so it does not rely on the default.

The Python kernels now fill NumPy-owned uninitialized outputs directly.
Gaussian keeps output accumulators in registers across taps and uses an
uninitialized intermediate buffer; it retains the required floating precision
between axes. uint8 Gaussian and Sobel avoid a full converted input image.
Sobel only computes borders and the unprocessed SIMD tail in scalar code.
Otsu avoids OpenMP on serial calls, uses bounded 32-bit histogram lanes and
reduces thread-local counts after work completes. Constant images bypass
histogram construction after an equality scan. General random-input Otsu
performance remains limited by histogram load/store dependencies.

To preserve an installed baseline before editing, copy its package directory
to `build/baseline/_baseline` (including `__init__.py` and the extension binary),
then rebuild/install the edited source. Compare in the same process:

```sh
python benchmarks/run.py --threads 1 --baseline-dir build/baseline --output benchmarks/optimized-single-thread.json
python benchmarks/histogram.py --baseline-dir build/baseline
```

The benchmark records the preserved extension's SHA-256. The baseline binary is
a local build artifact and is not distributed in this source repository; the
historical raw reports remain available without it. Ordinary comparisons
against scikit-image do not require a baseline directory.

Parity tests cover tiny/odd images, impulses, constants, strided arrays,
nonfinite floating pixels, histogram ties, dtype/range behavior, invalid
arguments, concurrent calls, and serial/parallel execution. To repeat fallback
validation in PowerShell:

```powershell
$env:SKIMAGEXX_DISABLE_SIMD = '1'
.\.venv\Scripts\python -m pytest -q
Remove-Item Env:SKIMAGEXX_DISABLE_SIMD
```

## Next milestones

1. Expand dtype, border, channel, and output-buffer parity for native functions.
2. Expand morphology and add connected components and region measurements with the same
   parity and benchmarking requirements.
3. Add geometric transforms, feature detection, and restoration incrementally.
4. Add ARM NEON and broader CPU/compiler validation; no ARM SIMD claim is made.

Algorithm semantics follow the upstream [color API](https://scikit-image.org/docs/0.26.x/api/skimage.color.html)
and [filters API](https://scikit-image.org/docs/0.26.x/api/skimage.filters.html).

The [threshold parameter report](benchmarks/threshold-parameters.md) compares all
three new selectors across layouts, bin counts, constant inputs, supplied
histograms and ISODATA all-threshold output. Timing includes image copies and
histogram metadata. Tests require exact threshold values and matching dtypes.

Yen's compatibility logarithm has an optional eight-lane AVX2/FMA path in
`src/simd_threshold_fma.cpp`. It is dispatched only after the runtime AVX2/OS
checks and a separate FMA CPU-feature check. CPUs without FMA and builds with
SIMD disabled use scalar `std::fma`; the library has no hardware FMA requirement.
Only this translation unit uses the additional `-mfma` flag on GCC/Clang.

`skimagexx/threshold.hpp` exposes floating `histogram_edges` and the shared
`threshold_indices` selector for standalone C++. It returns the selected bin
indices (all fixed points for ISODATA); the caller maps them to its bin centers.
The `float_centers` flag preserves float32 ISODATA arithmetic. An optional
`bin_width` preserves integer-center spacing when conversion to double would
lose it. `threshold_logs` provides the dispatched compatibility logarithm.
The [Yen tuning experiment](benchmarks/threshold-tuning.md) retains measured
before/after snapshots, noisy repeats and the discarded metadata-scan attempt.

`skimagexx/local_threshold.hpp` provides the standalone local-threshold kernel.
Niblack and Sauvola preserve raw intensity ranges, including integer images.
Float32 statistics round the window sums before division, matching the original;
NumPy integer window scalars can promote that division without changing the
stored statistics dtype. Strong scalar weights/ranges preserve their staged
promotion. NaN `r` in the C++ parameter struct selects half the input dtype range.

The native local-threshold domain uses finite images/parameters whose padded
prefix sums, statistics and final expressions fit their working dtypes. Float16,
non-native byte order, subclasses, array-valued weights, nonfinite values,
zero/subnormal ranges, overflowing typed window products and extreme arithmetic
use explicit upstream fallback. Window metadata errors preserve upstream
exception types, including NumPy's unsigned padding-width rejection.
The [local threshold report](benchmarks/local-threshold-parameters.md) covers
128 one-worker comparisons across all 11 native input types, window shapes/sizes, layouts,
N-D volumes and scalar-promotion cases. Numerical parity is checked before timing.

Alpha compositing uses SIMD with strict floating-point order and optional global workers.
Grayscale expansion copies raw values with SIMD and follows physical stride order to
reduce cache misses on planar/Fortran views. Object-backed arrays and subclasses,
unsupported alpha array casts, signaling NaNs in alpha/background metadata, and exceptional RGBA arithmetic use
explicit compatibility fallback. Native RGBA requires native byte order and finite
input with magnitude at most `sqrt(finfo(input.dtype).max)/2`; quiet-NaN backgrounds follow upstream behavior.

The [alpha report](benchmarks/alpha-parameters.md) records 208 one-worker
comparisons against original scikit-image. The [fixed grayscale experiment](benchmarks/alpha-tuning.md)
preserves before/after and confirmation timings for reversed, stepped and endian storage.

Laplace and Roberts use optional global workers and runtime-dispatched AVX2,
including a vector path along single-column images. Half decoding requires AVX2
integer operations only, without an F16C requirement; byte-order conversion is
also vectorized. Native floating inputs currently require finite values with
magnitude at most `sqrt(finfo(working_dtype).max)/16`. Roberts magnitude calls on float32/64 input
with a non-default underflow policy use explicit upstream fallback. Other
unsupported classes/options also retain explicit compatibility fallback.
Nonempty float16 masks raise the same upstream `RuntimeError`; empty masks follow
upstream broadcasting. The [edge report](benchmarks/edge-parameters.md) records
one-thread comparisons for dtypes, dimensions, layouts and masks, while the
[fixed column experiment](benchmarks/edge-tuning.md) retains tuning observations.

The pure C++ interface in `include/skimagexx/small_edge.hpp` includes typed
numeric and binary16 entry points with caller-owned output buffers.

Native dtype checks compare equivalent NumPy descriptors by storage semantics,
so views created with `dtype.newbyteorder('=')` can swap backends consistently.
The per-function descriptor suite also checks warnings, copy/identity behavior
and rejection of opposite-endian data by typed low-level interfaces.


Gaussian-family image arithmetic uses runtime-dispatched SIMD and the global
optional worker limit. Convolution starts with the center coefficient and adds
symmetric pairs in SciPy order, accumulating in double precision and rounding
to the output dtype after each axis. Half input uses float32 working storage.
Small NumPy coefficient arrays preserve typed-sigma rounding; image convolution,
raw/normalized conversion, difference and sharpening run in C++.

Explicit Gaussian float32/64 output buffers retain identity, including supported
strided and endian storage. Partial input/output aliases and self-overlapping
outputs require explicit fallback. Difference of Gaussians and unsharp masking
currently require finite working ranges without exceptional arithmetic; unsupported
precision, arithmetic and channel-loop cases also report fallback.

The [Gaussian-family report](benchmarks/gaussian-parameters.md) covers 225
one-worker comparisons against original scikit-image. The
[fixed interior-span experiment](benchmarks/gaussian-tuning.md) preserves the
initial implementation, optimized results and an independent confirmation,
including slower observations. The native C++ interface in
`include/skimagexx/gaussian_nd.hpp` accepts ordered axis kernels and caller-owned
output buffers without requiring Python.


Integral images retain logical axis order while processing dense physical layouts,
so Fortran and transposed images preserve upstream output strides. Integer sums
use defined unsigned modular arithmetic, including signed output bit patterns.
AVX2 accelerates raw integer conversion, independent column sums and integer
prefixes; floating prefixes retain upstream addition order. All stages share the
global optional worker limit. NumPy handles small dtype metadata and output
allocation; raw conversion and cumulative image arithmetic run in C++.
Strided 64-bit single-chain inputs avoid a dense copy, including reversed,
unaligned and nonnative-endian storage. Normalization and keep-order allocation
share the native call. The [paired experiment](benchmarks/integral-stride-tuning.md)
retains all 148 one-thread workloads, rejected variants and an independent repeat.

The native integral-image working range excludes nonfinite and potentially
overflowing floating arithmetic. Unsupported target precisions and input
subclasses use explicit fallback. Zero-dimensional inputs preserve upstream
identity and dtype-validation order. The [integral report](benchmarks/integral-parameters.md)
checks 148 one-worker workloads, and `include/skimagexx/integral.hpp` exposes
caller-owned output buffers with optional dense physical strides for C++ callers.

Integral-window queries read corners directly through the table's byte strides,
without scanning or copying the image. Each output window retains upstream's
ordered double additions; AVX2 adds independent windows, and large batches can
use the global worker limit. NumPy handles coordinate metadata and output
allocation. The [window-query benchmark](benchmarks/integrate-parameters.md)
covers 144 workloads, including single-window overhead and 65,539-window batches.
`include/skimagexx/integrate.hpp` exposes the same corner calculation for C++
callers with caller-owned output storage. The corner count grows as 2**ndim.
The [coordinate experiment](benchmarks/integrate-metadata-tuning.md) preserves
paired baseline/current measurements, an independent confirmation, diagnostic
profiles, and all slower results against the preceding C++ version.

Local-block means read raw strided storage and apply virtual after-only padding.
AVX2 sums large spans, groups small output blocks, normalizes byte order, and
streams unit-factor conversion. Reversed inputs retain logical output order;
half conversion needs no F16C instructions. Every output block is independent
under the optional global worker limit. The C++ interface is
`include/skimagexx/downscale.hpp`; no Python dependency is needed for native use.
Metadata-only NumPy allocation preserves broadcast/singleton output strides,
with protected-memory tests verifying that it never reads image pixels.
Scalar padding casts retain upstream warning counts and validation order.
Nonfinite/risky floating arithmetic, nonscalar constants, input subclasses,
and nondefault floating underflow or callback policies currently use explicit
fallback. The dedicated report keeps all 120 workloads per worker setting.
The [independent local-mean confirmation](benchmarks/downscale-tuning.md)
measures all 120 cases faster than original scikit-image: 3.04× geometric-mean
speedup with one worker and 5.52× with four. The
[optimization history](benchmarks/downscale-tuning.md) preserves the initial
port, intermediate regressions and paired confirmation against its measured
binary. The final one-worker aggregate is 1.87× faster than that initial C++
build; individual regressions against it remain follow-up targets.
