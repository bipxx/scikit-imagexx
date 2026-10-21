# scikit-imagexx

An incremental C++17 rewrite of selected scikit-image algorithms, with NumPy
bindings, numerical parity checks, explicit AVX2 SIMD, OpenMP parallelism, and
reproducible benchmarks. The native port is in progress. Python compatibility
namespaces let applications swap backends while unsupported calls use explicitly
reported upstream fallback. Fallback is compatibility, not a native rewrite.

On an i7-12700K with **one worker**, the optimized 2048² uint8 paths measured
5.30× (RGB), 5.46× (Gaussian σ=1), 8.79× (Sobel), and 14.96× (Otsu) faster
than scikit-image in one local run. Allocation and conversion are included.
See the full [benchmark report](benchmarks/optimized-single-thread.md), including
floating-point inputs, smaller sizes, raw samples, and before/after regressions.

## Supported parity surface

The reference is scikit-image **0.26.0**. Native kernels require nonempty inputs.
There are currently **15 partially native public functions**. The
[API inventory](coverage/README.md) tracks 446 exported public callable names across 17
modules; 431 remain pending native implementation and dedicated parity tests.
This is not full scikit-image coverage.

| Function | Supported inputs and options | Numerical contract |
|---|---|---|
| `rgb2gray` | 3D RGB; uint8, float32, float64; arbitrary channel axis | Same luminance weights; uint8 normalized to float64 |
| `gray2rgb` | 2D grayscale; uint8, float32, float64; arbitrary output channel axis | Exact replication with original dtype |
| `gaussian` | 2D grayscale; same dtypes; scalar nonnegative sigma; `mode='nearest'` or `'reflect'`; `preserve_range`; nonnegative `truncate` | Separable axis-0 then axis-1 convolution; same kernel radius rounding and border conventions |
| `sobel`, `prewitt`, `scharr` | 2D grayscale; same dtypes; reflected borders; magnitude or axis 0/1 | Same normalization; uint8 normalized to float64 |
| `sobel_h/v`, `prewitt_h/v`, `scharr_h/v` | 2D grayscale; same dtypes; reflected borders | Signed directional gradients |
| `threshold_otsu` | 2D uint8 grayscale | Exact scalar threshold; integer histogram, first maximum wins ties |
| `threshold_mean` | 2D grayscale; same dtypes | Double accumulation; output scalar dtype follows upstream; numerical tolerance applies |
| `invert` | 2D grayscale; same dtypes; `signed_float` | Dtype-preserving inversion, including signed floating zero |

Float32 outputs are checked with `rtol=atol=2e-6`, float64 with
`rtol=atol=2e-13`. Float inputs retain dtype and range. Gaussian's intermediate
image retains input floating precision. Noncontiguous arrays are copied; input
arrays are never modified. Public wrappers preserve upstream signatures and
forward unsupported dtypes/options with a `FallbackWarning`; disable fallback
to require native execution. Low-level `_core` functions reject unsupported
inputs. Native Gaussian radius is capped at 100000.

N-dimensional images, signed/16-bit integers, masks, negative/sequence edge axes,
anisotropic sigma, other boundary modes, floating-point Otsu histograms, and
Gaussian output buffers currently require upstream fallback. The C++ API takes
contiguous buffers; caller-supplied pointer lengths must match dimensions.

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
scikit-image. `filters`, `color`, and `util` expose native wrappers. Other top-level
processing namespaces such as `morphology`, `measure`, and `transform` currently
expose upstream-only objects, reported with warnings. Classes and nested modules
are original upstream objects, not accelerated implementations. This does not
patch `sys.modules`, change already imported scikit-image symbols, or guarantee
every nested import path is interchangeable. Use the module alias/import change
shown above. Unknown options retain upstream behavior through fallback.

All 446 inventoried names have availability/signature checks. These checks do
not invoke the algorithms and do not establish numerical parity for pending ports.
The 15 native wrapper signatures have per-function parity tests. Tests also run
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

## CPU execution

| Algorithm | SIMD work | Parallel work |
|---|---|---|
| RGB to gray | Contiguous AVX2 loads/shuffles; fused uint8 normalization | Independent pixel ranges |
| Gaussian | Register-resident AVX2 double accumulation; fused uint8 conversion | Independent rows in each pass |
| Sobel | AVX2 fused derivatives and square root | Independent rows |
| Otsu | AVX2 constant-image detection and variance scoring | Per-thread eight-lane histograms, then reduction |
| Prewitt/Scharr and directional edges | AVX2 fused gradient arithmetic | Independent rows |
| Gray to RGB / invert | AVX2 replication / arithmetic | Independent pixel ranges |
| Mean threshold | AVX2 double-precision reduction | Per-worker partial sums |

Histogram counting uses scalar independent lanes, since AVX2 has no efficient
conflict-safe histogram scatter. Prefix sums, borders, tails, and small kernels
also include scalar work. SIMD is applied where suitable; not every instruction
is vectorized.

AVX2 code lives in a separate translation unit. Runtime CPU/OS checks keep it
off unsupported machines. Portable scalar code is always available. There are
no global `-march=native`, AVX2, or fast-math requirements. Disable the compiled
SIMD backend with `-DSKIMAGEXX_AVX2=OFF`, or disable dispatch for a process by
setting `SKIMAGEXX_DISABLE_SIMD=1` **before its first kernel call**.

Parallel execution is **optional and off by default**. All four algorithms share
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

## Benchmarks and validation

[Current per-function performance](benchmarks/function-coverage.md) compares
**every native public function** with original scikit-image. Its raw JSON records
input dtype/size, worker count, error, samples, and source hash. Fallback is
disabled during timing and parity verification. These results include Python
dispatch and allocation costs. Ratios below one are retained as regressions;
CI checks benchmark coverage and execution, not hardware-dependent speed ratios.

```sh
python tools/inventory.py
python benchmarks/coverage.py --sizes 256 1024 --threads 1 4
```

[Current single-thread results and same-run before/after comparison](benchmarks/optimized-single-thread.md)
and [Otsu input distributions](benchmarks/histogram-results.md) measure the
optimized kernels. The [initial multithreaded results](benchmarks/results.md)
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
timings, loop counts, versions, source hash, and the invocation. Rebuild after
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
2. Add morphology, connected components, and region measurements with the same
   parity and benchmarking requirements.
3. Add geometric transforms, feature detection, and restoration incrementally.
4. Add ARM NEON and broader CPU/compiler validation; no ARM SIMD claim is made.

Algorithm semantics follow the upstream [color API](https://scikit-image.org/docs/0.26.x/api/skimage.color.html)
and [filters API](https://scikit-image.org/docs/0.26.x/api/skimage.filters.html).
