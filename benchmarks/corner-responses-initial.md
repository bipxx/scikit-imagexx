# Corner responses: initial measurements

Measured source **0513a44d3f837881d50b432a9e24f5a8121d3eba** implements
`feature.corner_harris`, `corner_shi_tomasi` and `corner_foerstner` in C++17,
sharing the native structure tensor and fused AVX2 response calculations.
Global workers default to one and can optionally be increased.

The isolated [measurement](https://github.com/bipxx/scikit-imagexx/actions/runs/39881327150)
passed on Linux and Windows against **scikit-image 0.26.0**. Exact package
binary hashes, source fingerprints and seven timing samples per implementation
were verified. Timings include allocation, conversion and Python dispatch.

| One-worker geometric-mean speedup | Linux | Windows |
|---|---:|---:|
| Harris (96 cases) | 3.225440× | 2.333236× |
| Shi–Tomasi (84 cases) | 3.486305× | 2.377552× |
| Förstner (84 cases) | 3.590115× | 2.850550× |

All 264 one-worker cases are faster on each OS. Minimum speedups by API are
2.357085× / 2.366795× / 2.509188× on Linux and
1.505951× / 1.654599× / 1.367157× on Windows. All 1,056 timed rows,
including four-worker controls, pass parity with **zero numerical absolute
difference**. Linux's slowest cases use strided float32 arrays; Windows's
slowest cases are broadcast integer Förstner calls. These remain in the corpus.

The fixed corpus covers 12 real dtypes, seven layouts/options per API
(contiguous, strided, Fortran, zero sigma, anisotropic sigma, tiny and
broadcast), plus 12 Noble-method Harris controls. Images are 256×257, seed
8311; worker caps are 1 and 4 and reference thread pools are limited to one.

Native scope includes ordinary real 2-D arrays/memmaps and finite Harris
scalars, preserving float32 intermediate rounding, typed scalar promotion,
zero-trace masks, output layout, memmap type and Förstner tuple results.
Unsafe arithmetic, unsupported protocols and non-default underflow policies
use explicit whole-function compatibility fallback. This is partial API coverage.

Production packages pass 5,904 dedicated/API/metadata/namespace checks on each
runtime, including NumPy 1.24. Dedicated combined suites pass 1,450 cases with
SIMD enabled and disabled. Native C++ tests pass 6/6 on Windows and Linux,
including OpenMP/AVX2 disabled; relocated SDK consumers pass 5/5 on all three
configurations. ASan/UBSan pass 6/6 normally and with constrained OpenMP teams.
No local performance timings were run.

Raw samples: [Linux](corner-responses-initial-linux.json),
[Windows](corner-responses-initial-windows.json). Exact packages, manifests,
source archives and validation logs are preserved in ignored build checkpoints.
The [full CI](https://github.com/bipxx/scikit-imagexx/actions/runs/39881327501)
passed all three jobs: Windows, Linux and minimum NumPy. The implementation
is retained; subsequent changes are reports and merge bookkeeping, and runtime
sources match the measured revision.
