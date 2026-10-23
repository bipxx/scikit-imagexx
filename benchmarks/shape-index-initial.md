# Shape index: initial measurements

Measured source **fc32f93c03504fd6d39ff99a9e9021f5d0ea29e3** implements
`feature.shape_index` in C++17. Native finite Hessians and symmetric eigenvalues
feed an AVX2 arctangent-polynomial response. Eight-pixel worker partitions keep
serial and parallel output identical. Global workers default to one.

The isolated [measurement](https://github.com/bipxx/scikit-imagexx/actions/runs/39891161346)
passed on Linux and Windows against **scikit-image 0.26.0**. Exact preserved
binary hashes, source fingerprints and seven timing samples per implementation
were verified. Timings include allocation, conversion and Python dispatch.

| One-worker result | Linux | Windows |
|---|---:|---:|
| Geometric-mean speedup | 3.787195× | 2.689617× |
| Slowest case speedup | 2.321666× | 1.714554× |
| Cases faster than upstream | 120/120 | 120/120 |

All 480 timed rows, including four-worker controls, pass numerical, dtype,
shape, stride and NaN-mask checks. Maximum finite absolute error is
**1.1920929e-7 for float32** and **2.2204461e-16 for float64** on both platforms.
Signed-zero regression cases also pass. Flat regions retain NaNs. Broadcast
and zero images contribute the slowest cases and remain in the fixed corpus.

The corpus covers 12 real dtypes × 10 cases: contiguous, strided, Fortran,
tiny, broadcast, zero image, zero sigma, anisotropic sigma, reflect and mirror.
Images are 256×257, seed 8311; worker caps are 1 and 4, seven samples and a
15 ms target per sample. Reference thread pools are limited to one.

Native scope includes ordinary real 2-D arrays/memmaps, boolean and all integer
widths, float16/32/64, anisotropic sigma and ndimage border modes. Integer
normalization, float32 intermediate rounding, owned C output, flat-region NaNs,
signed zeros and small-axis errors are tested. Unsupported protocols, unsafe
arithmetic and non-default underflow policies use whole-function fallback.
This is partial API coverage.

Production packages pass 8,144 dedicated/API/metadata/namespace checks per
runtime, including minimum NumPy 1.24. Combined shape/Hessian suites pass
1,472 cases per SIMD mode per runtime. Untimed corpus checks pass 240 shape
and 480 Hessian cases per runtime. Native C++ tests pass 9/9 on Windows and
Linux, including OpenMP/AVX2 disabled; relocated SDK consumers pass 8/8 on all
three configurations. ASan/UBSan pass 9/9 normally and with constrained OpenMP
teams. No local performance timings were run.

Raw samples: [Linux](shape-index-initial-linux.json),
[Windows](shape-index-initial-windows.json). Exact packages, manifests,
source archives, failed worker-boundary trial and validation logs remain in
ignored build checkpoints. The [full CI](https://github.com/bipxx/scikit-imagexx/actions/runs/39891161029)
is still running; retention awaits all three jobs.
