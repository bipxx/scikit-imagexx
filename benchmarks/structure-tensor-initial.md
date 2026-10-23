# Structure tensor: initial measurements

Measured source **ccab3b0460c69265388c2899c5b9c59a25657739** implements
`feature.structure_tensor` in C++17. Sobel derivatives retain SciPy's double
accumulation and per-axis rounding. Tensor products retain the image precision;
Gaussian smoothing reuses the native separable implementation. AVX2 processes
independent pixels, and global workers default to one.

The isolated [measurement](https://github.com/bipxx/scikit-imagexx/actions/runs/39877936033)
passed on Linux and Windows against **scikit-image 0.26.0**. Exact preserved
binary hashes, source fingerprints and seven samples per implementation were
verified. Results include allocation, integer normalization, array preparation
and Python dispatch.

| One-worker result | Linux | Windows |
|---|---:|---:|
| Geometric-mean speedup | 2.984798× | 2.263855× |
| Slowest case speedup | 1.771831× | 1.502563× |
| Cases faster than upstream | 120/120 | 120/120 |

All 480 timed rows pass numerical and output-layout checks. Maximum numerical
absolute difference is **zero** across both worker caps and both platforms.
Dedicated signed-zero cases also pass. Higher-rank short-axis inputs contribute
the slowest Linux cases; Windows also exposes bool zero-sigma preparation costs.
These workloads remain in the raw reports.

The fixed corpus has 12 real dtypes × 10 workloads: contiguous, strided,
Fortran, zero sigma, anisotropic sigma, tiny, broadcast, 3-D, 4-D and 5-D.
The seed is 8311. Ordinary 2-D images are 256×257; volume shapes are
31×33×35, 7×9×11×13 and 3×5×7×9×11. Both worker caps 1 and 4 are measured;
reference thread pools are limited to one.

Supported native calls use real arrays or memmaps, ranks 2–5 after squeezing,
ordinary finite sigmas with safe typed scalar powers, all ndimage border modes
and grid aliases, and `rc`/`xy` ordering. Nonfinite/extreme data, unsafe typed
sigma powers, unusual protocols and non-default underflow policies use explicit
compatibility fallback. Ordinary shape and sigma-length errors match upstream.

Local production-package validation passes 5,126 dedicated/API/namespace and
metadata checks on each runtime, including minimum NumPy 1.24. The dedicated
750-case suite passes with SIMD enabled and disabled. Standalone C++ tests pass
on Windows and Linux, with OpenMP/AVX2 enabled and disabled; relocated SDK
consumers pass 4/4 on Windows and both Linux configurations. AddressSanitizer
and UndefinedBehaviorSanitizer pass all five native tests, including large
5-D arrays and constrained OpenMP teams. No local performance timings were run.

Raw samples: [Linux](structure-tensor-initial-linux.json),
[Windows](structure-tensor-initial-windows.json).
The initial source archive, package manifests, exact Actions packages, failed
shape-error trial and validation logs are retained under ignored
`build/checkpoints/`. Exact local wheels remain in their `build/tensor-*`
directories.
The implementation is retained after all three
[full CI](https://github.com/bipxx/scikit-imagexx/actions/runs/39877936041) jobs
passed: Windows, Linux and minimum NumPy. Subsequent edits are reports and
merge bookkeeping; runtime sources match the measured revision.
