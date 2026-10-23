# Hessian determinant native port

Candidate `feature.hessian_matrix_det` preserves the public scikit-image 0.26.0
signature. It is not yet retained on main: full validation and isolated
performance measurements are pending.

The two upstream paths are implemented independently in C++:

- Two-dimensional approximation normalizes integers as `img_as_float`, promotes
  half to float32, and accumulates an ordered float64 integral image. Eight
  clipped integral boxes form each response. Start coordinates are clipped
  before deriving end coordinates, and each box sum is clamped nonnegative,
  including signed images. Geometry uses the original even size and Python
  floor division for negative sizes. AVX2 handles four interior responses with
  contiguous loads; scalar borders preserve the same arithmetic and signed zeros.
- Exact mode in ranks 2–5 reuses native finite-difference Hessians and computes
  partial-pivot LU per pixel, with first-maximum tie selection, swap signs,
  ordered diagonal logarithms and signed exponential reconstruction. Float32
  matrices compute in double before casting, as in NumPy. AVX2 processes four
  matrices at a time; scalar tails and portable builds use the same precision.

Both paths use the optional global worker cap (default one), release the GIL,
and return owned C arrays. Approximation returns float64, exact mode float32
or float64. Strides, endian/unaligned/readonly inputs, dtype-tag stripping,
memmap-to-array results, zero sigma, empty approximation, small-axis exact
errors and ignored `approximate` objects in dimensions above two are tested.
Nonfinite/extreme arithmetic, unsupported sigma types/protocols, ranks outside
the native domain and non-default underflow policy use one explicit whole-call
fallback. This is partial input coverage, not a claim of complete API parity.

Reference algorithms: the pinned [scikit-image box kernel](https://github.com/scikit-image/scikit-image/blob/v0.26.0/src/skimage/feature/_hessian_det_appx_pythran.py)
and [NumPy determinant implementation](https://github.com/numpy/numpy/blob/v2.5.4/numpy/linalg/umath_linalg.cpp).
Their licenses are already distributed in `third_party/` and installed SDKs.

## Fixed performance corpus

`benchmarks/hessian_det.py` fixes seed 9231, all twelve real dtypes, and ten
cases per dtype per algorithm path. Approximation includes contiguous,
strided, Fortran, tiny, broadcast, zero, zero-sigma, negative-sigma, even-size
and large-sigma cases. Exact mode retains the first seven and adds 3-D,
4-D and 5-D volumes, including anisotropic smoothing. No adverse cases may
be removed. There are 240 cases per worker setting and 480 rows per OS at
caps one and four. Upstream reference pools are limited to one. Each row
contains seven randomized reference/C++ samples, allocation and destruction
included, targeting 15 ms per batch with 20 ms cooldown. Source, wheel and
binary hashes accompany the raw results. Measurements run in isolated Actions;
local corpus checks use only `--check-only`.

## Trial log

The first Windows package passed 590 dedicated checks with SIMD disabled.
SIMD checks caught subtraction-from-zero losing the sign of zero during
negation. Explicit sign-bit negation fixed this; the second package passed
590 dedicated checks with SIMD enabled as well. Initial packages and failing
logs are preserved. Broader validation, additional amplitude/volume checks,
minimum NumPy, native/SDK/sanitizer checks and benchmarks are in progress.
No local performance timings have run.
