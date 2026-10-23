# Frangi and ridge-Hessian prototypes

The unregistered prototypes preserve the public signatures of `filters.frangi`
and `filters.hessian`. They are not retained on main and have no performance
claim yet. Native benchmark and broader integration gates remain pending.

C++ computes Gaussian Hessians, orders eigenvalues by magnitude, forms ridge
ratios and Hessian norms, initializes automatic gamma once from the first
scale, and evaluates the response and cross-scale maximum. The ridge-Hessian
wrapper changes nonpositive final responses to one. AVX2 handles independent
pixels and optional global workers default to one. Scalar parameter metadata
preserves NumPy's division/power promotion, including differences between
NumPy 1.24 and 2.x. Empty scales preserve ignored arguments and output layout.

## Explicit native math provider

Near-zero 3-D eigenvalues require the same LAPACK rounding as the Python
reference: the ratio's absolute clipping floor can amplify a small eigenvalue
residual, especially for raw full-width integer magnitudes. Diagnostics confirm
that the native and reference Hessian fields are bit-identical in the initial
failing cases; the eigensolver's residual was the source of response differences.

The Python bindings therefore use an explicit native linear-algebra provider
for 3-D. C++ packs small double symmetric matrices in bounded chunks and calls
the installed NumPy native LAPACK gufunc loop directly, without Python image
operations or the GIL. Two-dimensional eigenvalues retain the C++ formula.
This is a dependency on NumPy's native math runtime, not a rewritten LAPACK
implementation. Future measurements must disclose the provider. The standalone
C++ API accepts an optional native eigenvalue-provider callback and defaults
to its own Jacobi/formula solver; identical sensitive rounding is not claimed
for that default solver.

Float32 exponential rounding also matters: a one-ULP difference near exp(x)=1
can change ridge-Hessian's final zero-to-one decision. The prototype uses
NumPy's licensed float32 approximation and evaluation order with explicit
float FMA. Its near-one output decisions pass 10,001 exact-value checks;
10,007 additional negative exponents pass the specified error budget. The
surrounding arithmetic is SIMD, with per-lane exponential evaluation at this
stage. Vectorization of that component remains a benchmark-driven follow-up.
The existing NumPy third-party license covers reused coefficients and code.

## Validation and retained trials

Strict Python parity budgets remain 2e-6 for float32 and 2e-13 for float64.
The fourth prototype passes 916 checks per SIMD mode on minimum-NumPy Linux;
Windows passes the same checks. These include original failing large-integer
mirror and zero-sigma 3-D cases, exact worker-cap equality, and monkeypatches
forbidding Python image-operation calls. The fifth adds empty-scale ignored
parameter and permuted-storage cases: 966 checks pass on Windows and Linux.
The combined branch passes 6,160 Windows integration checks and 966 dedicated
checks in both SIMD and scalar modes. Modern NumPy 2.5.4 Linux passes all 966
dedicated checks locally. Windows and Linux native builds pass 13 tests;
relocated SDK consumers pass 12 tests, including portable Linux without AVX2
or OpenMP. Linux ASan/UBSan passes all 13 tests with ordinary and dynamic
OpenMP teams.

Full CI exposed three strict ridge-Hessian failures on the hosted modern-Linux
runner, all involving mirror boundaries and the discontinuous zero-to-one
finish. The same cases pass on Windows and minimum-NumPy Linux. They remain
failures, with unchanged parity budgets. An untimed diagnostic compares
Hessian bits, native responses from supplied reference fields, and NumPy versus
libm exponential decisions on that runner. Registration and retention on main
remain pending resolution.

`benchmarks/frangi.py` fixes 576 cases across both APIs, 12 input dtypes and
ranks two/three; caps one/four produce 1,152 rows. It includes full-width
integers, mirror boundaries, zero sigma, default scales, typed parameters,
strided/Fortran/broadcast storage and white ridges. All 1,152 untimed checks
pass locally on Windows, minimum-NumPy Linux and modern-NumPy Linux. Reports
record the native LAPACK provider explicitly. Actual timings remain pending.
All initial strict failures, source snapshots and exact prototype wheels are
retained in ignored checkpoints. No local performance timings have run.
