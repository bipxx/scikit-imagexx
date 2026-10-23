# Sato ridge filter native port

The candidate preserves `filters.sato`'s signature and implements 2-D/3-D
Gaussian Hessians, symmetric eigenvalues, clipping, geometric mean, sigma-square
scaling and cross-scale maximum in C++. AVX2 processes response pixels and
negation; optional global workers default to one. Integer input is cast to
floating point without range normalization. Half promotes to float32; typed
sigma factors follow the installed NumPy scalar-promotion rules. Endian,
strided, unaligned, readonly, broadcast and memmap inputs have dedicated tests.
Empty scale lists retain upstream layout/subclass behavior; nonempty responses
are owned C arrays. Unsafe arithmetic, unusual protocols and unsupported
scales use explicit whole-function fallback. This is partial API coverage.

## Numerical contract

Two-dimensional comparisons use relative/absolute budgets 2e-6 for float32
and 2e-13 for float64, with the absolute term scaled by max(1,input amplitude).
For 3-D, the geometric mean is ill-conditioned at a zero second eigenvalue.
LAPACK can return a small positive residual where native Jacobi returns zero;
its square root amplifies that residual. Exact LAPACK output is not claimed.

The parity oracle first independently compares every Hessian component. It
propagates the measured matrix perturbation plus a 64-double-epsilon allowance
at the spectral scale through clipping, product and square root. This yields
a per-pixel conditioning allowance, rather than a blanket relaxed tolerance.
The explicit eigensolver allowance covers the tested 3×3 solver domain; it is
not a proof for arbitrary matrices. `tools/sato_parity.py` records that contract.
Tests retain mirror boundaries, rank-one quadratic/ridge inputs and an analytic
all-one matrix with exact zero response. Negative controls verify that the
oracle rejects ordinary response errors and nonzero constant-image responses.

The [pinned upstream source](https://github.com/scikit-image/scikit-image/blob/v0.26.0/src/skimage/filters/ridges.py)
defines the algorithm. Existing third-party licenses cover reused components.

## Fixed benchmark corpus

`benchmarks/sato.py` uses seed 7933, both ranks, all twelve real dtypes and ten
cases each: contiguous, strided, Fortran with typed float64 sigma, tiny,
broadcast, zero, white ridges with constant border, zero sigma, default scales
and mirror borders. There are 240 cases per worker cap, 480 rows per OS at
caps one and four. Each timed row stores seven randomized samples per
implementation, allocation/destruction included, 15 ms target batches and
20 ms cooldown. Reference pools use one worker. Source/binary hashes and the
typed case-specification hash are recorded. Parity diagnostics include raw
error, amplitude-scaled error, conditioning allowance and affected pixels.
No case may be removed after observing measurements.

## Local validation and trial history

The initial prototype's strict comparisons caught the predicted 3-D mirror
conditioning issue. Its source, failing checks and exact packages are retained.
After adding the explicit conditioning oracle, checks caught an endian float32
output-dtype error; this was corrected without changing the response algorithm.
The second package passes 833 dedicated checks; the registered third package
passes 834, then 882 after expanded half/integer/bool sigma promotion cases.
Windows integration passes 8,960 checks (6 skipped); minimum-NumPy Linux passes
8,920 (46 skipped). Both runtimes pass the 480-row untimed corpus. Native tests
pass 11/11 on Windows and accelerated/portable Linux; relocated SDK consumers
pass 10/10 on all three. ASan/UBSan pass 11/11 normally and with constrained
dynamic OpenMP teams. No local performance timings have run.

Frozen source **a5f4a68ee5cb9868b45068e6a8b33fff28142ab6** passes all 882
expanded dedicated checks per SIMD mode per runtime. The final untimed corpus
has 20/480 rows per runtime needing the conditioning allowance, all retained.
Maximum input-scaled absolute error is 1.567116e-9; raw maximum is 1.781590e10
because integer inputs, including full-width uint64, are cast without range
normalization. For a float64 mirror case, the raw error is 3.457321e-9.
These are numerical-parity limits, not exact output equality. Reports retain
both raw and scaled errors so input scaling does not hide the difference.

[Full CI](https://github.com/bipxx/scikit-imagexx/actions/runs/39905542928)
and [isolated performance measurements](https://github.com/bipxx/scikit-imagexx/actions/runs/39905542679)
are running against that frozen source. This candidate is not retained on main,
and no speedup is claimed yet. Exact wheels, source archive, initial failures,
and final validation logs are preserved in ignored checkpoints.

## Initial Linux measurements

The Linux job of the [isolated measurement](https://github.com/bipxx/scikit-imagexx/actions/runs/39905542679)
completed successfully against frozen source `a5f4a68`. Source fingerprint,
exact package/wheel bytes, binary hash, typed case-specification hash, all
480 rows and seven samples per implementation per row are verified.

| One-worker result (120 cases per rank) | Linux |
|---|---:|
| 2-D geometric mean | 7.255148× |
| 2-D minimum | 3.016982× |
| 2-D cases faster | 120/120 |
| 3-D geometric mean | 5.537527× |
| 3-D minimum | 2.861452× |
| 3-D cases faster | 120/120 |

All 240 one-worker Linux cases are faster. The measured 2-D outputs are exact.
In 3-D, 20/240 rows across both worker settings require the documented
conditioning allowance. Maximum input-scaled error is 1.381438e-9 and raw
error 2.033356e10 for raw integer magnitudes. These are numerical comparisons,
not exact LAPACK parity. Raw samples: [Linux](sato-first-linux.json).
Windows measurements and full CI are still running; retention is pending.
