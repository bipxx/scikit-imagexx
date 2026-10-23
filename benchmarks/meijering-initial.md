# Meijering ridge filter native port

This candidate preserves `filters.meijering`'s public signature. It is not yet
retained on main; isolated measurements are verified below, and full CI
remains the retention gate.

## Native scope and arithmetic

Ranks 1–5, all twelve real dtypes, ndarray/memmap storage and ordinary finite
scale sequences use native dispatch. Integer input is cast to float64 without
range normalization; half promotes to float32. Black/white ridges, ndimage
borders, dtype tags, endian/unaligned/strided/broadcast input, empty-scale
layout/subclass behavior and typed alpha/scales have dedicated parity checks.
Unsupported protocols/ranks/scales and unsafe arithmetic use explicit whole-call
fallback. The upstream supports higher dimensions; this is partial coverage.

C++ reuses native Gaussian Hessians and symmetric eigenvalues, then projects
through the circulant matrix with diagonal one and off-diagonal alpha cast
to field precision. It selects the first response of largest absolute
magnitude, retains its sign, and clips negative results to zero. Each scale
is normalized by its own global maximum before the cross-scale pixel maximum.
It does not normalize after combining scales. The implementation follows the
positive default alpha in the upstream code, `1/(ndim+1)`.

Projection uses ordered nonfused multiplication/addition in field precision;
AVX2 handles independent pixels, normalization and maxima. Optional global
workers default to one; a maximum reduction is independent of partition order.
The GIL is released. Public C++ buffers are dense/aligned and separate; unsafe
or nonfinite fields/prior outputs reject the native result.

Parity budgets are 3e-6 relative and absolute for float32, and 3e-13 for
float64. Exact LAPACK or arbitrary BLAS arithmetic identity is not claimed.
Tests include signed magnitude ties, negative selected responses, polynomial
and tilted/axis ridges, alpha changes and serial/parallel equality.

The [pinned upstream algorithm](https://github.com/scikit-image/scikit-image/blob/v0.26.0/src/skimage/filters/ridges.py)
and already-distributed third-party licenses cover the reused components.

## Fixed benchmark corpus

Seed 8621 fixes ranks 1–5 and all twelve dtypes. Ten cases each retain
contiguous, strided, Fortran with nondefault alpha, tiny, broadcast, zero,
zero-sigma, default scales, mirror, and white ridges with negative alpha and
constant border. Shapes are 257; 128×129; 9×11×13; 3×4×5×6; and 2×3×4×5×6.
There are 600 cases per worker cap, 1,200 rows per OS at caps one and four.
Reference pools use one worker. Seven randomized samples per implementation
include allocation, conversion, dispatch and output destruction, with a 15 ms
batch target and 20 ms cooldown. Source/binary and fixed-case hashes accompany
raw reports. No cases may be removed after observing results. Local corpus
checks use only `--check-only`.

## Trial history and validation

The first package passes 715 Windows checks per SIMD mode and 752 expanded
checks on Windows and minimum-NumPy Linux. The second adds validation for
nonfinite prior output in the public C++ response API; all 752 dedicated
checks per SIMD mode pass on both runtimes. Windows native tests pass 12/12
and relocated SDK consumers 11/11. Linux accelerated/portable tests also pass
12/12, and relocated SDK consumers 11/11 in each configuration. ASan/UBSan
pass 12/12 normally and with constrained dynamic OpenMP teams.
The registered third package passes 780 dedicated checks per SIMD mode per
runtime. Windows integration passes 9,814 checks (6 skipped); minimum-NumPy
Linux passes 9,774 (46 skipped). All 1,200 untimed corpus rows pass per runtime.
Initial packages and source archives remain preserved. No local timings ran.

Frozen runtime **f8cb5b4ba12c9cf6a4734c5fdbe6b610953e2c5c** is under
[full CI](https://github.com/bipxx/scikit-imagexx/actions/runs/39907246156)
and [isolated measurement](https://github.com/bipxx/scikit-imagexx/actions/runs/39907246433).
Later merged Sato measurement reports do not change its runtime sources.
Verified measurements are recorded below; retention awaits full CI.

## Verified isolated measurements

Both measurement jobs passed against frozen source `f8cb5b4`. Source and binary
hashes, exact wheel/package bytes, the fixed 600-case specification, all 1,200
rows per OS and seven samples per implementation per row were verified.

| One-worker result (120 cases per rank/OS) | Linux | Windows |
|---|---:|---:|
| Rank 1 geometric mean | 7.488270× | 5.322143× |
| Rank 2 geometric mean | 7.688674× | 6.176978× |
| Rank 3 geometric mean | 5.616934× | 3.810500× |
| Rank 4 geometric mean | 5.387629× | 3.685129× |
| Rank 5 geometric mean | 5.006351× | 3.682196× |
| Minimum over all ranks | 2.993998× | 2.070080× |
| Cases faster over all ranks | 600/600 | 600/600 |

All one-worker cases are faster per OS. Rank 1 measured outputs are exact;
maximum absolute error across all ranks and worker caps is 2.384186e-7
(float32). No adverse cases were removed. Comparisons are candidate versus
upstream within each runner, not paired candidate-to-candidate measurements.
Raw samples: [Linux](meijering-first-linux.json),
[Windows](meijering-first-windows.json). Full CI remains the retention gate.
