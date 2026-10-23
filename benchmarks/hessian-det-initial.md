# Hessian determinant native port

Candidate `feature.hessian_matrix_det` preserves the public scikit-image 0.26.0
signature. It is not yet retained on main: full CI is pending. Isolated
performance measurements are verified below, including slower cases.

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
minimum NumPy, native/SDK/sanitizer checks and benchmarks were then run.

Frozen runtime source **f9c80e2272609267606740029d94477517dc7797** passes
609 dedicated checks per SIMD mode on Windows and NumPy 1.24 Linux, with
two exact-mode scalar-overload cases intentionally skipped because they are
supported rather than fallback cases. The Linux integration set passes
8,820 checks; all 480 untimed corpus rows pass on each runtime. Windows's full
run passes 73,847 checks with one inventory-snapshot failure: collection
preceded inventory regeneration. All 456 inventory checks pass on rerun.
The dedicated amplitude and large-volume worker checks added during that
run pass separately. This is not described as a green full-suite run.

Independent native tests pass 10/10 on Windows, Linux with OpenMP/AVX2,
and portable Linux with both disabled. Relocated SDK consumers pass 9/9
in all three configurations. ASan/UBSan pass 10/10 normally and with dynamic
OpenMP teams constrained to two threads. Exact packages, wheel/source hashes,
the original failing packages and logs remain preserved under ignored build
checkpoints. No local performance timings have run.

The [isolated measurement](https://github.com/bipxx/scikit-imagexx/actions/runs/39898516081)
passed on both platforms against the frozen revision. Source and binary hashes,
exact package bytes, the fixed case hash, all 480 rows per OS and seven samples
per implementation per row were verified. Approximation has zero numerical
difference across all timed rows; exact mode's maximum absolute error is
1.214307e-17 Linux / 1.387779e-17 Windows.

| One-worker result (120 cases per mode/OS) | Linux | Windows |
|---|---:|---:|
| Approximation geometric mean | 3.115030× | 2.472311× |
| Approximation minimum | 0.907579× | 0.937262× |
| Approximation cases faster | 119/120 | 117/120 |
| Exact geometric mean | 5.686609× | 4.411284× |
| Exact minimum | 3.997354× | 2.593907× |
| Exact cases faster | 120/120 | 120/120 |

The four slower cases are tiny 7×9 approximations: Linux float64 (0.907579×),
Windows bool (0.937262×), float16 (0.966917×), and float32 (0.955755×).
No cases were removed. A follow-up reduces scalar dispatch and boundary
processing overhead; its correctness and performance gates remain separate.
Raw samples: [Linux](hessian-det-initial-linux.json),
[Windows](hessian-det-initial-windows.json). Candidate/upstream comparisons
are within each runner; these are not paired candidate-to-candidate results.

[Full CI](https://github.com/bipxx/scikit-imagexx/actions/runs/39898515905)
is still running. Retention awaits this gate.

## Small-image follow-up

Frozen source **b93d1082fdb79a415c45623eae5497f7e3f2dae0** addresses the four
tiny approximation cases without changing the corpus. Python caches the
ordinary scalar type sets, avoids repeated dtype/import work, and uses the
finite interval check directly. Bindings read native dtype/alignment metadata
without repeated Python attribute calls. Approximation hoists row clipping
and row-pointer arithmetic out of scalar pixel loops, shares those pointers
with the SIMD interior, and vectorizes the integral-image safety scan.
The numerical expression and clipping order are unchanged.

Local checks pass 1,080 dedicated cases per SIMD mode per runtime, including
all short row widths around SIMD boundaries, negative/even/oversized box
geometry, and the arithmetic safety threshold. Windows integration passes
9,766 checks; the NumPy 1.24 Linux set passes 9,310 (Windows additionally ran
the 456 inventory checks). All 480 untimed corpus rows pass per runtime.
Native tests pass 10/10 and relocated SDK consumers 9/9 on Windows and Linux
with acceleration enabled and portable Linux with it disabled. ASan/UBSan pass
10/10 normally and with constrained dynamic teams. No local timings ran.

The follow-up [isolated measurement](https://github.com/bipxx/scikit-imagexx/actions/runs/39899942605)
passed both OSes. Frozen revision, source and binary hashes, exact wheel/package
bytes, unchanged case hash, all 480 rows per OS and seven samples per
implementation per row are verified.

| One-worker result (120 cases per mode/OS) | Linux | Windows |
|---|---:|---:|
| Approximation geometric mean | 4.009510× | 4.895439× |
| Approximation minimum | 1.770231× | 1.472212× |
| Approximation cases faster | 120/120 | 120/120 |
| Exact geometric mean | 5.571440× | 4.330357× |
| Exact minimum | 3.816523× | 2.705763× |
| Exact cases faster | 120/120 | 120/120 |

All 240 one-worker cases are faster on each OS. The formerly slower tiny
approximations now measure 1.770231× for Linux float64 and 1.678733× /
1.472212× / 1.585138× for Windows bool / float16 / float32. No adverse cases
were removed. Approximation has zero numerical difference across all timed
rows; exact mode's maximum absolute error is 2.081668e-17 Linux /
1.214307e-17 Windows. Raw samples: [Linux](hessian-det-small-linux.json),
[Windows](hessian-det-small-windows.json). Comparisons are candidate versus
upstream within each runner, not paired candidate-to-candidate measurements.

The follow-up [full CI](https://github.com/bipxx/scikit-imagexx/actions/runs/39899942360)
is still running; retention awaits that gate.
