# Frangi and ridge-Hessian candidate

`filters.frangi` and `filters.hessian` preserve scikit-image 0.26.0 signatures,
module exports and root convenience aliases. They are native-partial candidates
on the feature branch, not yet retained on main. Full CI and the new float32
SIMD experiment remain pending. The initial whole-field measurements retain
four slower Windows zero-sigma cases, so that baseline is not the final
performance result.

## Implementation and numerical scope

C++ computes Gaussian Hessians, orders eigenvalues by magnitude, forms ridge
ratios and Hessian norms, initializes automatic gamma once from the first
scale, and evaluates response and cross-scale maximum. Ridge-Hessian changes
nonpositive final responses to one. The native subset covers real 2-D/3-D
arrays and memmaps, all integer widths and float16/32/64, ordinary scale
sequences, legacy scale controls, ridge polarity and ndimage boundaries.
Unsafe arithmetic and unsupported protocols use explicit whole-function
fallback. Scalar metadata preserves NumPy 1.24/2.x promotion. Empty scales
preserve ignored arguments and output storage layout.

AVX2 handles independent pixels; optional globally configured workers default
to one. The standalone C++ API also builds without OpenMP or AVX2.

### Explicit native math providers

Python bindings use two native NumPy math providers, with no Python image
operations or GIL held during kernel execution:

- In 3-D, C++ packs symmetric double matrices in bounded chunks and calls the
  installed NumPy LAPACK `eigvalsh_lo` C loop. The 2-D solver uses a C++ formula.
- Float64 exponentials call NumPy's native exp C loop once per whole factor
  field, using separate input/output buffers. SIMD preparation and combination
  reuse the now-dead eigenvalue scratch buffer. Whole-field calls preserve
  host vector/tail dispatch independently of C++ worker partitions.

These are dependencies on NumPy's native math runtime, not rewritten LAPACK
or a NumPy-free claim. Standalone C++ accepts optional eigenvalue and exp
providers and defaults to its own Jacobi/formula solver and libm. Identical
sensitive rounding is not claimed for those standalone defaults.

The baseline's float32 exp uses NumPy's BSD-licensed approximation and explicit
scalar FMA inside SIMD response arithmetic. A new candidate evaluates ordinary
float32 exponents with vector FMA after runtime CPU checks, preserving the
portable helper for subnormal/exceptional exponents and CPUs without FMA.
Its speedup is not measured yet. The existing NumPy third-party license covers
reused coefficients and evaluation order.

## Initial measurements: whole-field baseline

Measured source: `841693bfc8b859eee4c8fa9cc5fb331dc7491ac4`, before the new
float32 vector-FMA experiment. The completed Linux job in
[run39914328727](https://github.com/bipxx/scikit-imagexx/actions/runs/39914328727)
passed both OS jobs, with 1,152 verified rows per OS.
[Raw Linux samples](frangi-first-linux.json) and
[raw Windows samples](frangi-first-windows.json) preserve every case and repetition.

| OS | API | Rank | One-worker geometric mean vs scikit-image | Minimum | Faster cases |
|---|---|---:|---:|---:|---:|
| Linux | Frangi | 2 | 6.438435x | 1.148832x | 144/144 |
| Linux | Frangi | 3 | 2.694105x | 1.175280x | 144/144 |
| Linux | Ridge-Hessian | 2 | 6.261027x | 1.153385x | 144/144 |
| Linux | Ridge-Hessian | 3 | 2.691994x | 1.181634x | 144/144 |
| Windows | Frangi | 2 | 3.652343x | 0.911185x | 142/144 |
| Windows | Frangi | 3 | 1.924361x | 1.122754x | 144/144 |
| Windows | Ridge-Hessian | 2 | 3.607668x | 0.925382x | 142/144 |
| Windows | Ridge-Hessian | 3 | 1.924135x | 1.092483x | 144/144 |

Linux is faster in all 576 one-worker cases; Windows in 572/576. The four slower
Windows cases are 2-D zero-sigma inputs: Frangi float16/float32 at
0.993501x/0.911185x and ridge-Hessian float16/float32 at
0.944945x/0.925382x. They remain in the fixed corpus and motivate the vector-FMA
trial; they have not been hidden, removed or declared optimized.

Across both worker caps, maximum absolute error for 3-D Frangi is
2.220446049250313e-16 Linux / 1.6653345369377348e-16 Windows; the other groups
have zero recorded difference. Strict budgets are 2e-6 float32 / 2e-13 float64;
no conditioning allowance or relaxed zero-decision test is used.

The fixed corpus contains both APIs, 12 input dtypes and ranks two/three,
with contiguous, strided, Fortran, tiny, broadcast, zero, zero-sigma,
default-scale, mirror, white/multiple-scale, automatic-gamma and typed-gamma
cases. Ordinary shapes are 128x129 and 9x11x13; these results do not establish
large-volume throughput. The case-spec SHA256 is
`5b56ac97111f39a52057aac850e4ab04c10bce80931b28adea07960e2e73357f`.
Reference thread pools are one; C++ caps are one/four. Timings include warmed
Python calls, native input conversion, allocation and output destruction, with
seven samples per implementation, a 15 ms target batch and 20 ms cooldown.
The report discloses both math providers. Verification checks frozen source,
exact wheel/package bytes, binary hashes, corpus order, all samples, medians,
ratios and dtype/error contracts. This is an isolated CI measurement; no local
timing calls have run. Different runner trials do not prove paired improvement.

## Parity evidence and rejected trials

The whole-field implementation passes 973 dedicated checks per SIMD mode on
Windows, minimum-NumPy Linux and modern-NumPy Linux, plus all 1,152 untimed
corpus checks per runtime. Registered exports add 24 backend-swap pipelines;
shared API/inventory checks pass on all three runtimes. Windows and Linux
native builds pass 13 tests; relocated SDK consumers pass 12 tests in Windows,
Linux SIMD/OpenMP and portable Linux configurations. Linux ASan/UBSan passes
13 tests with ordinary and dynamic OpenMP teams.

Initial 3-D failures were amplified near-zero eigensolver residuals; Hessian
fields were bit-identical. A native LAPACK provider corrected those cases.
A later hosted AVX-512 failure came from exp(x)==1 rounding: NumPy returned
the float just below one where libm returned one, and ridge-Hessian's final
zero-to-one step amplified the difference. The first double-exp provider used
four-element calls; hosted NumPy 1.24 exposed length-dependent dispatch,
causing 14 strict/worker failures. That trial is rejected even though its
benchmark corpus was faster. Raw packages, samples and failures are retained
in ignored checkpoints.

The whole-field correction has a native controlled-rounding regression that
requires one provider call per factor and verifies tiny positive responses
survive the final transform. A same-host
[audit39915622311](https://github.com/bipxx/scikit-imagexx/actions/runs/39915622311)
ran NumPy 1.24 and 2.5 on an actual AVX-512 CPU. Both expose five sensitive
points with bit-identical Hessians, and the native final responses match the
reference; both SIMD and scalar parity suites pass. Earlier successful AVX2
jobs alone were not treated as proof of the original AVX-512 case.

The new float32 vector-FMA candidate has additional bitwise comparisons with
the portable helper over ordinary, subnormal and boundary exponents, including
100,003 seeded values. It passes all 1,004 dedicated checks per SIMD mode and all 1,152 untimed
corpus checks on Windows, minimum-NumPy Linux and modern-NumPy Linux. Its
expanded hosted, SDK and performance gates remain pending before retention.
