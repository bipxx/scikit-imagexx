# Frangi and ridge-Hessian benchmarks

`filters.frangi` and `filters.hessian` preserve scikit-image 0.26.0 signatures,
module exports and root convenience aliases. The retained runtime is
`064b83a5e1b30762ef3d74c1a7ecaf4b5ac0f3d8`; final measured revisions below
contain the same C++/Python/build sources and benchmark script, with subsequent
workflow/audit corrections only. Native-partial scope and explicit math
providers remain documented below.

## Final dispatch-compatible measurements

[Linux run39928998691](https://github.com/bipxx/scikit-imagexx/actions/runs/39928998691)
measures `f5d7d5fff520a28e9ff84193b0ab99e32550d58b`;
[Windows run39928737175](https://github.com/bipxx/scikit-imagexx/actions/runs/39928737175)
measures `74908fa7049bd65f7aa14a97243b694a6067ce7a`.
[Full CI39928737496](https://github.com/bipxx/scikit-imagexx/actions/runs/39928737496)
passes Windows, Linux and minimum NumPy for the measured Windows revision.
Runtime/build/benchmark sources are identical between both measurements and that
CI revision. Both artifacts pass frozen-source, exact wheel/package, binary,
complete corpus, raw sample, median, ratio and numerical-bound verification.

[Final Linux raw samples](frangi-dispatch-linux.json) and
[final Windows raw samples](frangi-dispatch-windows.json) retain all 1,152 rows
per OS, including worker caps one/four, seven samples per implementation,
and the preserved `841693b` baseline in the same process. Candidate and baseline
arrays compare exactly before timing each case. Normal float32 math dispatch
is `cpp-fma`; generic NumPy math dispatch is tested separately without timing.

| OS | API | Rank | One-worker geometric mean vs scikit-image | Minimum | Faster cases |
|---|---|---:|---:|---:|---:|
| Linux | Frangi | 2 | 6.039846x | 3.151641x | 144/144 |
| Linux | Frangi | 3 | 2.732022x | 1.313810x | 144/144 |
| Linux | Ridge-Hessian | 2 | 5.942942x | 3.136441x | 144/144 |
| Linux | Ridge-Hessian | 3 | 2.728814x | 1.306546x | 144/144 |
| Windows | Frangi | 2 | 4.620256x | 2.018954x | 144/144 |
| Windows | Frangi | 3 | 2.285534x | 1.280185x | 144/144 |
| Windows | Ridge-Hessian | 2 | 4.523177x | 2.019928x | 144/144 |
| Windows | Ridge-Hessian | 3 | 2.289118x | 1.265602x | 144/144 |

All 576 one-worker cases beat scikit-image on each OS. Overall geometric means
are **4.044561x Linux / 3.233638x Windows**. Across 96 float32-output cases,
same-runner gains over the preserved C++ baseline are **1.246366x / 1.167357x**;
individual paired minima are 0.955126x / 0.900568x. The 480 float64 controls
measure 0.997449x / 1.011386x. Aggregate improvements do not establish a gain
for every individual baseline comparison. Float32-output errors are zero;
maximum absolute 3-D Frangi float64 error is 2.220446049250313e-16 on both OSes,
and the other groups have zero recorded difference.

The fixed case-spec SHA256 remains
`5b56ac97111f39a52057aac850e4ab04c10bce80931b28adea07960e2e73357f`.
The ordinary images are 128x129 and 9x11x13; these measurements do not establish
large-volume throughput. Exact math-provider dependencies are disclosed below.

An earlier Windows audit failed on an assertion about an unused hardware
feature group. The matrix's default fail-fast then cancelled Linux after only
302 rows. That partial report is retained as failed evidence and excluded from
all results above. The audit now checks disabled compiled targets and actual
NumPy exp dispatch; independent benchmark jobs use `fail-fast: false`.

## Additional generic-dispatch compatibility gate

After these measurements, a NumPy generic-dispatch probe found two float32
ridge-Hessian mismatches on Linux in the measured `4dfa658` prototype. Its
normal-dispatch full CI and performance results remain recorded; those jobs
alone do not qualify generic NumPy dispatch. The correction passes the final measurements and full CI above.

Bindings now probe the active NumPy float-exp C loop near one at initialization.
When its rounding differs from the C++ approximation, a native float-exp
provider evaluates whole factor fields, preserving the planned narrow/wide
division and reusing scratch buffers. Matching runtimes keep the optimized C++
FMA or portable helper. This is native math reuse without Python image calls.
Standalone C++ exposes an optional `FrangiFloatExpProvider` as well. Benchmark
metadata records the actual float32 math provider.

The expanded untimed gate disables NumPy's AVX2/FMA/AVX-512 dispatch targets in
fresh processes and tests both C++ SIMD and scalar modes. NumPy 2.5's actual
exp dispatch target is recorded because hardware component bits can remain
true after its feature groups are disabled. Local normal and generic-dispatch
suites pass all 1,005 cases on Windows and both tested Linux NumPy runtimes;
hosted CI and final measurement gates now pass. Local SDK/native/portable and
sanitizer checks pass as well (13 native and 12 relocated SDK tests per
configuration).

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

Python bindings use native NumPy math providers, with no Python image
operations or GIL held during kernel execution:

- In 3-D, C++ packs symmetric double matrices in bounded chunks and calls the
  installed NumPy LAPACK `eigvalsh_lo` C loop. The 2-D solver uses a C++ formula.
- Float64 exponentials call NumPy's native exp C loop once per whole factor
  field, using separate input/output buffers. SIMD preparation and combination
  reuse the now-dead eigenvalue scratch buffer. Whole-field calls preserve
  host vector/tail dispatch independently of C++ worker partitions.

- Float32 uses the C++ FMA/portable helper when it matches the active NumPy
  loop. A near-one initialization probe selects a whole-field native NumPy
  float-exp provider when generic dispatch has different rounding.

These are dependencies on NumPy's native math runtime, not rewritten LAPACK
or a NumPy-free claim. Standalone C++ accepts optional eigenvalue and per-precision exp providers.
Its defaults use its own Jacobi/formula solver, libm double exp and the float32
approximation. Identical sensitive rounding is not claimed without matching
providers.

The baseline's float32 exp uses NumPy's BSD-licensed approximation and explicit
scalar FMA inside SIMD response arithmetic. A new candidate evaluates ordinary
float32 exponents with vector FMA after runtime CPU checks, preserving the
portable helper for subnormal/exceptional exponents and CPUs without FMA.
The paired measurements below quantify its gains. The existing NumPy
third-party license covers reused coefficients and evaluation order.

## Earlier SIMD FMA paired measurements (before generic-dispatch correction)

[Measurement39917576124](https://github.com/bipxx/scikit-imagexx/actions/runs/39917576124)
passed both OS jobs. Each runner compares scikit-image, this candidate and the
preserved whole-field baseline `841693b` in one process, using the same arrays,
reference pools of one, global C++ caps one/four, seven samples per implementation
and the unchanged case corpus. Candidate and baseline arrays must compare exactly
before timing every case. Both native providers are disclosed in report metadata,
and `frangi_fma=true` confirms the measured CPU dispatch. Exact source, current
and baseline wheel/package bytes, binary hashes, samples, medians, loop counts,
ratios, cases, dtypes and numerical bounds have been verified.

[Raw SIMD-FMA Linux samples](frangi-fma-linux.json) and
[raw SIMD-FMA Windows samples](frangi-fma-windows.json) retain all 1,152 rows per OS,
including all three sets of seven timing samples in every row.

| OS | API | Rank | One-worker geometric mean vs scikit-image | Minimum | Faster cases |
|---|---|---:|---:|---:|---:|
| Linux | Frangi | 2 | 6.066064x | 3.141553x | 144/144 |
| Linux | Frangi | 3 | 2.791526x | 1.292099x | 144/144 |
| Linux | Ridge-Hessian | 2 | 5.901990x | 3.058646x | 144/144 |
| Linux | Ridge-Hessian | 3 | 2.787797x | 1.310107x | 144/144 |
| Windows | Frangi | 2 | 4.952729x | 1.868207x | 144/144 |
| Windows | Frangi | 3 | 2.398190x | 1.328340x | 144/144 |
| Windows | Ridge-Hessian | 2 | 4.815697x | 1.857366x | 144/144 |
| Windows | Ridge-Hessian | 3 | 2.388229x | 1.316755x | 144/144 |

All 576 one-worker cases per OS are faster than scikit-image. Overall geometric
means are 4.085565x Linux / 3.418738x Windows. Strict budgets remain 2e-6 float32 /
2e-13 float64. Float32 outputs have zero recorded error; maximum absolute error
for 3-D Frangi float64 is 2.220446049250313e-16 Linux /
1.6653345369377348e-16 Windows; the other groups have zero recorded difference.

Across 96 float32-output one-worker cases, same-runner geometric mean gains over
the preserved C++ baseline are **1.253302x Linux / 1.179685x Windows**. Individual
paired minima are 0.988563x / 0.908922x; aggregate gains do not imply improvement
in every individual timing. The 480 unchanged float64-output controls measure
0.998112x / 1.016309x. Across all 576 cases, paired geometric mean gains are
1.036713x / 1.041875x.

The four previously slower Windows cases remain in the corpus. Their new
one-worker speedups and direct same-runner gains are:

| 2-D zero-sigma case | Linux vs scikit-image | Linux paired gain | Windows vs scikit-image | Windows paired gain |
|---|---:|---:|---:|---:|
| Frangi float16 | 5.258650x | 4.090554x | 3.377179x | 2.365240x |
| Frangi float32 | 5.276510x | 4.212234x | 3.265597x | 2.380186x |
| Ridge-Hessian float16 | 5.164442x | 4.041797x | 3.309298x | 2.357224x |
| Ridge-Hessian float32 | 5.222183x | 4.157446x | 3.265487x | 2.358065x |

These paired measurements establish improvement over the preserved C++ package
on the same runner. Separate runner trials below establish their own comparisons
against scikit-image; their ratios alone do not prove paired code improvement.

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
[audit39917576333](https://github.com/bipxx/scikit-imagexx/actions/runs/39917576333)
ran NumPy 1.24 and 2.5 on an actual AVX-512 CPU. Both expose five sensitive
points with bit-identical Hessians, and the native final responses match the
reference; both SIMD and scalar parity suites pass. Earlier successful AVX2
jobs alone were not treated as proof of the original AVX-512 case.

The new float32 vector-FMA candidate has additional bitwise comparisons with
the portable helper over ordinary, subnormal and boundary exponents, including
100,003 seeded values. It passes all 1,004 dedicated checks per SIMD mode and all 1,152 untimed
corpus checks on Windows, minimum-NumPy Linux and modern-NumPy Linux. Its
local native/portable/SDK/sanitizer and hosted AVX-512 parity gates pass. The
paired measurements above passed normal-dispatch CI; the generic-dispatch
correction supersedes that prototype for retention.

The final float-provider runtime passes 1,005 dedicated cases in normal SIMD,
normal scalar, generic-NumPy/C++-SIMD and generic-NumPy/C++-scalar configurations
on Windows, NumPy 1.24 Linux and NumPy 2.5 Linux. Hosted
[audit39928479221](https://github.com/bipxx/scikit-imagexx/actions/runs/39928479221)
passes both NumPy runtimes, including sequential runtimes on the same CPU and
all generic dispatch checks. That same-host CPU was AVX2; this job is not
represented as an AVX-512 reproduction. The current runtime's
[audit39927544130](https://github.com/bipxx/scikit-imagexx/actions/runs/39927544130)
also has an actual AVX-512 NumPy 2.5 job with five sensitive points: all six
Hessian fields are bit-identical and native final responses match upstream
exactly. The earlier FMA audit above retains the actual same-host AVX-512
NumPy 1.24/2.5 evidence for that earlier revision.
