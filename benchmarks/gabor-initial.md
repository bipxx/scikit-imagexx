# Gabor kernel experiments

The first native kernel source `9ac953b` fuses rotation, envelope evaluation,
complex modulation and normalization into one C++ traversal. AVX2 evaluates four
pixels at a time; independent rows optionally use the global worker cap. The
default cap stays one. No padded grids or full-size intermediate arrays are
allocated. `filters.gabor` image convolution remains pending native coverage.

[Initial run 39857475219](https://github.com/bipxx/scikit-imagexx/actions/runs/39857475219)
compares against scikit-image **0.26.0** using 36 fixed cases per OS, both complex
precisions, worker caps one/four, and seven raw samples per call. Cases include
tiny, anisotropic, phase-offset, broad-bandwidth, negative-frequency and large
kernels. Timing includes scalar planning, allocation, Python dispatch and output
destruction. Reference pools are fixed to one and native fallback is disabled.
The downloaded exact package, source manifest and measured binary hashes agree.

| OS | one-worker upstream/native geometric mean | minimum one-worker speedup | cases slower than upstream |
|---|---:|---:|---:|
| Linux | 3.191268x | 2.481019x | 0 / 36 |
| Windows | 3.632875x | 3.218904x | 0 / 36 |

Raw reports: [Linux](gabor-initial-linux.json), [Windows](gabor-initial-windows.json).
All 144 timed rows meet the declared tolerance. The maximum absolute complex
error is 6.939e-18; complex64 outputs are exact in this corpus. The general budget
is 3e-7 for complex64 or 3e-14 for complex128, relative plus absolute error scaled
by the largest kernel envelope. These measurements do not establish parity or
performance for every possible parameter combination.

Initial dedicated parity passes 430 cases per SIMD mode on Windows and minimum
NumPy. Native checks pass 2/2 on Windows, and 2/2 in Linux normal, dynamic-team,
portable and ASan/UBSan configurations (including dynamic teams under sanitizers).
The C++ test checks pinned numerical constants and worker consistency separately
from Python.

A subsequent scalar-promotion probe found that overflowed float16 powers and
wrapped negative integer powers could produce a misleading finite-core error.
The adapter now plans these scalar operations silently and delegates unsupported
results before replaying upstream's warnings/errors exactly once. NumPy 1.24's
wider scalar powers remain native. The ordinary Python scalar path avoids an
error-state context. Updated dedicated parity passes 440 cases per SIMD mode on
both runtimes, plus 4,755 API/metadata/namespace checks on each. C++ is unchanged.
This correction has passed fresh performance and full CI validation and is
retained for merge. The initial source, exact packages, failure probes and raw
measurements remain preserved; the initial timing is not attributed to the
corrected adapter.

## Corrected scalar planning

[Run 39858777112](https://github.com/bipxx/scikit-imagexx/actions/runs/39858777112)
measures corrected source `c2a57c8` on the same 36-case specification, with the
same worker caps and seven samples. Both OS jobs pass; downloaded source and
binary manifests match. All 144 timed rows satisfy parity and every one-worker
case is faster than original scikit-image.

| OS | one-worker upstream/native geometric mean | minimum one-worker speedup | cases slower than upstream |
|---|---:|---:|---:|
| Linux | 3.053686x | 2.667177x | 0 / 36 |
| Windows | 3.582205x | 2.678664x | 0 / 36 |

Raw reports: [Linux](gabor-corrected-linux.json), [Windows](gabor-corrected-windows.json).
The maximum absolute complex error remains 6.939e-18, with exact complex64
outputs in this corpus. These are independent upstream comparisons; the two
runs do not establish a paired performance change between native revisions.
[Final-source CI 39858776366](https://github.com/bipxx/scikit-imagexx/actions/runs/39858776366)
has passed all three jobs: minimum NumPy, Linux and Windows. Corrected source
`c2a57c8` is retained; source and exact measured packages remain preserved.
