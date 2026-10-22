# Minimum-threshold optimization experiment

All headline observations use the same39 fixed one-worker workloads, including31 successful thresholds and8 matching convergence errors. Public calls include dispatch, histogram construction, required input copying, allocation, smoothing and exception/result destruction. The [complete JSON](minimum-tuning.json) retains every raw sample, measurement hash, source fingerprint, binary hash and slower case.

| Observation | Native geometric mean ms | Original/current | Preserved/current | Faster than original |
|---|---:|---:|---:|---:|
| initial | 0.348033 | 16.5940x | — | 39/39 |
| avx2_peaks | 0.250953 | 19.5475x | 1.1869x | 39/39 |
| independent_confirmation | 0.259137 | 19.8754x | 1.1901x | 39/39 |

The initial implementation already replaced the Python peak loop with native code. The next experiment compares eight adjacent bin pairs with AVX2, carries direction across equal-value plateaus and packet boundaries, and stops after a third peak is found. The portable loop uses the same early stop. Both preserve the ordered rolling double accumulator and float32 rounding after each smoothing pass.

The preserved baseline is the exact installed package identified by the initial measurement; it was not rebuilt. The paired harness validates its measurement hash, source fingerprint, binary map and ISA before measuring all three backends in randomized order. Separate-session absolute times vary; before/after claims use paired preserved/current ratios.

Independent confirmation includes separate one- and four-worker limits. Image histograms use optional global workers; ordered histogram smoothing and selection remain serial. [Parameter results](minimum-parameters.md) record every case.

| Worker limit | Successful thresholds | Original/current for successful thresholds | Matching convergence errors | Original/current for errors |
|---:|---:|---:|---:|---:|
| 1 | 31 | 31.8037x | 8 | 3.2150x |
| 4 | 31 | 40.3750x | 8 | 6.2300x |

Dedicated parity covers dtype/layout preservation, image and supplied histogram calls, read-only/unaligned/endian histogram storage, extreme count ratios, packet boundaries, scalar/empty inputs, exact iteration limits, ignored image arguments when a histogram is supplied, warning behavior and effectful iteration conversions. Raw C++ plateau oracles and guarded storage passed Linux address/undefined-behavior sanitizers with AVX2, disabled SIMD and a portable build. Unsupported input/metadata domains use explicit compatibility fallback; the documented native scope remains partial.

[Registered source CI](https://github.com/bipxx/scikit-imagexx/actions/runs/39709688860) and [AVX2 candidate CI](https://github.com/bipxx/scikit-imagexx/actions/runs/39708717908) are pending at this publication checkpoint. [Initial prototype CI](https://github.com/bipxx/scikit-imagexx/actions/runs/39706493010) passed its full Windows/Linux SIMD, portable C++, dynamic-team and minimum-NumPy matrix. The registered source passed 5,228 selected local tests in each SIMD mode; its C++ loops also passed the sanitizer checks described above.

Final cases slower than original scikit-image:

None in this confirmation.

Final one-worker cases slower than the initial C++ build:

| Dtype | Workload | Outcome | Workers | Preserved/current | Original/current |
|---|---|---|---:|---:|---:|
| uint8 | fortran | value | 1 | 0.9854x | 13.5262x |
| uint8 | stepped | value | 1 | 0.9868x | 10.5517x |
| uint8 | broadcast | value | 1 | 0.9378x | 5.4916x |
| uint8 | constant-failed | error | 1 | 0.9110x | 7.1447x |
| float32 | bimodal | value | 1 | 0.9468x | 6.4101x |
| float32 | seven-bins | error | 1 | 0.9575x | 1.6411x |
| float32 | constant-failed | error | 1 | 0.9758x | 2.8244x |
| float64 | reversed | value | 1 | 0.8762x | 4.3603x |
