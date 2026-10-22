# Minimum-threshold optimization experiment

All headline observations use the same39 fixed one-worker workloads, including31 successful thresholds and8 matching convergence errors. Public calls include dispatch, histogram construction, required input copying, allocation, smoothing and exception/result destruction. The [complete JSON](minimum-tuning.json) retains every raw sample, measurement hash, source fingerprint, binary hash and slower case.

| Observation | Native geometric mean ms | Original/current | Preserved/current | Faster than original |
|---|---:|---:|---:|---:|
| initial | 0.348033 | 16.5940x | — | 39/39 |
| avx2_peaks | 0.250953 | 19.5475x | 1.1869x | 39/39 |
| pre_warning_guard_confirmation | 0.259137 | 19.8754x | 1.1901x | 39/39 |
| independent_confirmation | 0.271167 | 18.5071x | 1.1699x | 39/39 |

The initial implementation already replaced the Python peak loop with native code. The next experiment compares eight adjacent bin pairs with AVX2, carries direction across equal-value plateaus and packet boundaries, and stops after a third peak is found. The portable loop uses the same early stop. Both preserve the ordered rolling double accumulator and float32 rounding after each smoothing pass.

A follow-up correctness check found that very large float64 supplied counts could overflow the private native-domain validation sum, producing an extra warning, callback or exception before upstream fallback. The final wrapper suppresses overflow only in that private check; upstream casting still observes the caller's warn/raise/call/print/log policy. Narrow-count validation keeps its original fast path. The prior confirmation remains recorded separately; this fix is retained for correctness independently of timing.

The preserved baseline is the exact installed package identified by the initial measurement; it was not rebuilt. The paired harness validates its measurement hash, source fingerprint, binary map and ISA before measuring all three backends in randomized order. Separate-session absolute times vary; before/after claims use paired preserved/current ratios.

Independent confirmation includes separate one- and four-worker limits. Image histograms use optional global workers; ordered histogram smoothing and selection remain serial. [Parameter results](minimum-parameters.md) record every case.

| Worker limit | Successful thresholds | Original/current for successful thresholds | Matching convergence errors | Original/current for errors |
|---:|---:|---:|---:|---:|
| 1 | 31 | 29.5781x | 8 | 3.0079x |
| 4 | 31 | 37.6845x | 8 | 6.1476x |

Dedicated parity covers dtype/layout preservation, image and supplied histogram calls, read-only/unaligned/endian histogram storage, extreme count ratios, packet boundaries, scalar/empty inputs, exact iteration limits, ignored image arguments when a histogram is supplied, warning behavior, five overflow policies across minimum/Yen/ISODATA and effectful iteration conversions. Raw C++ plateau oracles and guarded storage passed Linux address/undefined-behavior sanitizers with AVX2, disabled SIMD and a portable build. Unsupported input/metadata domains use explicit compatibility fallback; the documented native scope remains partial.

[Corrected source CI](https://github.com/bipxx/scikit-imagexx/actions/runs/39710832093) is pending at this publication checkpoint. [AVX2 candidate CI](https://github.com/bipxx/scikit-imagexx/actions/runs/39708717908) passed its full Windows/Linux SIMD, portable C++, dynamic-team and minimum-NumPy matrix. The corrected wrapper passed 6,334 selected local tests in each SIMD mode; its unchanged C++ loops also passed the sanitizer checks described above.

Final cases slower than original scikit-image:

None in this confirmation.

Final one-worker cases slower than the initial C++ build:

| Dtype | Workload | Outcome | Workers | Preserved/current | Original/current |
|---|---|---|---:|---:|---:|
| uint8 | broadcast | value | 1 | 0.9243x | 6.4223x |
| uint8 | iteration-limit | error | 1 | 0.9528x | 3.8657x |
| float32 | constant-failed | error | 1 | 0.9893x | 3.0672x |
| float64 | seven-bins | error | 1 | 0.9803x | 1.8866x |
| float64 | constant-failed | error | 1 | 0.9068x | 4.1226x |
