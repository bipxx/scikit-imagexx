# Integral-image native-call experiment

Floating range validation and summation now use one Python-to-C++ call. C-contiguous arrays take a direct layout path and use the C++ kernel's implicit C-order plan; Fortran/permuted arrays retain explicit dense strides. Output allocation follows NumPy keep-order semantics. Unsupported floating ranges return an internal status before arithmetic and use the existing explicit fallback. The standalone C++ API is unchanged.

| Observation | Current C++ geometric mean, ms | Preserved C++ geometric mean, ms | Baseline/current | Faster than scikit-image |
|---|---:|---:|---:|---:|
| Paired run | 0.147774 | 0.150170 | 1.0162x | 146/148 |
| Independent confirmation | 0.184978 | 0.191669 | 1.0362x | 145/148 |

Kept source `bda6212`. Confirmation lowers combined-endpoint latency by 3.49% against the preserved `e2b5c5a` baseline. That baseline predates integer streaming too, so this figure does not isolate the binding change. The fixed aggregate mixes different shapes and dtypes. [All148 cases and raw samples](integral-binding-tuning.json) remain available.

Confirmation retains a uint8 single column (0.995x), int16 single row (0.891x), and int16 single column (0.886x). The first run retains a slower int64 long line and float16 single column. No adverse workloads or samples are removed. Ordinary system load and scheduling affect individual results.

Validation: 5,861 integral, signature, alias, metadata, inventory and backend-swap pipeline cases passed in each SIMD mode on Windows and Linux after the change. The arithmetic kernels and their previously passing native/portable C++ and sanitizer checks are unchanged by this binding experiment.

The preserved baseline and current C++ binaries run beside original scikit-image in one process with one worker each. Seven warmed samples use randomized backend order. Both implementations must match upstream exactly before timing. Source fingerprints and baseline binary hashes are retained in each snapshot. Reproduce with the preservation helper and commands in the [integer-prefix experiment](integral-integer-tuning.md).

This historical experiment is separate from the final [all-operation verification](STATUS.md). The full report set is rebuilt against the final source before merging.
