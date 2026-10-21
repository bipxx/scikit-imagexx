# Integral-window coordinate experiment

Coordinate setup dominated profiled single-window calls. Exact two-dimensional integer coordinate arrays now avoid initial copies. A broadcast shape view replaces tiling, equal shapes skip redundant broadcasting, and boolean checks use direct ndarray reductions. Corner arithmetic, output allocation and the C++ binary are unchanged. Other coordinate objects retain the original conversion path.

| Observation | Current C++ geometric mean, ms | Preserved C++ geometric mean, ms | Baseline/current | Faster than scikit-image | Faster than prior C++ |
|---|---:|---:|---:|---:|---:|
| paired | 0.049962 | 0.056192 | 1.1247x | 144/144 | 121/144 |
| independent_confirmation | 0.052286 | 0.059630 | 1.1405x | 144/144 | 124/144 |

Kept source `1252016`, compared with preserved `214e889`. Independent confirmation lowers aggregate latency by 12.32%. [All raw reports, profiles and samples](integrate-metadata-tuning.json) include the initial port baseline and both paired runs. Profiles diagnose call overhead; the performance claim uses the unprofiled paired measurements.

Five workloads are slower than the prior C++ build in both paired runs and remain follow-up targets:

| Input | Workload | First baseline/current | Confirmation baseline/current |
|---|---|---:|---:|
| int16 | zero-stride | 0.7597x | 0.8573x |
| int32 | volume | 0.8788x | 0.9707x |
| uint64 | volume | 0.9104x | 0.9763x |
| float32 | batch-4097 | 0.7759x | 0.9011x |
| float64 | batch-65539 | 0.8395x | 0.9354x |

The first run retains 23 slower old/new cases and confirmation retains 20. Every case remains faster than original scikit-image in both runs. No cases or raw samples were removed. Ordinary system load, allocation and scheduling affect individual timings; these measurements do not establish universal speedups.

Validation: 6,046 shared integral/window, public API, dtype metadata and alias tests passed in each SIMD mode on Windows with one/four workers. The C++ binary matches the preserved baseline byte for byte; its earlier Windows portable and Linux ASan/UBSan checks remain applicable. The initial port passed 31,029 tests in each SIMD mode on Windows and Linux plus native/portable C++ and benchmark smoke checks. [Candidate CI](https://github.com/bipxx/scikit-imagexx/actions/runs/39660931141) subsequently passed 31,047 tests in each SIMD mode on each of Windows and Linux, plus native/portable C++ and original-library benchmark smoke checks.

To reproduce, build a wheel from a clean `214e889` checkout, preserve it with `tools/preserve_baseline.py`, then rebuild/install `1252016` and run the command recorded in the JSON contract twice. The preservation helper checks Python sources and records source and binary hashes; the benchmark validates the preserved binary and exact output parity before timing. This experiment is separate from the [all-operation aggregate](STATUS.md).
