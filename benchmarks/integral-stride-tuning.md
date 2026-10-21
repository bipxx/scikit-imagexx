# Integral-image strided-input and allocation experiment

Strided 64-bit single-chain inputs now use their raw byte strides, avoiding a dense copy. AVX2 gather, byte reversal and modular prefixes share the optional global worker limit. Normalization and keep-order output allocation run inside the existing native call, reducing Python setup for other input types too. The extension builds with NumPy 2 headers while retaining the declared NumPy 1.24 runtime minimum.

| Observation | Current ms, geometric mean | Preserved ms, geometric mean | Baseline/current | Faster than original | Faster than preserved |
|---|---:|---:|---:|---:|---:|
| raw_strided_prefix | 0.155222 | 0.155597 | 1.0024x | 147/148 | 67/148 |
| cached_tuple | 0.153156 | 0.153253 | 1.0006x | 146/148 | 64/148 |
| cached_plan | 0.157720 | 0.158818 | 1.0070x | 146/148 | 72/148 |
| cached_plan_confirmation | 0.153288 | 0.153141 | 0.9990x | 147/148 | 68/148 |
| fused | 0.164312 | 0.177249 | 1.0787x | 148/148 | 86/148 |
| fused_confirmation | 0.132023 | 0.141657 | 1.0730x | 147/148 | 87/148 |

Kept candidate `a25b740`, compared with preserved `6ed0541`. The first raw-prefix and Python-cache variants did not establish an aggregate gain. The final fused candidate improves the paired baseline/current ratio by 7.87% and 7.30% in independent runs. These ratios are speedups; confirmation reduces native aggregate latency by 6.80%. Absolute latency varies substantially between sessions, so only paired comparisons support this result.

[All six observations and raw samples](integral-stride-tuning.json) retain rejected variants, exact source fingerprints, preserved binary hashes and all slower cases. Workloads, seeds, repeat count and metric stayed fixed. Seven warmed medians use randomized backend order. No local builds or tests ran during measurement.

Workloads slower than the preserved C++ build in both final runs remain follow-up targets:

| Input | Workload | First baseline/current | Confirmation baseline/current |
|---|---|---:|---:|
| bool | stepped | 0.9905x | 0.9933x |
| bool | volume | 0.8567x | 0.9434x |
| bool | int64 | 0.9516x | 0.9784x |
| bool | uint64 | 0.9914x | 0.9308x |
| uint8 | volume | 0.9759x | 0.9541x |
| int8 | fortran | 0.9467x | 0.9245x |
| int8 | dtype-metadata | 0.9796x | 0.9800x |
| uint16 | stepped | 0.9395x | 0.9863x |
| uint16 | line | 0.9627x | 0.9378x |
| uint16 | volume | 0.9455x | 0.9166x |
| uint16 | int64 | 0.9627x | 0.9685x |
| int16 | reversed | 0.9654x | 0.9636x |
| int16 | volume | 0.9968x | 0.8677x |
| int16 | float64 | 0.9582x | 0.9259x |
| int16 | dtype-metadata | 0.9601x | 0.9139x |
| uint32 | volume | 0.9880x | 0.8493x |
| uint32 | int64 | 0.9869x | 0.9858x |
| uint32 | uint64 | 0.9504x | 0.9916x |
| int32 | fortran | 0.9997x | 0.9889x |
| int32 | int64 | 0.9097x | 0.9758x |
| int32 | uint64 | 0.9987x | 0.9009x |
| uint64 | reversed | 0.7924x | 0.9406x |
| uint64 | float64 | 0.7468x | 0.9247x |
| int64 | float64 | 0.9242x | 0.9771x |
| int64 | dtype-metadata | 0.8097x | 0.9847x |
| float32 | contiguous | 0.9846x | 0.9918x |
| float32 | line | 0.7019x | 0.9758x |
| float32 | dtype-metadata | 0.9200x | 0.9874x |
| float64 | contiguous | 0.9242x | 0.9636x |
| float64 | fortran | 0.9467x | 0.9782x |
| float64 | volume | 0.9585x | 0.9918x |

Slower than original scikit-image in either final run:

| Observation | Input | Workload | Original/current |
|---|---|---|---:|
| fused_confirmation | float64 | single-column | 0.9857x |

Validation: [Candidate CI](https://github.com/bipxx/scikit-imagexx/actions/runs/39677485689) passed 31,686 full-suite tests in each SIMD mode on each of Windows and Linux, original-library benchmark smoke checks, native/portable C++ checks, and dynamic OpenMP team checks. Its NumPy 1.24 job passed 1,582 tests using a wheel built against NumPy 2 headers. The full suite also passed locally. The 7,136-test selected integral/window, API, dtype and inventory suite passed in each SIMD mode on Windows and Linux, with worker limits one/four. A NumPy-2-built Linux extension also passed 520 exact cases under NumPy 1.26. Portable C++ previously passed Linux address/undefined-behavior sanitizers; the arithmetic sources are unchanged by the final binding experiment.

Reproduce by building a clean `6ed0541` wheel, preserving it with `tools/preserve_baseline.py`, building/installing `a25b740`, and running the recorded command twice without concurrent local builds/tests. This focused experiment is separate from the historical [all-operation checkpoint](STATUS.md).
