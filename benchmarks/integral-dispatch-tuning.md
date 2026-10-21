# Integral-image dispatch experiment

The fixed148-call workload covers twelve real input dtypes, ordinary/Fortran/reversed/stepped storage, 1D lines, thin rows/columns, volumes, explicit target precision, dtype tags and endian storage. Both libraries use one thread. Each call requires exact values, output dtype/metadata, physical strides and warnings before timing. Seven warmed samples include allocation, conversion, copying, dispatch and destruction.

| Observation | Source | Native geometric-mean latency, ms | Faster than scikit-image |
|---|---|---:|---:|
| Baseline | `7dd1136` | 0.250602 | 123/148 |
| Dispatch optimization | `e2b5c5a` | 0.176815 | 139/148 |
| Independent confirmation | `e2b5c5a` | 0.189932 | 142/148 |

Keep: confirmation reduces the fixed aggregate native latency by 24.21%. The aggregate mixes different shapes and dtypes; it is not the latency of one representative image. [All148 cases and raw samples](integral-dispatch-tuning.json) remain available.

The canonical untagged path uses precomputed NumPy dtype defaults. Tagged descriptors and unsupported dtype validation retain an empty metadata-only probe. Scalar validation order is unchanged. Cached dtype maxima and scalar finite checks remove additional Python/NumPy overhead. Image conversion and ordered cumulative arithmetic still run in C++; the C++ kernels do not change in this experiment.

Confirmation retains slower uint8/uint32/float16/float64 single columns and int64/uint64 long lines. The source adds no speed-ratio gate. Individual near-parity results can change with system load and scheduling.

Validation: 5,617 targeted integral, signature, alias, dtype-metadata, inventory and backend-swap pipeline cases passed in each SIMD mode on Windows and Linux after optimization. The baseline complete [Windows/Linux CI run](https://github.com/bipxx/scikit-imagexx/actions/runs/39646722531) passed30,167 cases per SIMD mode on each system, plus native and AVX2/OpenMP-disabled C++ checks. Local Linux AddressSanitizer/UndefinedBehaviorSanitizer checks passed; C++ source is unchanged by the dispatch optimization.

Reproduce with `python benchmarks/integral.py --threads 1 --repeat 7`. Finish all local builds/tests before timing. These snapshots preserve a fixed historical experiment; the full verification aggregate is regenerated separately after the final kept implementation.
