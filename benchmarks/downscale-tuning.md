# Raw local-mean optimization experiment

The native port reads raw strided storage, applies virtual after-only padding and preserves upstream output layout and dtype metadata. Large-span AVX2 reductions removed repeated padding and scalar conversion. Small-block packets handle reversed and endian inputs; streaming unit factors avoid unnecessary reduction and half conversion. Parallel execution uses the optional global worker limit.

| Observation | C++ geometric mean ms | Original/current | Preserved/current | Faster than original |
|---|---:|---:|---:|---:|
| initial | 0.401951 | 1.6321x | — | 65/120 |
| cast_parity | 0.358403 | 1.5708x | 0.9960x | 65/120 |
| span_packets | 0.327825 | 1.7202x | 1.1043x | 69/120 |
| small_packets | 0.246254 | 2.4594x | 1.5063x | 110/120 |
| unit_streaming | 0.183466 | 3.0511x | 1.8398x | 120/120 |
| independent_confirmation | 0.186777 | 3.0362x | 1.8713x | 120/120 |

All rows above use one worker. Initial, intermediate and independent final observations retain every raw sample and slower case in [the experiment JSON](downscale-tuning.json). The padding-cast fix was required for correctness and established no speed gain. The small-packet experiment regressed unit-factor conversion; those ten slower-than-original cases were retained and addressed by the streaming path. Absolute times vary between sessions, so improvements over the initial C++ build use paired measurements.

The final confirmation also records four-worker results in the experiment JSON, separately from the fixed one-worker metric. The [parameter report](downscale-parameters.md) is refreshed by whole-library verification and records its explicit worker settings. Speedups describe these seeded workloads on the recorded CPU, not every supported shape or parameter.

The independent final one-worker result is 3.0362x original and 1.8713x the preserved initial C++ build, reducing aggregate native latency by 46.56%. All 120 cases beat original in each final run. Four workers achieve 5.5235x original across their separate 120 cases. Forty-seven serial cases remain slower than the initial C++ build despite the aggregate improvement; they are listed below as follow-up targets.

Validation: the selected 7,749-test API, dtype, inventory and local-mean suite passed with SIMD enabled and disabled on Windows and Linux. The 2,767 dedicated cases include worker limits one/four, all real dtype widths, exact half rounding, warning/callback behavior, negative-zero handling, layout and endian tails. A protected-memory subprocess checks allocation metadata without reading pixels. AVX2 and portable native checks passed Linux address/undefined-behavior sanitizers. [Registered API candidate CI](https://github.com/bipxx/scikit-imagexx/actions/runs/39692001820) passed 34,485 full-suite tests in each SIMD mode on each OS, original-library benchmark smokes, native/portable C++ and dynamic-team checks. Its NumPy 1.24 job passed 4,349 tests using a wheel built against NumPy 2 headers. The preceding [arithmetic candidate CI](https://github.com/bipxx/scikit-imagexx/actions/runs/39691151992) passed 34,453 tests in each SIMD mode on each OS before registration.

Nonfinite/risky floating ranges, nonscalar constants, input subclasses and some nondefault floating policies retain explicit fallback. Reviewed coverage is partial. The preserved initial measured package has revision `2025094`, source and binary hashes embedded in the reports; it was not rebuilt after its baseline measurement.

Final one-worker cases slower than the preserved initial C++ build:

| Input | Workload | Preserved/current | Original/current |
|---|---|---:|---:|
| bool | factor2 | 0.5796x | 2.8912x |
| bool | factor4 | 0.6371x | 2.6937x |
| bool | fortran | 0.7074x | 3.1115x |
| bool | single-row | 0.9530x | 6.1162x |
| uint8 | factor2 | 0.6431x | 2.3800x |
| uint8 | factor4 | 0.6660x | 2.8647x |
| uint8 | fortran | 0.6869x | 2.9292x |
| uint8 | single-row | 0.7308x | 4.2064x |
| int8 | factor2 | 0.7255x | 3.0887x |
| int8 | factor4 | 0.7196x | 2.9233x |
| int8 | fortran | 0.7189x | 2.7350x |
| int8 | single-row | 0.7972x | 4.6409x |
| uint16 | factor2 | 0.7175x | 2.7493x |
| uint16 | factor4 | 0.7220x | 2.8363x |
| uint16 | fortran | 0.6915x | 2.8555x |
| int16 | factor2 | 0.6383x | 2.9004x |
| int16 | factor4 | 0.6745x | 2.7721x |
| int16 | fortran | 0.6010x | 2.7718x |
| int16 | single-row | 0.8088x | 4.0219x |
| uint32 | factor2 | 0.6321x | 3.0901x |
| uint32 | factor4 | 0.8375x | 2.8194x |
| uint32 | fortran | 0.6262x | 2.9240x |
| uint32 | single-row | 0.8261x | 4.7628x |
| int32 | factor2 | 0.6763x | 2.5907x |
| int32 | factor4 | 0.6620x | 2.6304x |
| int32 | fortran | 0.6519x | 2.5706x |
| int32 | single-row | 0.7980x | 4.0438x |
| uint64 | factor2 | 0.6117x | 2.4702x |
| uint64 | factor4 | 0.7203x | 2.5229x |
| uint64 | fortran | 0.6191x | 2.7627x |
| uint64 | single-row | 0.9932x | 4.1617x |
| int64 | factor2 | 0.6400x | 2.4794x |
| int64 | factor4 | 0.6973x | 2.9142x |
| int64 | fortran | 0.6296x | 2.6564x |
| int64 | single-row | 0.8456x | 4.8068x |
| float16 | factor2 | 0.8281x | 3.1420x |
| float16 | factor4 | 0.7937x | 3.6147x |
| float16 | fortran | 0.7253x | 3.8397x |
| float16 | single-row | 0.9006x | 4.2913x |
| float32 | factor2 | 0.4536x | 4.8092x |
| float32 | factor4 | 0.6775x | 5.0139x |
| float32 | fortran | 0.4859x | 7.0193x |
| float32 | single-row | 0.8279x | 4.8171x |
| float64 | factor2 | 0.6027x | 3.6073x |
| float64 | factor4 | 0.6542x | 3.9080x |
| float64 | fortran | 0.6309x | 4.3665x |
| float64 | single-row | 0.9802x | 4.5043x |

Final cases slower than original scikit-image:

| Input | Workload | Workers | Original/current |
|---|---|---:|---:|

None in this confirmation.
