# Label-color optimization experiment

The fixed core suite contains 101 workloads: label widths and layouts, custom backgrounds and axes, grayscale/RGB overlays, desaturation and grouped averages. Public timings include normalization, rank construction, copies, allocation, HSV processing and output destruction. Reference pools use one thread; native worker limits are explicit. The [complete tuning JSON](label-color-tuning.json) retains all observations, source/binary hashes, measurement hashes and raw samples.

| Observation | Original/current,1 worker | Initial/current,paired | Faster than original |
|---|---:|---:|---:|
| initial | 1.8935x | — | 101/101 |
| dense_lookup | 6.6745x | 3.3400x | 101/101 |
| sparse_groups | 6.9953x | 3.6956x | 101/101 |
| pre_growth_confirmation | 6.5142x | 3.1486x | 101/101 |
| adaptive_confirmation | 6.6606x | 3.4424x | 101/101 |

Dense labels use a bounded rank table instead of sorting N repeated values and binary-searching per pixel. AVX2 gathers read four ranks at a time; portable code and scalar tails use identical table values. Sparse labels use a 1024-slot table for up to 512 distinct keys, growing on demand to 8192 slots for up to 4096 keys. Only distinct keys are sorted; higher-cardinality inputs retain the general sorted path. Growth removes the measured 513-key cliff. The stress audit improves 1.7289x against the exact pre-growth package, with all 15 cases faster than scikit-image. Eight stress cases remain slower than the pre-growth package and are listed below. Validation precedes output writes. Optional workers honor the global limit, which defaults to 1.

The preserved baseline is the exact measured initial package, not a rebuild. Paired measurements validate its source/measurement/binary hashes and ISA, then randomize the order of all three backends. Absolute timings across sessions vary; before/after claims use paired ratios.

Independent confirmation is recorded in the [parameter report](label-color-parameters.md).

| Worker limit | Cases | Original/current | Initial/current,paired | Faster than original |
|---:|---:|---:|---:|---:|
| 1 | 101 | 6.6606x | 3.4424x | 101/101 |
| 4 | 101 | 8.7414x | 3.1584x | 101/101 |

The separate [cardinality audit](label-color-cardinality.md) checks 15 wide-label cases, including the 512/513-key growth boundary and 4096/high-cardinality fallback. The stress report compares against the preserved pre-growth package; core reports compare against the initial package. The pre-growth stress report, including all three cases slower than scikit-image, is embedded in the tuning JSON. These cases were added after tuning and do not change the 101-case optimization metric. All raw samples and adverse observations remain available.

Parity covers 11 real label dtypes, strides, endian/unaligned storage, scalar/empty behavior, channels, custom colors/backgrounds, ignored controls, typed means, large groups, explicit fallback and warnings. Grayscale and average results are exact; RGB overlays inherit the documented HSV error budget. NumPy before 2.3 uses buffer-sized contiguous reductions; native averages preserve that ordering and the caller's current buffer size. Raw C++ contracts passed address/undefined-behavior sanitizers in AVX2, disabled-SIMD and portable OpenMP-disabled builds. The pre-growth checkpoint passed 8,062 shared API/color tests in each SIMD mode. The adaptive source passed 1,480 dedicated parity tests in each SIMD mode and 1,480 minimum-NumPy cases; raw C++ growth boundaries also passed the sanitizer matrix. Export registration also corrected operation IDs to be independent of public export order.

Final source [CI](https://github.com/bipxx/scikit-imagexx/actions/runs/39720117307) was still running when this report was prepared.

Cases slower than original or the preserved comparator in independent confirmation (initial comparator for core cases; pre-growth comparator for stress cases):

| Input | Workload | Workers | Original/current | Preserved/current |
|---|---|---:|---:|---:|
| uint64 | sparse-groups-31 | 1 | 2.4139x | 0.9717x |
| uint64 | sparse-groups-512 | 1 | 2.4109x | 0.9732x |
| uint64 | sparse-groups-17157 | 1 | 1.9941x | 0.9659x |
| int64 | sparse-groups-31 | 1 | 2.5621x | 0.9577x |
| int64 | sparse-groups-512 | 1 | 2.2735x | 0.8925x |
| int64 | sparse-groups-17157 | 1 | 1.9774x | 0.9543x |
| float64 | sparse-groups-31 | 1 | 2.4491x | 0.9879x |
| float64 | sparse-groups-17157 | 1 | 1.9500x | 0.9607x |

Earlier adverse paired observations:

| Observation | Input | Workload | Initial/current | Original/current |
|---|---|---|---:|---:|
| dense_lookup | int32 | sparse-wide | 0.9997x | 1.3760x |
| dense_lookup | uint64 | sparse-wide | 0.9412x | 1.4270x |

Adverse observations from the pre-growth cardinality audit:

| Input | Workload | Original/pre-growth | Initial/pre-growth |
|---|---|---:|---:|
| uint64 | sparse-groups-513 | 0.7020x | 0.9528x |
| uint64 | sparse-groups-4096 | 1.2771x | 0.9776x |
| int64 | sparse-groups-513 | 0.7591x | 0.9984x |
| float64 | sparse-groups-513 | 0.8937x | 1.0677x |
| float64 | sparse-groups-4096 | 1.3578x | 0.8916x |
