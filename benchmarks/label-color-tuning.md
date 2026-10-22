# Label-color optimization experiment

The fixed core suite contains 101 workloads: label widths and layouts, custom backgrounds and axes, grayscale/RGB overlays, desaturation and grouped averages. Public timings include normalization, rank construction, copies, allocation, HSV processing and output destruction. Reference pools use one thread; native worker limits are explicit. The [complete tuning JSON](label-color-tuning.json) retains all observations, source/binary hashes, measurement hashes and raw samples.

| Observation | Original/current,1 worker | Initial/current,paired | Faster than original |
|---|---:|---:|---:|
| initial | 1.8935x | — | 101/101 |
| dense_lookup | 6.6745x | 3.3400x | 101/101 |
| sparse_groups | 6.9953x | 3.6956x | 101/101 |
| independent_confirmation | 6.5142x | 3.1486x | 101/101 |

Dense labels use a bounded rank table instead of sorting N repeated values and binary-searching per pixel. AVX2 gathers read four ranks at a time; portable code and scalar tails use identical table values. Sparse labels with up to 512 distinct keys use a bounded 1024-slot hash table and sort only the distinct keys. Higher-cardinality inputs retain the general sorted path. Validation precedes output writes. Optional workers honor the global limit, which defaults to 1.

The preserved baseline is the exact measured initial package, not a rebuild. Paired measurements validate its source/measurement/binary hashes and ISA, then randomize the order of all three backends. Absolute timings across sessions vary; before/after claims use paired ratios.

Independent confirmation is recorded in the [parameter report](label-color-parameters.md).

| Worker limit | Cases | Original/current | Initial/current,paired | Faster than original |
|---:|---:|---:|---:|---:|
| 1 | 101 | 6.5142x | 3.1486x | 101/101 |
| 4 | 101 | 8.6668x | 2.9402x | 101/101 |

The separate [cardinality audit](label-color-cardinality.md) checks 15 wide-label cases, including the 512/513-key hash boundary and 4096/high-cardinality fallback. These cases were added after tuning and do not change the 101-case optimization metric. All raw samples and adverse observations remain available.

Parity covers 11 real label dtypes, strides, endian/unaligned storage, scalar/empty behavior, channels, custom colors/backgrounds, ignored controls, typed means, large groups, explicit fallback and warnings. Grayscale and average results are exact; RGB overlays inherit the documented HSV error budget. NumPy before 2.3 uses buffer-sized contiguous reductions; native averages preserve that ordering and the caller's current buffer size. Raw C++ contracts passed address/undefined-behavior sanitizers in AVX2, disabled-SIMD and portable OpenMP-disabled builds. Local shared API/color tests passed 8,062 cases in each SIMD mode; minimum-NumPy tests passed 1,456 cases. Export registration also corrected operation IDs to be independent of public export order.

Final source [CI](https://github.com/bipxx/scikit-imagexx/actions/runs/39718980690) was still running when this report was prepared.

Cases slower than original or the initial C++ build in independent confirmation:

| Input | Workload | Workers | Original/current | Initial/current |
|---|---|---:|---:|---:|
| uint64 | sparse-groups-513 | 1 | 0.7020x | 0.9528x |
| uint64 | sparse-groups-4096 | 1 | 1.2771x | 0.9776x |
| int64 | sparse-groups-513 | 1 | 0.7591x | 0.9984x |
| float64 | sparse-groups-513 | 1 | 0.8937x | 1.0677x |
| float64 | sparse-groups-4096 | 1 | 1.3578x | 0.8916x |

Earlier adverse paired observations:

| Observation | Input | Workload | Initial/current | Original/current |
|---|---|---|---:|---:|
| dense_lookup | int32 | sparse-wide | 0.9997x | 1.3760x |
| dense_lookup | uint64 | sparse-wide | 0.9412x | 1.4270x |
