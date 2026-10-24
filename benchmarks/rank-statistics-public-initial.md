# Rank statistics: one-worker initial performance

Measured source: `ed8ca5cba87eeab4b1227a046316cb58ec587161`. [Exact GitHub Actions run](https://github.com/bipxx/scikit-imagexx/actions/runs/40049417301). Reference: scikit-image 0.26.0. Native workers: **1**; seven paired warmed samples. All three preparation jobs completed before fresh measurement hosts started.

Each profile passes 1,379 fixed canonical checks and four alias identities; 995 source cases and eight additional readonly errors remain untimed. Preparation requires workers 1/4 with normal SIMD and scalar dispatch. Each of four algorithms contributes 94 successful timing cases (376/profile). Timed scope includes public-call output allocation/reset/destruction; fixture construction is outside timing.

Ratios are reference time / C++ time; values below 1 are slower. These are per-host workload observations, with distinct CPU models across profiles. They do not establish full-library speed or a causal improvement over an older product.

| Algorithm | Windows / NumPy 2.5.4 | Linux / NumPy 2.5.4 | Linux / NumPy 1.24.4 |
|---|---:|---:|---:|
| `geometric_mean` | 4.040x | 3.723x | 4.076x |
| `noise_filter` | 1.554x | 1.659x | 2.029x |
| `subtract_mean` | 4.781x | 4.795x | 4.680x |
| `windowed_histogram` | 1.468x | 1.778x | 2.369x |

Geometric means above use 94 workloads per algorithm. Across all 376 calls, descriptive geometric means are 2.576x, 2.694x, 3.094x respectively. Slower observations are 56, 36, 28 of 376; all 120 are retained in the summary and unmodified raw reports.

## Workload cohorts

Pixel-count groups: tiny <200, small 200–4095, large ≥4096. Scalar and vector outputs, and 2D/3D inputs, remain separate.

| Cohort | Cases/profile | Windows / NumPy 2.5.4 | Linux / NumPy 2.5.4 | Linux / NumPy 1.24.4 |
|---|---:|---:|---:|---:|
| scalar / large / 2D | 24 | 1.403x | 1.578x | 1.330x |
| scalar / small / 2D | 150 | 3.670x | 3.750x | 4.247x |
| scalar / small / 3D | 60 | 2.276x | 2.088x | 2.211x |
| scalar / tiny / 3D | 48 | 4.058x | 3.886x | 4.505x |
| vector / large / 2D | 16 | 0.657x | 0.866x | 0.877x |
| vector / small / 2D | 78 | 1.731x | 2.061x | 2.904x |

Large vector-output workloads remain slower in all three profiles (16 cases/profile). Sparse-footprint `noise_filter` and stepped/uniform large images also appear among the slower cases. These workloads are queued for path/counter investigation; the ratios alone do not establish a specific internal cause.

## Exact evidence

All three artifact ZIP digests, 940 Git/sourceZIP members per profile, complete wheel RECORDs, four installed package trees, dependency library trees, 59 SDK members, canonical/alias admission, frozen original observations, paired sample order, medians and every ratio were independently checked offline. Each report serializes initial native workers 1 and one same-process CPU session for both backends. Runtime input wheels are retained (15 Windows / 14 Linux); other dependencies have version pins and archived wheel hashes. Four alias identities are serialized; their value comparisons execute but are not separately serialized. R4 does not contain the later R5/R6 untimed anti-forward profiler; exact wrapper/core source and owned compiled entries were audited.

| Profile | CPU | Slower / timed | Raw report |
|---|---|---:|---|
| Windows / NumPy 2.5.4 | Intel64 Family 6 Model 207 Stepping 2, GenuineIntel | 56/376 | [Raw samples](rank-statistics-public-windows-modern-initial.json) |
| Linux / NumPy 2.5.4 | AMD EPYC 9V74 80-Core Processor | 36/376 | [Raw samples](rank-statistics-public-linux-modern-initial.json) |
| Linux / NumPy 1.24.4 | AMD EPYC 7763 64-Core Processor | 28/376 | [Raw samples](rank-statistics-public-linux-minimum-initial.json) |

[Full summary and all regressions](rank-statistics-public-initial.json). Source, runtime and SDK hashes are embedded in the raw reports; retained artifact/proof locations are recorded in `coverage/evidence/rank-statistics-performance-initial.json`. No local operation timings were run.
