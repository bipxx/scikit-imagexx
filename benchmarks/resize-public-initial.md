# Public resize/rescale: one-worker baseline

The [completed isolated run](https://github.com/bipxx/scikit-imagexx/actions/runs/39957875987) measured source `66e5f78e4f5736597373bd5ccca0378f52d67980` against scikit-image 0.26.0. Each runtime passed all 373 strict checks before timing 372 cases with seven paired samples. One expected unsupported-endian-Half error is untimed. All samples and slower cases are retained.

| Runtime | Geometric mean speedup | Faster cases | Slower cases | Worst speedup |
|---|---:|---:|---:|---:|
| windows-modern | 1.318x | 323/372 | 49 | 0.285x |
| linux-modern | 1.431x | 320/372 | 52 | 0.324x |
| linux-minimum | 1.407x | 311/372 | 61 | 0.209x |

The primary result includes public Python dispatch, layout/endian normalization, temporary allocations, sampling, smoothing and clipping. It is not a kernel-only comparison. Reference thread pools and native workers were both limited to one; measurement hosts performed no builds or tests.

The slowest workloads are boolean reversed/broadcast/Fortran inputs. Nearest-neighbor paths, endian/broadcast storage and exceptional identity-range handling also contain slower cases. These are optimization targets; this baseline does not establish that every resize/rescale workload is faster.

The [summary and complete regression list](resize-public-initial.json) links all three raw reports and their hashes. The exact prepared packages, SDKs, dependency logs and source fingerprints are retained in the workflow artifacts and local protected baseline directories. No local timing calls were used.
