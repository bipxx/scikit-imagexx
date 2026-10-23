# Public resize/rescale: one-worker span-copy trial

The [completed isolated run](https://github.com/bipxx/scikit-imagexx/actions/runs/39987242238) measured source `b5d59bbaa71f4840264e756599cb5700c08407f6` against scikit-image 0.26.0. Each runtime passed all 373 strict checks before timing 372 cases with seven paired samples. One expected unsupported-endian-Half error is untimed. All samples and slower cases are retained.

| Runtime | Geometric mean speedup | Faster cases | Slower cases | Worst speedup |
|---|---:|---:|---:|---:|
| windows-modern | 1.270x | 314/372 | 58 | 0.493x |
| linux-modern | 1.489x | 317/372 | 55 | 0.497x |
| linux-minimum | 1.462x | 326/372 | 46 | 0.533x |

Hosted CPU models differ from the initial run. These original-library ratios are valid for the recorded hosts, but before/after ratios do not isolate the code change. A same-host previous/current/original comparison remains pending.

The primary result includes public Python dispatch, layout/endian normalization, temporary allocations, sampling, smoothing and clipping. It is not a kernel-only comparison. Reference thread pools and native workers were both limited to one; measurement hosts performed no builds or tests.

The slowest workloads are boolean reversed/broadcast/Fortran inputs. Nearest-neighbor paths, endian/broadcast storage and exceptional identity-range handling also contain slower cases. These are optimization targets; this baseline does not establish that every resize/rescale workload is faster.

The [summary and complete regression list](resize-public-span-copy-trial.json) links all three raw reports and their hashes. The exact prepared packages, SDKs, dependency logs and source fingerprints are retained in the workflow artifacts and local protected baseline directories. No local timing calls were used.
