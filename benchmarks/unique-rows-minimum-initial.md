# UniqueRows: first minimum-profile comparison

**Overall slower across these nine workloads: 4 of 9 faster; geometric mean 0.910341233x.** This is the first whole-public-call original/current comparison for `util.unique_rows` at source `ecd2242e851641b137ec0a169623732b0ed5f8f8`, on one minimum-profile Linux host. Every slower case remains below.

[Raw report](unique-rows-minimum-initial.json) preserves all 126 timing samples and original numeric text. Speedup means scikit-image median / C++ public-call median; values below 1 indicate a loss. Added time is C++ minus original.

| Workload | dtype | N x D | Layout | Unique rows | Original ms | Current ms | Speedup | Added ms |
|---|---|---|---|---:|---:|---:|---:|---:|
| u8-short-small-unique | uint8 | 256 x 2 | C | 256 | 0.015730099 | 0.014541816 | 1.081714891x | -0.001188283 |
| u8-short-large-unique | uint8 | 65536 x 2 | C | 65536 | 8.529302500 | 11.874585000 | 0.718282155x | +3.345282500 |
| f32-coordinates-large-unique | float32 | 65536 x 3 | C | 65536 | 11.575462000 | 12.593178000 | 0.919185133x | +1.017716000 |
| f64-coordinates-duplicates | float64 | 65536 x 3 | C | 4096 | 13.871039000 | 12.484491000 | 1.111061636x | -1.386548000 |
| u8-wide32-common-prefix | uint8 | 32768 x 32 | C | 32768 | 14.052338000 | 6.386700000 | 2.200250207x | -7.665638000 |
| f64-wide64-all-equal | float64 | 16384 x 8 | C | 1 | 0.747689286 | 1.567377400 | 0.477032070x | +0.819688114 |
| f32-strided-duplicates | float32 | 16384 x 4 | row-and-column-step2 | 1024 | 2.265744857 | 2.747303333 | 0.824715942x | +0.481558476 |
| f64-fortran-duplicates | float64 | 8192 x 8 | F | 2048 | 1.618831000 | 1.449872636 | 1.116533245x | -0.168958364 |
| u8-readonly-duplicates | uint8 | 65536 x 4 | readonly | 256 | 5.852370667 | 10.452944000 | 0.559877740x | +4.600573333 |

| Cohort | Wins / cases | Geometric mean |
|---|---:|---:|
| duplicates | 2 / 4 | 0.869965140x |
| equal | 0 / 1 | 0.477032070x |
| prefix | 1 / 1 | 2.200250207x |
| unique | 1 / 3 | 0.893861624x |
| dtype float32 | 0 / 2 | 0.870670221x |
| dtype float64 | 2 / 3 | 0.839561399x |
| dtype uint8 | 2 / 4 | 0.989107031x |

All slower workloads:
- `u8-short-large-unique`: 0.718282155x, +3.345282500 ms.
- `f32-coordinates-large-unique`: 0.919185133x, +1.017716000 ms.
- `f64-wide64-all-equal`: 0.477032070x, +0.819688114 ms.
- `f32-strided-duplicates`: 0.824715942x, +0.481558476 ms.
- `u8-readonly-duplicates`: 0.559877740x, +4.600573333 ms.

## Measured boundary and correctness

Fixed seed 2026, seven paired rounds targeting 19 ms, one global worker and four reference pool limits of one. These are real N-by-D rows, with short unique rows, floating coordinates, duplicates, a 32-byte common prefix, all-equal 64-byte rows, strided input, Fortran input and read-only input. Inputs contain at most 65,536 rows and 1,048,576 elements.

Whole public calls include contiguous preparation/copies when needed, row view, native index allocation, byte sorting/equality with first-occurrence selection, the literal final NumPy advanced-index gather, and output destruction. Generation, the independent oracle and parity guards are outside timing. The final gather remains an explicit NumPy dtype/layout transfer provider; there is no owned-gather claim.

Before timing, the pinned original and current public API match an independent unsigned-byte row oracle in complete output bytes, dtype, metadata, shape, strides, ownership and layout. The genuine `unique_rows_apply` C return is observed while original API and NumPy unique forwarding are blocked. Actual native first indices are compared byte for byte against independently derived first-occurrence indices before timing; inputs and backing pixels remain unchanged. Hooks are restored before timers. Exported outputs and indices are hashes only: offline review authenticates source prechecks and hashes rather than reconstructing full arrays.

Installed parity passed 30 normal/scalar XML pairs (6,698 checks), color regression 3,696, matching 1,612, smoke 1,431 across 244 registered functions, plus 4/4/4/9 intake guards. Source census is 241 partially native image APIs plus three metadata APIs; full four-profile qualification remains **203 image APIs**. This first comparison provides no prior-C++ improvement, full-library speed or new full-profile qualification claim.

## Provenance

- [Genuine run 40695006125](https://github.com/bipxx/scikit-imagexx/actions/runs/40695006125): successful minimum producer followed by isolated measurement; 23 older timing steps skipped.
- [Retained artifact 12820230837](https://github.com/bipxx/scikit-imagexx/actions/runs/40695006125/artifacts/12820230837): 84,816,311 bytes, 86 members, downloaded once; all member sizes/CRC/SHA checked.
- Host: `Linux-7.0.0-1017-azure-x86_64-with-glibc2.39`; reported CPU architecture `x86_64`, model not recorded. CPython 3.11.17, NumPy 1.24.4, SciPy 1.15.3, scikit-image 0.26.0, AVX2 enabled. Timestamp: 2026-10-28T13:32:05.257840+00:00.
- Exact source, core, product/SDK and all 13 runtime dependency wheels match the prepared manifest and before/after identity receipts. Product wheel has 159 members and SDK 71; 1,756 committed source blobs authenticated.

| Proof | SHA-256 |
|---|---|
| Raw report | `35c9ae60d7f02cc09c9a712cde813dfe8d49ba9e4a46d5d670edeb5e6754bde3` |
| Artifact ZIP | `8f209d1ea6ce66efae8ce3709cf896e05c57ca2decfafb59f82ec9246c88ce8c` |
| Harness | `6aa6dfe18b02214236f7cc1bcab5bacc09b301a343a9370f50a01e2f1ecb9cbf` |
| Prospective workload contract | `1af9719b21d980ce68a34fd93e47eb537547caa89fa4df93a7e7ed658ac42674` |
| Original Python source | `72d88a45b800ed9adae7ec80e7c80aba9d1f92c3ee3d21d8731fcefc753bd751` |
| Installed native core | `fd52afb2ba6bad3a91dcf6e0adc5ee10f0754f664321d0746c9e13bb86b0d06b` |
| Canonical product source fingerprint | `f309d77013b500ee7cdf1e63f3a407466bcb5451ffd93dd6a27b7c900b9ccf45` |
| Independent review | `35731f3466ff1eb09b133eaa3a16b766c85478bbdff4997e9b8be0c862193abb` |

The first review stopped on a copied per-API smoke-count assertion (three instead of five nodes for the three-dtype UniqueRows/LabelPoints entries). Its failed receipt is retained; an additive metadata-only correction passed. The measured source, harness, reference records, sample values and correctness thresholds were unchanged, and the experiment was not repeated.
