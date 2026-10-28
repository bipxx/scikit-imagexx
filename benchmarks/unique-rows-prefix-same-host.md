# UniqueRows: prefix-cache candidate on the same host

**KEEP under the fixed contract.** Across nine public workloads, geometric-mean gain over the C++ baseline is **1.851910811x direct / 1.860385115x normalized**. The balanced original/current ratio changes from 0.832751797x baseline to 1.549239047x candidate on this host. All three targets repeat above 1.10x; all four previous original wins remain wins in both candidate passes; both float controls pass the inclusive 0.97x floor.

**Remaining original loss:** read-only uint8 duplicates are still slower than scikit-image in both candidate passes (0.853826662x / 0.857999083x, +1.090820000 / +1.062475500 ms); balanced ratio 0.855910330x. All nine balanced C++ direct and normalized gains exceed 1, but two controls have per-pass C++ regressions listed below. This is not an all-faster or full-library claim.

[All four raw reports and comparison](unique-rows-prefix-same-host.json) preserve **504 samples**, original numeric text, input/output and native-index hashes, metadata, medians and loop counts. The [first mixed result](unique-rows-minimum-initial.md) remains published.

| Workload | Baseline C++ ms (GM) | Candidate C++ ms (GM) | Original / baseline | Original / candidate | Direct gain | Normalized gain | Added ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| u8-short-small-unique | 0.015102699 | 0.008696399 | 0.979287344x | 1.694219818x | 1.736661190x | 1.730053828x | -0.006406300 |
| u8-short-large-unique | 13.194537499 | 7.162996489 | 0.690208315x | 1.275438514x | 1.842041598x | 1.847903720x | -6.031541010 |
| f32-coordinates-large-unique | 13.714774146 | 7.543315696 | 0.811022978x | 1.493002757x | 1.818136042x | 1.840888357x | -6.171458450 |
| f64-coordinates-duplicates | 13.451723789 | 13.202932407 | 1.037073874x | 1.063759408x | 1.018843646x | 1.025731565x | -0.248791382 |
| u8-wide32-common-prefix | 6.608266455 | 6.379504669 | 1.928382119x | 2.059952873x | 1.035858863x | 1.068228570x | -0.228761786 |
| f64-wide64-all-equal | 1.768742659 | 0.104675330 | 0.372572407x | 6.191545925x | 16.897416647x | 16.618369505x | -1.664067329 |
| f32-strided-duplicates | 2.940216064 | 1.901927763 | 0.799662139x | 1.233056579x | 1.545913637x | 1.541971939x | -1.038288302 |
| f64-fortran-duplicates | 1.577044517 | 1.525532756 | 1.064360270x | 1.112922556x | 1.033766408x | 1.045625798x | -0.051511761 |
| u8-readonly-duplicates | 11.549414289 | 7.472335282 | 0.553997814x | 0.855910330x | 1.545623136x | 1.544970589x | -4.077079007 |

Original ratios mean scikit-image time / current time; below 1 is slower than original. Direct gain compares C++ baselines to candidate; below 1 is a C++ regression. Added time is candidate minus baseline. The table uses geometric means of the two respective passes; every individual median remains in the raw reports.

| Workload | Original reference ms during baseline (GM) | Original reference ms during candidate (GM) | Original / candidate pass 2 | Original / candidate pass 3 |
|---|---:|---:|---:|---:|
| u8-short-small-unique | 0.014789882 | 0.014733612 | 1.686951286x | 1.701519668x |
| u8-short-large-unique | 9.106979501 | 9.135961595 | 1.274056910x | 1.276821616x |
| f32-coordinates-large-unique | 11.122996967 | 11.262191134 | 1.490274848x | 1.495735660x |
| f64-coordinates-duplicates | 13.950431296 | 14.044743559 | 1.056934833x | 1.070628049x |
| u8-wide32-common-prefix | 12.743262866 | 13.141478969 | 1.996796557x | 2.125106748x |
| f64-wide64-all-equal | 0.658984710 | 0.648102111 | 6.442176077x | 5.950666433x |
| f32-strided-duplicates | 2.351179468 | 2.345184541 | 1.233378944x | 1.232734299x |
| f64-fortran-duplicates | 1.678543528 | 1.697799815 | 1.058820502x | 1.169789038x |
| u8-readonly-duplicates | 6.398350273 | 6.395648955 | 0.853826662x | 0.857999083x |

All five loss cases from the separate first Release are retained here. Initial ratios come from that earlier host and are context, not the denominator of the same-host gain.

| Initial loss case | First Release original/current | Same-host baseline original/current | Same-host candidate original/current | Candidate original win in both passes? |
|---|---:|---:|---:|---|
| u8-short-large-unique | 0.718282155x | 0.690208315x | 1.275438514x | yes |
| f32-coordinates-large-unique | 0.919185133x | 0.811022978x | 1.493002757x | yes |
| f64-wide64-all-equal | 0.477032070x | 0.372572407x | 6.191545925x | yes |
| f32-strided-duplicates | 0.824715942x | 0.799662139x | 1.233056579x | yes |
| u8-readonly-duplicates | 0.559877740x | 0.553997814x | 0.855910330x | no |

Per-pass C++ regressions against the geometric mean of the two baselines:

| Control / pass | Direct gain | Normalized gain | Scope |
|---|---:|---:|---|
| u8-wide32-common-prefix, pass 2 | 0.998405977x | 1.035477636x | Balanced gain and both original wins still pass |
| f64-fortran-duplicates, pass 2 | 0.984723155x | 0.994795214x | Balanced gain and both original wins still pass |

| Group | Cases | Direct gain | Normalized gain | Original / candidate |
|---|---:|---:|---:|---:|
| targets | 3 | 3.636981724x | 3.620172979x | 1.890727602x |
| preserved_original_wins | 4 | 1.173238783x | 1.186542924x | 1.425718714x |
| remaining_float_loss_controls | 2 | 1.676508664x | 1.684813993x | 1.356818659x |
| duplicates | 4 | 1.259518929x | 1.264304473x | 1.057255390x |
| equal | 1 | 16.897416647x | 16.618369505x | 6.191545925x |
| prefix | 1 | 1.035858863x | 1.068228570x | 2.059952873x |
| unique | 3 | 1.798375192x | 1.805463871x | 1.477621828x |
| float32 | 2 | 1.676508664x | 1.684813993x | 1.356818659x |
| float64 | 3 | 2.610859030x | 2.612158190x | 1.942536341x |
| uint8 | 4 | 1.504370062x | 1.515586891x | 1.397103914x |

## Frozen decision and public-call boundary

One candidate was frozen at 2026-10-28T13:55:39.787189Z; deadline 15:55:39.787189Z. Metrics completed at 14:43:23.492137Z. Dispatch-to-job-start delay was **372 seconds** (14:35:41Z to 14:41:53Z), consuming the existing budget; 4,336.295052 seconds remained at completion. The clock and gates were not extended.

KEEP requires overall direct and normalized geometric means strictly above 1.05x. Each candidate pass for the three targets (all-equal 64-byte rows, read-only uint8 duplicates, large unique uint8 rows) must improve by more than 1.10x against the corresponding baseline geometric mean. All four previous original wins must remain above 1 in both candidate passes. The two remaining float32 loss controls must have balanced direct and normalized gains at least 0.97x. Correctness/identity gates and the deadline must pass. All observed gates passed; an all-original-wins condition was never added.

The candidate scans for all-equal rows before sorting and key allocation. Other rows of at most 16 bytes cache the first up-to-eight bytes as an unsigned lexical key; equal keys compare any remaining tail. Wider rows retain the previous comparator/SIMD sort path and avoid key allocation. The temporary key array costs 8N bytes (at most 512 KiB here), with allocation, initialization and packing inside timing. Original-index ties, first occurrence, chunk sorting/merging and optional global workers remain fixed. The unchanged wide SIMD object and public native ABI were verified. Observed gains belong to the whole bundle; no measured per-stage causal contribution is claimed.

The fixed order is baseline/candidate/candidate/baseline on one host, each report containing nine workloads and seven paired rounds targeting 19 ms. Global workers and all four reference pools are one. Whole public calls include contiguous preparation/copy, row view, index allocation, sorting/equality, the literal final NumPy advanced-index gather, other allocation and output destruction. The final gather remains an explicit dtype/layout transfer provider.

Before every report timer, the unchanged harness verifies complete output bytes and dtype/layout/ownership metadata against pinned original and an independent unsigned-byte row oracle; actual native first indices match independent first-occurrence indices byte for byte. Genuine `xx._core.unique_rows_apply` returns are observed with original and NumPy-unique forwarding blocked, then hooks restored. Inputs/backing pixels stay unchanged. Exports are hash-only payloads; offline review authenticates the source assertions, matching input/output/index hashes, all 504 samples/medians, both original product identities, and eight before/after identities.

All six native modes passed on the first attempt: normal 11,860, scalar 10,306, sanitizer 11,860/10,306 and portable 5,154 each. All 86 frozen returns, old native word/FP control streams, 416 saved private observations and 344 native acceptances remain exact; private/public gates total 856. Both genuine product preparations passed 30 normal/scalar XML pairs (6,698), color 3,696, matching 1,612, smoke 1,431 across 244 registered functions and 4/4/4/9 intake guards. Source census is 241 partially native image APIs plus three metadata APIs. Full four-profile qualification remains **203 image APIs**; this comparison grants no additional promotion.

## Provenance

- Baseline `ecd2242e851641b137ec0a169623732b0ed5f8f8`; candidate `d8adeb419d72bf284f1b4a04ce20e32a9be52074`, imported into main `9939eba` after independent KEEP review. Only the native leaf/test and prospective candidate contract differ; header, SIMD source, binder, Python, CMake, original source and harness stay fixed.
- [Single ABBA run 40707480850](https://github.com/bipxx/scikit-imagexx/actions/runs/40707480850), controller `4bc5a099ae79f899699ca91cc2980c3e1ec66319`.
- [Baseline Release 40695006125](https://github.com/bipxx/scikit-imagexx/actions/runs/40695006125) / [artifact 12820230837](https://github.com/bipxx/scikit-imagexx/actions/runs/40695006125/artifacts/12820230837).
- [Candidate preparation 40702637014](https://github.com/bipxx/scikit-imagexx/actions/runs/40702637014) / [artifact 12823857042](https://github.com/bipxx/scikit-imagexx/actions/runs/40702637014/artifacts/12823857042), all 24 timing steps skipped.
- [Comparison artifact 12824517648](https://github.com/bipxx/scikit-imagexx/actions/runs/40707480850/artifacts/12824517648): 40,098 bytes, 39 members, downloaded once; all member sizes/CRC/SHA verified.
- CPU: Intel Xeon Platinum 8573C. CPython 3.11.17, NumPy 1.24.4, SciPy 1.15.3, scikit-image 0.26.0; AVX2 enabled. Exact common 13 dependency wheels and scientific package/library hashes, with no normalization. Each product wheel has 159 members and SDK 71. Baseline source has 1,756 committed blobs; candidate 1,757.

| Proof | SHA-256 |
|---|---|
| Combined raw publication | `f19828543917df13099c6e4a3dca16f1464fe1cdadce4ed2adb6e8e12eb414a3` |
| Comparison ZIP | `548a40d970ee1b7becf1d8294b47e7fb0f689a9cb495d65eef68c9c5e26bcc59` |
| Baseline ZIP | `8f209d1ea6ce66efae8ce3709cf896e05c57ca2decfafb59f82ec9246c88ce8c` |
| Candidate ZIP | `355736e04190ae6efa5eabd99659c5b4066ae0145eb1b049bb0f2e7177e130d7` |
| Unchanged harness | `6aa6dfe18b02214236f7cc1bcab5bacc09b301a343a9370f50a01e2f1ecb9cbf` |
| Nine-workload contract | `1af9719b21d980ce68a34fd93e47eb537547caa89fa4df93a7e7ed658ac42674` |
| Prospective candidate contract, raw and canonical LF | `2c4f2ed8fbeaca18d686e896ac9f6a24f612dc341331f12974ffd4f55c0468a2` |
| Baseline genuine native core | `fd52afb2ba6bad3a91dcf6e0adc5ee10f0754f664321d0746c9e13bb86b0d06b` |
| Candidate genuine native core | `26194ffb2ce1c009a7743c5a4b2bf3a9f244d0a4fe3e39fd0a5d8480cce38d00` |
| Candidate preparation review | `703ffb54e37723256646183b8611543ebb6b8c503d8e431cd40df3c4b78b66f9` |
| Independent comparison review | `a49aa03b2be0996d74818a1614147ea33fc40d2c305624b86884d958b43905af` |
| Root independent recomputation | `9494c1604b266f23d1f9b93d7c1d8bfcad0cff63fd955483fc6e8c134b2ea561` |
