# UniqueRows: tie-free sort on the same host

**KEEP under the frozen contract.** Across nine public workloads, geometric-mean gain over the kept prefix-cache C++ baseline is **1.114885217x direct / 1.111665044x normalized**. Original/current changes from 1.579057771x baseline to 1.755383326x candidate. All nine workloads beat scikit-image in both candidate passes. The read-only duplicate target improves 1.508190531x direct and 1.473357156x normalized.

**C++ regressions remain:** all-equal 64-byte rows have balanced direct gain 0.997972986x and normalized gain 0.998745898x. Three per-pass regression observations are listed below. This is a scoped nine-workload result, with no full-library performance claim.

[All four raw reports and comparison](unique-rows-tie-free-same-host.json) retain **504 samples**, original numeric text, medians, loop counts, input/output/native-index hashes and metadata. The [initial five-loss result](unique-rows-minimum-initial.md) and [prefix-cache comparison](unique-rows-prefix-same-host.md) remain published.

| Workload | Baseline C++ ms (GM) | Candidate C++ ms (GM) | Original / baseline | Original / candidate | Direct gain | Normalized gain | Added ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| u8-short-small-unique | 0.010155387 | 0.010144567 | 1.729745342x | 1.738221123x | 1.001066538x | 1.004900017x | -0.000010820 |
| u8-short-large-unique | 6.346398775 | 6.230805962 | 1.295043956x | 1.307330938x | 1.018551824x | 1.009487695x | -0.115592813 |
| f32-coordinates-large-unique | 6.705330634 | 6.512445063 | 1.491369057x | 1.524976242x | 1.029617996x | 1.022534453x | -0.192885571 |
| f64-coordinates-duplicates | 13.422617411 | 11.695441111 | 0.958444498x | 1.104098413x | 1.147679449x | 1.151969066x | -1.727176300 |
| u8-wide32-common-prefix | 6.417818673 | 6.251383774 | 2.133326335x | 2.172405921x | 1.026623689x | 1.018318616x | -0.166434898 |
| f64-wide64-all-equal | 0.099894393 | 0.100097291 | 7.655732624x | 7.646131552x | 0.997972986x | 0.998745898x | 0.000202899 |
| f32-strided-duplicates | 1.869714385 | 1.502361511 | 1.142428850x | 1.448192037x | 1.244516963x | 1.267643090x | -0.367352874 |
| f64-fortran-duplicates | 1.416840758 | 1.233614688 | 1.146933702x | 1.310139343x | 1.148527796x | 1.142297363x | -0.183226071 |
| u8-readonly-duplicates | 6.295862963 | 4.174448012 | 0.890741877x | 1.312380918x | 1.508190531x | 1.473357156x | -2.121414951 |

Ratios below 1 mean a regression against the stated denominator. Added time is candidate minus baseline. Each balanced value uses the geometric mean of two passes; all individual medians remain in the raw JSON.

| Workload | Original ms during baseline (GM) | Original ms during candidate (GM) | Original / candidate pass 2 | Original / candidate pass 3 |
|---|---:|---:|---:|---:|
| u8-short-small-unique | 0.017566233 | 0.017633501 | 1.765413728x | 1.711447365x |
| u8-short-large-unique | 8.218865379 | 8.145725403 | 1.315927930x | 1.298790111x |
| f32-coordinates-large-unique | 10.000122622 | 9.931323998 | 1.536984474x | 1.513061829x |
| f64-coordinates-duplicates | 12.864833805 | 12.912917967 | 1.115288136x | 1.093020957x |
| u8-wide32-common-prefix | 13.691301585 | 13.580543122 | 2.241344291x | 2.105587929x |
| f64-wide64-all-equal | 0.764764762 | 0.765357058 | 7.634701124x | 7.657579093x |
| f32-strided-duplicates | 2.136015654 | 2.175707977 | 1.418833904x | 1.478157641x |
| f64-fortran-duplicates | 1.625022415 | 1.616207137 | 1.321261586x | 1.299110727x |
| u8-readonly-duplicates | 5.607988791 | 5.478465914 | 1.317385158x | 1.307395687x |

All five losses from the first Release are retained below. Those initial ratios came from a separate host and are context, not denominators for this experiment. The read-only loss remaining after the prefix-cache change is now an original win in both passes (1.317385158x / 1.307395687x).

| Initial loss case | First Release original/current | This baseline original/current | This candidate original/current | Both candidate passes beat original? |
|---|---:|---:|---:|---:|
| u8-short-large-unique | 0.718282155x | 1.295043956x | 1.307330938x | yes |
| f32-coordinates-large-unique | 0.919185133x | 1.491369057x | 1.524976242x | yes |
| f64-wide64-all-equal | 0.477032070x | 7.655732624x | 7.646131552x | yes |
| f32-strided-duplicates | 0.824715942x | 1.142428850x | 1.448192037x | yes |
| u8-readonly-duplicates | 0.559877740x | 0.890741877x | 1.312380918x | yes |

Every per-pass C++ regression against the geometric mean of the two baselines:

| Workload / candidate pass | Direct gain | Normalized gain |
|---|---:|---:|
| u8-short-small-unique, pass 3 | 0.993166432x | 0.989421578x |
| u8-wide32-common-prefix, pass 3 | 1.020935930x | 0.986997580x |
| f64-wide64-all-equal, pass 2 | 0.992352622x | 0.997252843x |

| Group | Cases | Direct gain | Normalized gain | Original / candidate |
|---|---:|---:|---:|---:|
| targets | 1 | 1.508190531x | 1.473357156x | 1.312380918x |
| preserved_original_wins | 8 | 1.073561431x | 1.073203768x | 1.820375478x |
| other_eight_cpp_controls | 8 | 1.073561431x | 1.073203768x | 1.820375478x |
| duplicates | 4 | 1.254165317x | 1.252077627x | 1.287664831x |
| equal | 1 | 0.997972986x | 0.998745898x | 7.646131552x |
| prefix | 1 | 1.026623689x | 1.018318616x | 2.172405921x |
| unique | 3 | 1.016344013x | 1.012279894x | 1.513275079x |
| float32 | 2 | 1.131979267x | 1.138511631x | 1.486088305x |
| float64 | 3 | 1.095704975x | 1.095363672x | 2.228036733x |
| uint8 | 4 | 1.120929740x | 1.110718232x | 1.595412027x |

## Fixed gates and measured scope

One candidate was frozen at 2026-10-28T14:57:38.559206Z with a two-hour deadline of 16:57:38.559206Z. Metrics completed at **15:31:31.846743Z**, 5,166.712463 seconds before deadline. Dispatch-to-job-start delay was **4 seconds** (15:29:54Z to 15:29:58Z), consuming the existing budget. No extension, variant, retiming or retry occurred.

KEEP requires overall direct and normalized geometric means strictly above 1.02x. Read-only uint8 duplicates must exceed 1.20x direct and normalized in each candidate pass against the two-baseline geometric mean, and beat original in both passes. The other eight workloads must retain original wins in both passes and balanced direct and normalized gains at least 0.97x. All correctness, source, identity and deadline gates passed. Target direct passes were 1.513673659x / 1.502727264x; normalized passes 1.478975215x / 1.467760437x.

The candidate removes original-index tie ordering from byte-key sorting and merges. A linear scan compacts each equal-key group while retaining its minimum original index. Cached unsigned prefixes, the all-equal shortcut, SIMD comparison, chunking, workers, native ABI, binder, Python and final literal NumPy advanced-index gather remain fixed. Results apply to this complete change; no measured per-stage causal claim is made.

The unchanged harness times whole public calls in baseline/candidate/candidate/baseline order: allocation, contiguous preparation, row view, index allocation, sorting/equality/compaction, final literal NumPy gather and output destruction. Seed 2026, nine cases, seven paired rounds targeting 19 ms, global workers one and all four reference pools one. The final gather remains a provider; it is not claimed as owned C++ gathering.

Before each report timer, pinned original and an independent unsigned-byte row oracle verify full output bytes and dtype/layout/ownership metadata. Actual native first indices match independent first-occurrence indices byte for byte. Genuine `xx._core.unique_rows_apply` C returns are observed while original and NumPy-unique forwarding are blocked, then hooks are restored. Inputs/backings remain unchanged. Raw exports retain hashes rather than duplicate image payloads. Offline review verified all 504 samples and medians, all four report fingerprints, both original product identities and eight pre/post identities.

Six native modes passed first attempt: normal 12,148; scalar 10,594; sanitizer 12,148/10,594; portable 5,298 each. All 86 frozen returns and 809 previous native record payloads per worker remain exact. Private/public component gates total 856; all 416 prior observations and 344 C acceptances remain exact. Both genuine preparations passed 30 normal/scalar XML pairs (6,698), color 3,696, matching 1,612, smoke 1,431 across 244 functions, and 4/4/4/9 intake guards.

**Measured product scope:** exact d8 baseline and 7aa candidate use 159 wheel members and SDK71, with 241 partially native image APIs plus three metadata APIs. The separately arriving installed-header relocation/SDK79 change is outside this experiment. Full four-profile qualification remains **203 image APIs**; this result grants no promotion.

## Provenance

- Baseline `d8adeb419d72bf284f1b4a04ce20e32a9be52074`; candidate `7aa059ddb1c2f196c3dbba9cf4d638ae1a761e67`. Only native leaf, header comment, native tests and prospective candidate contract differ. SIMD source, binder, Python, CMake, original source and harness remain fixed.
- [Single ABBA run 40716546947](https://github.com/bipxx/scikit-imagexx/actions/runs/40716546947), controller `2d6115f06daf47bff173deafe5bd3bb8395f18f7`.
- [Baseline preparation 40702637014](https://github.com/bipxx/scikit-imagexx/actions/runs/40702637014) / [artifact 12823857042](https://github.com/bipxx/scikit-imagexx/actions/runs/40702637014/artifacts/12823857042).
- [Candidate preparation 40712229136](https://github.com/bipxx/scikit-imagexx/actions/runs/40712229136) / [artifact 12828765895](https://github.com/bipxx/scikit-imagexx/actions/runs/40712229136/artifacts/12828765895). Both preparations skipped all 24 timing steps.
- [Comparison artifact 12829280728](https://github.com/bipxx/scikit-imagexx/actions/runs/40716546947/artifacts/12829280728): 40,044 bytes, 39 members, downloaded once; every member size/CRC/SHA verified.
- CPU: **AMD EPYC 7763 64-Core Processor**. CPython 3.11.17, NumPy 1.24.4, SciPy 1.15.3, scikit-image 0.26.0, AVX2 enabled. All 13 dependency wheels and scientific package/library identities match exactly; no normalization. Baseline source has 1,757 committed blobs, candidate 1,758.

| Proof | SHA-256 |
|---|---:|
| Combined raw publication | `8a1d9301f4f0c14f16a6441b42dbd20095b451d0c7bdc90967f2869cc280b529` |
| Comparison ZIP | `a6f8b99026df7a57816f0e4966436961981fd0c6d55ae50bf624e4a1b04ed103` |
| Baseline ZIP | `355736e04190ae6efa5eabd99659c5b4066ae0145eb1b049bb0f2e7177e130d7` |
| Candidate ZIP | `b9f233f4b6ed05efe1b844ab19f1e4d8454d31f53dc56fa285e1860fb1321087` |
| Unchanged harness | `6aa6dfe18b02214236f7cc1bcab5bacc09b301a343a9370f50a01e2f1ecb9cbf` |
| Nine-workload contract | `1af9719b21d980ce68a34fd93e47eb537547caa89fa4df93a7e7ed658ac42674` |
| Prospective candidate contract, raw and LF | `f2bc8c10b2ae017a98c8b6dbc936be6d54b2b9d48acc12f7e2d98b5cebba05ea` |
| Baseline native core | `26194ffb2ce1c009a7743c5a4b2bf3a9f244d0a4fe3e39fd0a5d8480cce38d00` |
| Candidate native core | `a45f73ec949964e0ebb6ed8a28bf78c662ff30fcb788e3e289e78e38ced08d3e` |
| Candidate preparation review | `5f09b60b92175026810b68ebb13a6bde853d6af2a19c717bec928e78bec1901c` |
| Independent comparison review | `301c15ccd6b9274e70f538fde77b6ad36eef0a3fae69100cf7c56d124583da2f` |
| Independent per-pass loss receipt | `b050b9f128685f50e028b1152d8c5df18b5d5df2e6d8c5d5febb42a8344758c5` |
