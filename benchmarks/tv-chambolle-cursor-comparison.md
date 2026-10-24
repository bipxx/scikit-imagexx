# TV Chambolle cursor comparison

This independently reviewed one-worker campaign compares the cursor increment with the prior C++ product and scikit-image 0.26.0. Aggregate gains coexist with individual regressions; this does not complete the performance goal.

[Actions run 40119412201](https://github.com/bipxx/scikit-imagexx/actions/runs/40119412201) used immutable controller `d48a17ed41376eecac63f5a5f75bd6ec72a0d6f1`. Actual before product/source is `3cdd3637ca78343edb2cb554b77269de5f5eaf20`; after is `5e068a5a2267c3a917701d5348449972fad162a7`. Both have 192 image APIs, 3 metadata APIs, 110 wheel members, 63 SDK members and 18 SDK families. `6ff02f63f6033cb186d3b93d22a778fd8d0624a3` is only the authenticated intended-before source-equivalence anchor, not a relabelled wheel.

| Host profile | Speedup vs original (GM) | Speedup vs prior C++ (GM) | Slower than original | Slower than prior C++ |
|---|---:|---:|---:|---:|
| windows | 1.315585× | 1.206708× | 30/70 | 30/70 |
| modern | 1.152370× | 1.306207× | 34/70 | 5/70 |
| minimum | 1.243727× | 1.193946× | 30/70 | 23/70 |

Ratios greater than 1 favor after. Each ratio uses per-role median times, then the summary takes their geometric mean. The full-precision [CSV](tv-chambolle-cursor-comparison.csv) retains all 210 rows, workload specifications, canonical 400-case indices, 21 paired sample values per row, calibration and both regression flags. The [raw archive](tv-chambolle-cursor-comparison-raw.zip) preserves original report bytes and all 4,410 samples.

## Cohorts

| Profile | Cohort | Cases | Speedup vs original (GM) | Speedup vs prior C++ (GM) | Slower original / prior C++ |
|---|---|---:|---:|---:|---:|
| windows | channels | 12 | 0.940925× | 1.287990× | 6 / 4 |
| windows | layout | 10 | 1.787074× | 1.175831× | 2 / 5 |
| windows | raster | 12 | 0.864466× | 1.256339× | 7 / 5 |
| windows | small | 12 | 1.583037× | 1.157392× | 5 / 4 |
| windows | tiny | 12 | 3.293214× | 0.994499× | 3 / 7 |
| windows | volume | 12 | 0.719991× | 1.403745× | 7 / 5 |
| modern | channels | 12 | 0.815424× | 1.393251× | 6 / 0 |
| modern | layout | 10 | 1.795563× | 1.269205× | 2 / 2 |
| modern | raster | 12 | 0.540284× | 1.452400× | 9 / 0 |
| modern | small | 12 | 1.499725× | 1.264327× | 6 / 0 |
| modern | tiny | 12 | 3.405684× | 1.015760× | 3 / 3 |
| modern | volume | 12 | 0.624065× | 1.498632× | 8 / 0 |
| minimum | channels | 12 | 0.889396× | 1.221291× | 6 / 6 |
| minimum | layout | 10 | 1.980896× | 1.217350× | 2 / 3 |
| minimum | raster | 12 | 0.509698× | 1.254572× | 9 / 4 |
| minimum | small | 12 | 1.702194× | 1.166937× | 5 / 3 |
| minimum | tiny | 12 | 4.035467× | 1.024881× | 0 / 3 |
| minimum | volume | 12 | 0.648435× | 1.302756× | 8 / 4 |

Cohorts remain separate: tiny/small, raster, volume, channels and layout. The cursor changes address traversal; an unchanged contiguous path and tiny-call variation must not be described as a measured cursor-kernel improvement. The cohort ratios are observed outcomes, not isolated attribution to every case. All 129 cases slower than either comparator remain in the raw union list and CSV.

## Largest absolute regressions against before

| Profile | Case index / ID | Before ms | After ms | Added ms |
|---|---|---:|---:|---:|
| windows | 361 / `raster/raster/f4/converge` | 52.0829 | 61.2879 | 9.205 |
| windows | 365 / `raster/raster/f8/converge` | 25.729 | 29.1552 | 3.4262 |
| windows | 357 / `raster/raster/u1/converge` | 12.8704 | 14.0623 | 1.1919 |
| windows | 373 / `raster/volume/f4/converge` | 17.0841 | 18.1221 | 1.038 |
| minimum | 362 / `raster/raster/f8/one` | 2.99569925 | 3.559546 | 0.56384675 |
| minimum | 354 / `raster/raster/u1/one` | 3.12450725 | 3.64353767 | 0.519030417 |
| windows | 356 / `raster/raster/u1/fixed8` | 6.61665 | 7.04415 | 0.4275 |
| minimum | 356 / `raster/raster/u1/fixed8` | 6.10243 | 6.520927 | 0.418497 |

These cases, together with still-slower-than-original raster/channel/volume cases in the CSV, are explicit inputs for the next single-admission investigation. No negative case or tolerance has been dropped.

## Method and provenance

All three preparations checked the frozen 400-case corpus through original/before/after workers 1 and 4 in normal and SIMD-disabled modes. A verified common barrier preceded fresh, artifact-only measurement hosts. Measurements used three isolated role processes, native default/global worker count 1, fallback disabled for native calls, seven rotating paired triads and the unchanged 70 timing IDs. No builds or tests occurred during measurement. CPU models, complete process identities, role sources, calibration and raw elapsed samples are preserved.

| Profile | Actual CPU model |
|---|---|
| windows | AMD64 Family 25 Model 1 Stepping 1, AuthenticAMD |
| modern | AMD EPYC 7763 64-Core Processor |
| minimum | AMD EPYC 7763 64-Core Processor |

The before/after full toolchain, dependency, flag and header common-input model was equal after masking only the two reviewed TV implementation translation units. Full qualified product/source/wheel/RECORD/SDK/runtime seals were verified and their exact hashes are in the [evidence JSON](../coverage/evidence/tv-chambolle-cursor-performance.json) and raw experiment. Large compiled products and dependency wheels stay in protected scientific storage rather than this Git archive. The before [qualification run](https://github.com/bipxx/scikit-imagexx/actions/runs/40098671228) and after [qualification run](https://github.com/bipxx/scikit-imagexx/actions/runs/40111184948) retain their original revision labels.

The ZIP includes exact experiment bytes, preparation reports, barrier records, permits, complete timing reports and artifact SHA/CRC inventories. Three claims and nine startup receipts were deterministically reconstructed from frozen controller field order plus captured permit/report metadata; every reconstructed SHA matched its original pin. They are explicitly labelled **source-derived**, never original raw uploads. The old initial f871 baseline/report is unchanged and remains a separate campaign.

Independent root review verified all 357 retained campaign records (538,098,154 bytes), all 6,917 artifact members and the complete before/after product seals, then compared every CSV value and all 85 archive members with their original bytes. Publication proof SHA256: `66bf1baf26a2a40b7244349a5a1068eb38134f8f3132437def96e0ff585ae6b9`. Historical draft-status fields inside the immutable raw archive are retained unchanged.

This campaign supports only these workloads on these hosts. Cross-host/cross-campaign CPU changes, tiny-call variance and individual regressions preclude universal speed claims. Measurements are accepted with every regression retained; optimization remains incomplete.
