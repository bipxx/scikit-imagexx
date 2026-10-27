# Crofton packed threshold masks: controlled keep

The one-file candidate was **kept** after a same-host baseline/candidate/candidate/baseline comparison. Across the fixed six default `directions=4` cases, the geometric gain was **1.2050264092x direct** and **1.2115521645x reference-normalized**. All four float targets beat both baseline passes in both candidate passes under both metrics; their direct gains were 1.2828x to 1.3594x. The kept source is [`cf1a472`](https://github.com/bipxx/scikit-imagexx/commit/cf1a4724573498be22dc27d7bdbaf5bc70115521), merged as [`eb1eb05`](https://github.com/bipxx/scikit-imagexx/commit/eb1eb05e8e44fa7156406726a6df40e0891c98c3).

The [byte-exact raw archive](perimeter-crofton-packed-same-host.zip) contains all four reports and **336 samples**, the CPU record, offline-install logs, product selections, and all eight before/after identity checks. Each report retains seven paired rounds per case. Timings include the public call's preparation and allocation, use one worker with fallback disabled, and limit original thread pools to one thread.

| Square size | Dtype | Baseline ms | Candidate ms | Direct gain | Normalized gain | Added microseconds | Original/candidate | Max absolute error |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 256 | uint8 | 0.096896003 | 0.096590719 | 1.003160590x | 1.009491906x | -0.305284 | 7.232559118x | 2.273736754e-13 |
| 256 | float32 | 0.134798041 | 0.105081529 | 1.282794818x | 1.294354998x | -29.716512 | 6.687065864x | 2.273736754e-13 |
| 256 | float64 | 0.154058807 | 0.115836084 | 1.329972505x | 1.332682136x | -38.222723 | 6.072253790x | 2.273736754e-13 |
| 1024 | uint8 | 1.255055681 | 1.262338753 | 0.994230493x | 0.997524423x | +7.283072 | 7.417635961x | 0 |
| 1024 | float32 | 1.820745257 | 1.375549428 | 1.323649460x | 1.337502426x | -445.195829 | 6.823097010x | 0 |
| 1024 | float64 | 2.081568667 | 1.531234621 | 1.359405436x | 1.361293492x | -550.334046 | 6.171478195x | 0 |

Each role's time is the geometric mean of its two per-report medians; repetitions were not pooled. Direct gain is baseline/candidate time. Normalized gain is the ratio of the candidate's original/current speedup to the baseline's original/current speedup. The original/candidate column is the candidate role's geometric speedup over scikit-image 0.26.0. The archive preserves every individual median, raw sample, and error.

The sole loss was the **1024 uint8 control: +7.283072 microseconds**, 0.994230493x direct and 0.997524423x normalized. It did not lose against both baseline passes in both candidate passes. No cause is assigned to this change. The two uint8 controls together measured 0.998685560x direct and 1.003490325x normalized. All six cases beat the original in both candidate passes: the same-host original-comparison geometric score was **5.5429825345x for baseline and 6.7156124874x for candidate**. These scores are not multiplied by the earlier comparison on another host.

The measured change replaces scalar extraction of 32/64-bit threshold-mask bytes with ordered SIMD packs and a shuffle. The byte/16-bit paths, predicates, tails, histogram, weighted arithmetic, worker policy, Python providers, parity bounds and benchmark body are unchanged. NumPy/Python coefficient construction, scalar boxing and input/storage preparation remain explicit providers; special input types retain their existing count providers.

The [controlled run 40431796952](https://github.com/bipxx/scikit-imagexx/actions/runs/40431796952) completed at **2026-10-27 01:19:49 UTC**, before the frozen **01:30:38.357512 UTC** deadline. It compared baseline source `06953fe3` / Release [40427614006](https://github.com/bipxx/scikit-imagexx/actions/runs/40427614006) with candidate `cf1a472` / prepare-only Release [40430707979](https://github.com/bipxx/scikit-imagexx/actions/runs/40430707979). Both offline products had identical 13 dependency wheels; the authenticated source map differed only in the permitted SIMD leaf. All eight product/source/runtime/package/core/SDK checks matched their prepared identities.

Runtime: **AMD EPYC 7763**, Linux x86_64, CPython 3.11.17, NumPy 1.24.4, SciPy 1.15.3 and scikit-image 0.26.0. This is one host and one runtime profile. It does not establish universal speedup or change **full supported-profile qualification of 203 image APIs**.

Correctness evidence remained separate from timing. Candidate preparation passed 3,310 focused checks across 14 normal/scalar pairs, 3,696 color checks, 1,612 matching checks and 1,385 smoke checks for 231 registered functions; all 13 standalone timing steps were skipped. Private component validation passed 404 frozen public checks plus 88 shared checks, with 400 saved rows exactly matching baseline. The unchanged native cohort passed 4,176 controls in each normal/scalar/sanitizer mode. Additional private exhaustive mask, tail, canary and special-word controls passed 119,756 assertions in each of eight baseline/candidate normal/scalar/sanitizer runs, with identical outputs and FP state.

Provenance:

- Compact artifact `12702632139`: 25,554 bytes, 36 members, SHA-256 `d6d29ecce627ff773f9152a3eb9efd441f6535e0865cc14d6df6adb5ff8a6660`; all members passed size/CRC/SHA verification.
- Frozen contract SHA-256: `123eb4389f06a48e2bcdf25233ce798e20f4a03621c6be1c456e6c5ab2bde8ef`.
- Independent review: `build/phase3-crofton-packed-threshold-same-host-v1/final-review.json`, SHA-256 `7c9629c0fc2d04491a243a065aabe644955a72f6b9d6f5fa9d8df918d3e6d17b`. Its pre-decision status is preserved; the subsequent root decision and merge are recorded separately.
