# Dense byte masks v4: controlled comparison and keep

**Kept and merged:** source `0135305a02f8ca9f9efd3b2f6d0853ad95ba42c2` is now main. Against kept float64-v3, this controlled comparison gives **2.277537× on the fixed normalized metric and 2.287450× direct public-call speedup**. All seven intersection workloads improve; five beat scikit-image in both candidate passes. **Sixteen colocalization workloads still trail the original.**

[Run 40270844150](https://github.com/bipxx/scikit-imagexx/actions/runs/40270844150) reused the exact products in one AMD EPYC 9V74 job: Linux, CPython 3.11.17, NumPy 1.24.4, SciPy 1.15.3 and scikit-image 0.26.0. All library and upstream pools used one thread. The order was baseline, candidate, candidate, baseline (ABBA), retaining all 73 workloads and seven paired original/current rounds in every pass.

The 14-line C++ addition validates dense uint8/int8 masks with a byte OR reduction, then copies already validated 0/1 bytes in the existing worker partition. Bool, other dtypes and incompatible layouts keep their prior routes. Bounds, overlap and full invalid-before-write checks remain; the binder still rechecks after allocation callbacks. Native controls pass 2,727 in each normal/scalar mode, preserving all prior 1,577 controls.

## Four reports and fixed metric

Each score is the equal-weight geometric mean of original/current median ratios over all 21 colocalization workloads. Compare the geometric mean of both candidate scores with that of both baseline scores. No case or repeat is removed.

| Pass | Product | Fixed 21-case score | GM original median ms | GM current median ms |
|---|---|---:|---:|---:|
| 1-baseline | f64-v3 | 0.09152190269340× | 0.123496333 | 1.349363698 |
| 2-candidate | byte-v4 | 0.20978238569253× | 0.123332930 | 0.587908891 |
| 3-candidate | byte-v4 | 0.20854532512611× | 0.123030653 | 0.589946829 |
| 4-baseline | f64-v3 | 0.09215389681155× | 0.123939753 | 1.344921454 |

Aggregate baseline **0.09183735610744×**, candidate **0.20916294086185×**: normalized gain **2.277536612×**, direct current-call gain **2.287449537×**. Original-reference latency in candidate passes is **0.995666385×** baseline latency (-0.433%). Baseline pass 4 versus 1 drifts +0.359%; candidate pass 3 versus 2 drifts -0.245%.

## All 21 comparisons, including every loss

Normalized gain compares original/current ratios; direct gain compares baseline/candidate current-call medians. Values below 1 are losses. There are **12 gains and 9 losses under each measure**; the two loss sets differ. The last column is the candidate original/current geometric mean: values below 1 remain regressions against scikit-image. IC = `measure.intersection_coeff`; MCC = `measure.manders_coloc_coeff`; MOC = `measure.manders_overlap_coeff`. Suffixes are the unchanged `future/<workload>` IDs.

| Case | Normalized gain | Direct time gain | Candidate original/current |
|---|---:|---:|---:|
| IC/tiny | 2.137448× | 2.143605× | 1.628787× |
| IC/cache | 36.909466× | 37.016668× | 1.885458× |
| IC/memory | 61.928058× | 63.784598× | 2.202478× |
| IC/volume | 53.942463× | 54.391859× | 2.194431× |
| IC/stepped | 1.082162× | 1.084394× | 0.311902× |
| IC/masked | 3.061785× | 3.132671× | 0.850111× |
| MCC/tiny | 0.983711× **loss** | 0.990149× **loss** | 0.342472× |
| MCC/cache | 0.993046× **loss** | 0.995096× **loss** | 0.044958× |
| MCC/memory | 1.008760× | 0.999770× **loss** | 0.069543× |
| MCC/volume | 0.983642× **loss** | 0.998433× **loss** | 0.068055× |
| MCC/stepped | 0.997351× **loss** | 1.003283× | 0.114435× |
| MCC/masked | 1.014334× | 1.019663× | 0.295191× |
| MOC/tiny | 0.996440× **loss** | 0.997979× **loss** | 0.248847× |
| MOC/cache | 0.987777× **loss** | 1.005639× | 0.030015× |
| MOC/memory | 1.036302× | 0.996443× **loss** | 0.051864× |
| MOC/volume | 0.983682× **loss** | 1.002636× | 0.046409× |
| MOC/stepped | 0.994359× **loss** | 0.998853× **loss** | 0.067534× |
| MOC/masked | 1.013510× | 1.013912× | 0.171115× |
| IC/cache-F | 37.169407× | 36.972412× | 1.917634× |
| MCC/cache-F | 0.985779× **loss** | 0.998104× **loss** | 0.045606× |
| MOC/cache-F | 1.012456× | 0.997950× **loss** | 0.030523× |

## All seven intersection workloads

| Workload | Direct gain | Original/current pass 2 | Original/current pass 3 | Beats original in both? |
|---|---:|---:|---:|---|
| tiny | 2.143605× | 1.643977× | 1.613738× | Yes |
| cache | 37.016668× | 1.897222× | 1.873767× | Yes |
| memory | 63.784598× | 2.201885× | 2.203072× | Yes |
| volume | 54.391859× | 2.181143× | 2.207800× | Yes |
| stepped | 1.084394× | 0.312285× | 0.311520× | No |
| masked | 3.132671× | 0.871761× | 0.828998× | No |
| cache-F | 36.972412× | 1.924176× | 1.911115× | Yes |

**Tiny, cache, memory, volume and cache-F now beat the original in both passes.** Stepped and masked improve over v3 but remain slower than scikit-image, as do all 14 MCC/MOC workloads. All 52 footprint workloads remain faster than the original in every pass.

The other 14 colocalization workloads have a direct geometric-mean ratio of **1.001252×**; the unchanged 52 footprint controls have **1.002149×**. No broad gain is inferred for those controls. The raw reports retain every original-library regression and its absolute added latency; the worst remaining penalty is MOC/memory, about **30.37 ms added** in each candidate pass.

## Checks and exact raw evidence

Candidate preparation run `40270607172` reused producer `40270045032` and passed 785 public checks per normal/scalar mode, 1,318 smoke checks, 13 metadata checks and 73 corpus checks per mode. `prepare_only=true` correctly skipped the redundant standalone timing job. The controlled host ran no builds or pytest; its four fresh 73-case checks are bitwise identical, with matching warnings, errors, ownership and native-stage witnesses. All before/after offline identities, source/wheel/core hashes and upstream dependencies match their selected products.

The [byte-exact compact result ZIP](phase3-byte-same-host.zip) contains **all four 73-case reports, 4,088 raw samples, logs, exit statuses, identity checks and archive selections**. Artifact `12627962939`: **134,381 bytes**, 30 CRC-checked members, SHA-256:

`cbf94c7b9698b5c60ecc96e8b1672451c542a3707d29464619fe79b5079c6d26`

Baseline: preparation run `40267273827`, source `ac87e239c5c98b96f9703b431a091d79e25bbfa7`, artifact `12627440319`. Candidate: preparation run `40270607172`, source `0135305a02f8ca9f9efd3b2f6d0853ad95ba42c2`, artifact `12627936362`. Both exact archive SHA-256 selections are inside the compact ZIP. The new candidate preparation ZIP is retained once in `build/phase3-byte-results-v4/`; the old baseline was neither redownloaded nor rehashed.

Independent review: `build/phase3-byte-same-host-results-v4/review.json`, SHA-256 `a15dc948a8fc0953834dfb6b8f1c585322273aac79dd170853e59a3fe400e627`.

This is one minimum-Linux 9V74 host with two repeats per product, not a universal-profile result or a statistical significance claim. Do not multiply this ratio by earlier gains measured on other CPUs to claim cumulative speedup. Timings include allocation, owned orchestration, declared NumPy providers and output destruction; they do not isolate C++ kernels or grant combined-product acceptance. The known masked-overlap return difference from older 7763 captures is preserved; original/candidate agree exactly within this job and across its four reports.

Next: guarded float32 loads for ten workloads are in source/control work. No performance result is available.
