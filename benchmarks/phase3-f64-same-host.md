# Float64 direct reads v3: controlled comparison and keep

**Kept and merged:** measured source `ac87e239c5c98b96f9703b431a091d79e25bbfa7` is integrated in main `bae051a428759ec02b90bc50b564201c23b80691` with unchanged product source. The controlled comparison gives **+2.94% on the fixed normalized metric and +2.03% in direct public-call speedup**. It resolves the earlier [cross-CPU hold](phase3-f64-v3.md); the earlier apparent 12% gain is not a code-attributed result.

[Run 40268472820](https://github.com/bipxx/scikit-imagexx/actions/runs/40268472820) reused the exact width-v2 and f64-v3 products in one job on AMD EPYC 7763, Linux, CPython 3.11.17, NumPy 1.24.4, SciPy 1.15.3 and scikit-image 0.26.0. Every library and loaded upstream pool used one thread. Order was baseline, candidate, candidate, baseline (ABBA); each pass retained all 73 workloads and seven paired original/current rounds.

## Four reports, unchanged metric

Each score is the equal-weight geometric mean of original/current median ratios over all 21 colocalization workloads. The comparison divides the geometric mean of both candidate scores by that of both baseline scores. No case or repeat is dropped.

| Pass | Product | Fixed 21-case score | GM original median ms | GM current median ms |
|---|---|---:|---:|---:|
| 1-baseline | width-v2 | 0.09681514809864× | 0.180282203 | 1.862128055 |
| 2-candidate | f64-v3 | 0.09935584409481× | 0.180604975 | 1.817758949 |
| 3-candidate | f64-v3 | 0.10092452705057× | 0.184189727 | 1.825024428 |
| 4-baseline | width-v2 | 0.09773786506987× | 0.181262862 | 1.854581761 |

Aggregate baseline **0.09727541252333×**, candidate **0.10013711387383×**: normalized gain **1.029418548×**. The geometric mean of baseline/candidate current-call medians is **1.020293870×**. Original-reference latency is **1.008943186×** in candidate passes versus baseline passes (+0.894%); baseline pass 4 versus 1 drifts +0.544%, and candidate pass 3 versus 2 drifts +1.985%. This explains why the normalized and direct gains differ.

## All 21 case comparisons

Ratios above 1 favor v3; ratios below 1 are losses. **12 normalized gains / 9 losses; 7 direct gains / 14 losses.** IC = `measure.intersection_coeff`; MCC = `measure.manders_coloc_coeff`; MOC = `measure.manders_overlap_coeff`. Each suffix is the unchanged `future/<workload>` case ID.

| Case | Normalized gain | Direct time gain |
|---|---:|---:|
| IC/tiny | 1.002928× | 0.978470× **loss** |
| IC/cache | 0.988879× **loss** | 0.999719× **loss** |
| IC/memory | 1.031492× | 1.002637× |
| IC/volume | 0.990937× **loss** | 1.002953× |
| IC/stepped | 0.991516× **loss** | 0.996007× **loss** |
| IC/masked | 0.984246× **loss** | 0.977761× **loss** |
| MCC/tiny | 0.997565× **loss** | 1.009020× |
| MCC/cache | 1.006440× | 0.991525× **loss** |
| MCC/memory | 1.159761× | 1.131552× |
| MCC/volume | 1.115724× | 1.126324× |
| MCC/stepped | 1.006057× | 0.994215× **loss** |
| MCC/masked | 1.002619× | 0.983466× **loss** |
| MOC/tiny | 1.031389× | 0.992498× **loss** |
| MOC/cache | 1.004820× | 0.980184× **loss** |
| MOC/memory | 1.177532× | 1.196272× |
| MOC/volume | 1.189402× | 1.194292× |
| MOC/stepped | 0.966956× **loss** | 0.978128× **loss** |
| MOC/masked | 0.972109× **loss** | 0.968712× **loss** |
| IC/cache-F | 1.046337× | 0.992272× **loss** |
| MCC/cache-F | 0.994315× **loss** | 0.993236× **loss** |
| MOC/cache-F | 0.998991× **loss** | 0.983678× **loss** |

## Targeted float64 cases

The four intended float64 sum/product workloads improve in both candidate passes. These are public-call medians in milliseconds, not isolated kernel timings.

| Case | Baseline 1 | Candidate 2 | Candidate 3 | Baseline 4 | Direct gain |
|---|---:|---:|---:|---:|---:|
| MCC/memory | 21.790951 | 19.164569 | 19.298740 | 21.732065 | 1.131552× |
| MCC/volume | 5.499953 | 4.855986 | 4.917373 | 5.507798 | 1.126324× |
| MOC/memory | 49.918809 | 41.720047 | 41.796662 | 49.989857 | 1.196272× |
| MOC/volume | 12.671249 | 10.641368 | 10.637932 | 12.742553 | 1.194292× |

The other 17 colocalization cases have a direct geometric-mean ratio of **0.989618×**; the 52 unchanged footprint controls have **1.000196×**. The targeted benefit does not imply that every workload improved. **All 21 colocalization workloads remain slower than scikit-image in every pass**; all 52 footprint workloads are faster in every pass.

## Checks, raw evidence and limits

All four fresh 73-case checks are bitwise identical, including values, types, ownership, warnings, errors and native-stage witnesses. Every before/after offline identity check passes against its exact role source, wheel and core. The job retains **4,088 raw samples (1,022 per pass)**, exact recomputed medians/ratios and every regression. No tests or builds ran on the measurement host; the selected preparations retain their existing 785 normal + 785 scalar public checks, 1,318 smoke checks and 73 corpus checks per mode.

The [byte-exact compact result ZIP](phase3-f64-same-host.zip) contains all four full 73-case JSON reports, logs, exit statuses, offline identity checks and source/archive selections. It is artifact `12627781845`, **133,234 bytes**, 30 CRC-checked members, SHA-256:

`3f457840b3e20a083bd6492565e1c8b6d06b794d85504025747ee61f1a2e4d86`

Baseline input: run `40266173692`, source `4a78e945a5eccfb14c0887f5187378c5f4d76cf9`, prepared artifact `12625989299`. Candidate input: run `40267273827`, source `ac87e239c5c98b96f9703b431a091d79e25bbfa7`, prepared artifact `12627440319`. The archive includes both selected archive SHA-256 values and exact wheel/core identities. Independent review: `build/phase3-same-host-results-v3/review.json`, SHA-256 `3ff5d9afa50e7d3c66028040dfba87ecea6e8c88df79f168e6ba2586eb1df5e2`.

This is one minimum-Linux host with two repeats per product, without a statistical significance claim. Timings include allocation, owned orchestration, declared NumPy providers and output destruction. They do not establish full native API coverage, cross-profile speed or combined-product acceptance. The earlier cross-host one-ULP masked-overlap difference remains recorded; this same-host comparison has exact agreement across all four reports.

The subsequent [byte-mask v4 comparison](phase3-byte-same-host.md) is now measured and kept. Its separate same-host result preserves this earlier v3 evidence unchanged.
