# Colocalization width caching v2

Kept product source `4a78e94` was measured in [run 40266173692](https://github.com/bipxx/scikit-imagexx/actions/runs/40266173692), using the exact Release wheel from producer run `40265597625`. It is integrated in main `78976b7` with unchanged product source. Profile: Linux, CPython 3.11.17, NumPy 1.24.4, SciPy 1.15.3, scikit-image 0.26.0; one library worker and one thread in every loaded upstream pool.

The source change caches validated byte widths, sign masks and product masks outside pixel loops. Pairwise reduction order, buffer boundaries, FP behavior, providers and input/output guards remain unchanged. Retained native controls pass 1,577 checks in both normal and scalar modes, preserving all 1,549 prior controls.

Preparation passed 785 public checks in each normal/scalar mode, 1,318 smoke checks, 13 benchmark metadata checks and all 73 corpus checks in each mode. The fresh offline measurement host passed all 73 exact parity checks before seven paired rounds. All workload specifications and actual input/output semantics match v1 and the initial baseline; wheel/package/runtime identities match before and after measurement.

## Fixed 21-case metric

The metric is the geometric mean of original/current per-case median ratios; higher is better.

| Version | Original/current metric | Improvement over prior kept version | Improvement over baseline |
|---|---:|---:|---:|
| [Initial minimum](phase3-minimum-initial.md) | 0.03428367360269× | — | 1.000000× |
| [Affine v1](phase3-affine-v1.md) | 0.05747763987456× | 1.676531× | 1.676531× |
| Width v2 | 0.09195998447613× | 1.599926× | 2.682326× |

All 21 normalized ratios improved over v1, by 1.318724–1.883815×. **All 21 calls remain slower than scikit-image.** All 52 footprint controls remain faster than the original in this run. The worst remaining penalty is the memory-sized Manders overlap case: original 1.787758 ms versus v2 27.573586 ms, **+25.785828 ms**.

| API | Cases | v1 original/current | v2 original/current | Normalized improvement vs v1 |
|---|---:|---:|---:|---:|
| `measure.intersection_coeff` | 7 | 0.068119× | 0.112549× | 1.652244× |
| `measure.manders_coloc_coeff` | 7 | 0.064185× | 0.101312× | 1.578435× |
| `measure.manders_overlap_coeff` | 7 | 0.043431× | 0.068202× | 1.570359× |

Baseline, v1 and v2 ran on separate hosted machines. Normalized improvement is `(original/current)_v2 / (original/current)_v1`, not a same-host paired candidate/baseline latency. Timings include the complete public call, allocation, orchestration, explicit providers and output destruction; they do not isolate a kernel stage.

## Every remaining regression against scikit-image

| Workload | Original ms | v2 ms | Added ms | Original/v2 | Improvement vs v1 ratio |
|---|---:|---:|---:|---:|---:|
| `measure.manders_overlap_coeff/future/memory` | 1.787758 | 27.573586 | +25.785828 | 0.064836× | 1.883815× |
| `measure.manders_coloc_coeff/future/memory` | 0.801904 | 11.818046 | +11.016142 | 0.067854× | 1.813046× |
| `measure.intersection_coeff/future/memory` | 0.330965 | 8.607354 | +8.276389 | 0.038451× | 1.788033× |
| `measure.manders_overlap_coeff/future/volume` | 0.292849 | 6.601224 | +6.308375 | 0.044363× | 1.539385× |
| `measure.manders_coloc_coeff/future/volume` | 0.193707 | 2.914580 | +2.720873 | 0.066461× | 1.615867× |
| `measure.intersection_coeff/future/volume` | 0.093848 | 2.298636 | +2.204788 | 0.040828× | 1.719772× |
| `measure.manders_overlap_coeff/future/cache` | 0.053641 | 1.717002 | +1.663360 | 0.031241× | 1.575793× |
| `measure.manders_overlap_coeff/future/stepped` | 0.111139 | 1.726119 | +1.614980 | 0.064386× | 1.581582× |
| `measure.manders_overlap_coeff/future/cache-F` | 0.052359 | 1.665020 | +1.612661 | 0.031446× | 1.644367× |
| `measure.manders_overlap_coeff/future/masked` | 0.209804 | 1.370100 | +1.160296 | 0.153130× | 1.499875× |
| `measure.manders_coloc_coeff/future/cache` | 0.036834 | 0.755339 | +0.718505 | 0.048764× | 1.671685× |
| `measure.manders_coloc_coeff/future/stepped` | 0.083921 | 0.801877 | +0.717956 | 0.104656× | 1.553474× |
| `measure.manders_coloc_coeff/future/cache-F` | 0.035772 | 0.725924 | +0.690152 | 0.049278× | 1.683300× |
| `measure.intersection_coeff/future/masked` | 0.199791 | 0.791352 | +0.591561 | 0.252467× | 1.521059× |
| `measure.manders_coloc_coeff/future/masked` | 0.202907 | 0.733694 | +0.530787 | 0.276556× | 1.445435× |
| `measure.intersection_coeff/future/cache-F` | 0.028520 | 0.540259 | +0.511738 | 0.052790× | 1.694099× |
| `measure.intersection_coeff/future/cache` | 0.028227 | 0.538965 | +0.510738 | 0.052373× | 1.744537× |
| `measure.intersection_coeff/future/stepped` | 0.146167 | 0.540008 | +0.393841 | 0.270675× | 1.707672× |
| `measure.manders_overlap_coeff/future/tiny` | 0.011450 | 0.046475 | +0.035025 | 0.246369× | 1.321182× |
| `measure.manders_coloc_coeff/future/tiny` | 0.007888 | 0.022584 | +0.014696 | 0.349282× | 1.318724× |
| `measure.intersection_coeff/future/tiny` | 0.011937 | 0.015478 | +0.003540 | 0.771259× | 1.423978× |

[Exact raw checks, identities and seven paired samples](phase3-width-v2.json) retain all 73 cases. Full performance/preparation ZIPs, producer records and SHA/CRC retention evidence remain in `build/phase3-width-results-v2/` (artifacts `12626088928` and `12625989299`). Independent source/control and retained-result reviews found no scoped blocker.

This keep decision covers one minimum Linux profile, not modern/Windows performance or whole-product acceptance. The next f64-v3 candidate changes the known-float64 load path; it has no performance result yet. Preserve the metric, full corpus, exact gates and every regression.
