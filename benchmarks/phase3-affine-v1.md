# Colocalization affine indexing v1

Kept source `5bc2109` was measured in [run 40263992815](https://github.com/bipxx/scikit-imagexx/actions/runs/40263992815), using the exact Release wheel from producer run `40263072493`. Profile: Linux, CPython 3.11.17, NumPy 1.24.4, SciPy 1.15.3, scikit-image 0.26.0, AMD EPYC 9V45; the library and loaded upstream pools reported one thread.

All 73 workload specifications and actual fixture/output semantics match the [initial minimum baseline](phase3-minimum-initial.md). Preparation passed 785 public checks in each normal/scalar mode, 1,318 smoke checks, 13 benchmark metadata checks, and 73 corpus checks in each mode. A fresh offline host completed exact parity before seven paired rounds, with matching wheel/package/runtime identity before and after measurement. No cases, rounds, providers or parity assertions were removed.

The fixed 21-case geometric mean of original/current ratios rose from **0.03428367360269×** to **0.05747763987456×**, a **1.676531× normalized improvement**. Every case improved against the baseline ratio (1.302328–2.557942×). **All 21 calls are still slower than scikit-image.** All 52 footprint controls remain faster than the original in this run.

Baseline and candidate measurements used separate hosted machines of the same reported CPU model. Improvement means `(original/current)_candidate / (original/current)_baseline`; it is not a direct same-host paired baseline/candidate timing. Public-call timings include orchestration, allocation, explicit providers and output destruction.

| API | Cases | Baseline original/current | v1 original/current | Normalized improvement |
|---|---:|---:|---:|---:|
| `measure.intersection_coeff` | 7 | 0.040761× | 0.068119× | 1.671181× |
| `measure.manders_coloc_coeff` | 7 | 0.036401× | 0.064185× | 1.763285× |
| `measure.manders_overlap_coeff` | 7 | 0.027159× | 0.043431× | 1.599148× |

## Every remaining regression against scikit-image

| Workload | Original ms | v1 ms | Added ms | Original/v1 | Improvement vs baseline ratio |
|---|---:|---:|---:|---:|---:|
| `measure.manders_overlap_coeff/future/memory` | 1.486398 | 43.187462 | +41.701064 | 0.034417× | 1.612399× |
| `measure.manders_coloc_coeff/future/memory` | 0.756049 | 20.201417 | +19.445368 | 0.037426× | 1.496451× |
| `measure.intersection_coeff/future/memory` | 0.333828 | 15.523369 | +15.189541 | 0.021505× | 1.388035× |
| `measure.manders_overlap_coeff/future/volume` | 0.312666 | 10.849457 | +10.536791 | 0.028819× | 2.493444× |
| `measure.manders_coloc_coeff/future/volume` | 0.197623 | 4.804774 | +4.607151 | 0.041131× | 2.557942× |
| `measure.intersection_coeff/future/volume` | 0.093701 | 3.946943 | +3.853242 | 0.023740× | 2.078539× |
| `measure.manders_overlap_coeff/future/cache-F` | 0.053160 | 2.779775 | +2.726616 | 0.019124× | 1.480337× |
| `measure.manders_overlap_coeff/future/cache` | 0.054939 | 2.771111 | +2.716171 | 0.019826× | 1.466792× |
| `measure.manders_overlap_coeff/future/stepped` | 0.114344 | 2.808741 | +2.694397 | 0.040710× | 1.475940× |
| `measure.manders_overlap_coeff/future/masked` | 0.216941 | 2.124885 | +1.907945 | 0.102095× | 1.515354× |
| `measure.manders_coloc_coeff/future/cache-F` | 0.036807 | 1.257322 | +1.220515 | 0.029274× | 1.700571× |
| `measure.manders_coloc_coeff/future/cache` | 0.036160 | 1.239602 | +1.203442 | 0.029171× | 1.655825× |
| `measure.manders_coloc_coeff/future/stepped` | 0.081728 | 1.213143 | +1.131415 | 0.067369× | 1.669290× |
| `measure.intersection_coeff/future/masked` | 0.196698 | 1.185060 | +0.988362 | 0.165981× | 2.132656× |
| `measure.intersection_coeff/future/cache` | 0.028314 | 0.943133 | +0.914819 | 0.030021× | 1.447617× |
| `measure.intersection_coeff/future/cache-F` | 0.029002 | 0.930697 | +0.901695 | 0.031161× | 2.156445× |
| `measure.manders_coloc_coeff/future/masked` | 0.203911 | 1.065755 | +0.861844 | 0.191330× | 2.038337× |
| `measure.intersection_coeff/future/stepped` | 0.144177 | 0.909601 | +0.765425 | 0.158505× | 1.455369× |
| `measure.manders_overlap_coeff/future/tiny` | 0.011985 | 0.064270 | +0.052285 | 0.186476× | 1.369737× |
| `measure.manders_coloc_coeff/future/tiny` | 0.008033 | 0.030327 | +0.022295 | 0.264863× | 1.445062× |
| `measure.intersection_coeff/future/tiny` | 0.012139 | 0.022412 | +0.010273 | 0.541623× | 1.302328× |

[Raw checks, identities and paired samples](phase3-affine-v1.json) preserve all 73 cases. Preparation/performance ZIPs and retention hashes remain in `build/phase3-affine-results-v1/` (artifacts `12625871547` and `12625758201`). Independent retained-result review found no scoped blocker.

Next: optimize the remaining measured colocalization overhead using the unchanged 21-case metric and all 73 correctness/performance controls. This keep decision covers one minimum Linux profile; it does not establish modern/Windows performance, whole-product qualification, or that the three APIs beat the original library.
