# First one-thread benchmark of the eight additions

Run [40259547328](https://github.com/bipxx/scikit-imagexx/actions/runs/40259547328), minimum Linux: CPython3.11.17, NumPy1.24.4, SciPy1.15.3, scikit-image0.26.0, AMD EPYC9V45. Both native and loaded upstream pools reported one thread.

All785 public contracts passed in normal and scalar dispatch, along with the1,318-check registry smoke. All73 benchmark workloads passed exact public parity before measurement. Seven paired warmed rounds include public orchestration, allocations, conversion/providers and output destruction. The compiled wheel comes from1d37745; toolingd634cd2 has identical product source. This is scoped development performance, not full supported-profile or installed-client acceptance.

| API | Cases | Faster | Geometric mean speedup |
|---|---:|---:|---:|
| morphology.ellipse | 12 | 12 | 2.473x |
| morphology.octagon | 13 | 13 | 3.411x |
| morphology.octahedron | 12 | 12 | 1.132x |
| morphology.star | 6 | 6 | 4.063x |
| morphology.footprint_from_sequence | 9 | 9 | 4.578x |
| measure.intersection_coeff | 7 | 0 | 0.041x |
| measure.manders_coloc_coeff | 7 | 0 | 0.036x |
| measure.manders_overlap_coeff | 7 | 0 | 0.027x |

All52 footprint workloads were faster. All21 colocalization workloads were slower. The largest penalty is the1024x1024 float64 Manders overlap case: original1.407ms versus current65.912ms (+64.505ms). Colocalization optimization is required.

The fixed optimization metric is geometric-mean speedup across all21 colocalization workloads (higher is better):0.0342836736027x at baseline. Keep all73 workloads and exact parity gates unchanged; retain every slower case. No improvement is attributed to a particular stage without additional evidence.

[Raw checks, identities, paired samples, loops and every slower case](phase3-minimum-initial.json) are retained. Full raw preparation/result archives are preserved locally under build/phase3-minimum-results-v1, with sizes and SHA256 in retention.json.
