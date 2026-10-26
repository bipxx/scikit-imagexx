# Float64 direct reads v3: historical cross-host measurement

**Historical HOLD resolved:** v3 was subsequently kept after the [controlled same-host ABBA comparison](phase3-f64-same-host.md), then merged in main `bae051a` with product source equal to `ac87e23`. That comparison found +2.94% normalized and +2.03% direct aggregate gains. The raw report and all figures below remain the original cross-host observation.

At this earlier checkpoint, source `ac87e23` was unmerged and on HOLD; width v2 remained the kept baseline. The apparent 12% difference below did not establish a causal code gain because the CPU model changed.

Run [40267273827](https://github.com/bipxx/scikit-imagexx/actions/runs/40267273827) used the exact wheel from producer `40266743521`, source `ac87e239c5c98b96f9703b431a091d79e25bbfa7`. Profile: Linux, CPython 3.11.17, NumPy 1.24.4, SciPy 1.15.3, scikit-image 0.26.0, one library worker and one thread in every loaded upstream pool. **v3 ran on AMD EPYC 7763; kept width-v2 ran on AMD EPYC 9V45.**

Preparation passed 785 public checks in each normal/scalar mode, 1,318 smoke checks, 13 benchmark metadata checks and 73 corpus checks in each mode. The fresh measurement host passed all 73 exact original/candidate comparisons before seven paired rounds. Wheel/package/runtime identity matched before and after measurement. Retained native controls passed 1,577 checks per mode using the unchanged test object.

The source replaces two known-float64 dispatches with the same existing memcpy/byte-swap read helper. Arithmetic, reduction association and providers are unchanged. Combined swapped-and-unaligned float64 storage has no new targeted native test; source equivalence of the reused helper is the argument for that combination.

## Observed metric, not a code-attributed gain

| Measurement | Fixed 21-case original/current geometric mean | Cross-run ratio vs kept v2 |
|---|---:|---:|
| [Initial minimum](phase3-minimum-initial.md) | 0.03428367360269× | 0.372811× |
| [Kept width v2](phase3-width-v2.md) | 0.09195998447613× | 1.000000× |
| F64 v3 (cross-host) | 0.10307806470738× | 1.120901× |

The v3 score is 3.006623× the initial baseline ratio. Nineteen normalized case ratios improve over v2 and two worsen. **All 21 colocalization calls remain slower than scikit-image.** Changes in unchanged routines and the different CPU model prevent a causal interpretation. Normalization is `(original/current)_v3 / (original/current)_v2`, not paired baseline/candidate timing on one host.

## Both cross-run losses against v2

| Workload | v2 original/current | v3 original/current | v3/v2 normalized ratio | Relative loss |
|---|---:|---:|---:|---:|
| `measure.intersection_coeff/future/memory` | 0.038451445× | 0.035617984× | 0.926310667× | 7.369% |
| `measure.manders_overlap_coeff/future/stepped` | 0.064386461× | 0.064172615× | 0.996678716× | 0.332% |

One result also changes across hosts: `measure.manders_overlap_coeff/future/masked` has float64 raw bytes `60980e7fe1f2e63f` on v2/9V45 and `5f980e7fe1f2e63f` on v3/7763, a one-ULP difference. **Original and candidate agree exactly within each run.** All 73 input snapshots/specifications, result types, warnings and errors match across runs; cross-host return bits are not claimed identical. No assertion or tolerance was changed.

## Every regression against scikit-image

| Workload | Original ms | v3 ms | Added ms | Original/v3 |
|---|---:|---:|---:|---:|
| `measure.manders_overlap_coeff/future/memory` | 3.099946 | 42.120167 | +39.020221 | 0.073598× |
| `measure.manders_coloc_coeff/future/memory` | 1.586224 | 19.559038 | +17.972814 | 0.081099× |
| `measure.intersection_coeff/future/memory` | 0.587981 | 16.507977 | +15.919996 | 0.035618× |
| `measure.manders_overlap_coeff/future/volume` | 0.585671 | 10.694514 | +10.108843 | 0.054764× |
| `measure.manders_coloc_coeff/future/volume` | 0.339258 | 4.864219 | +4.524962 | 0.069746× |
| `measure.intersection_coeff/future/volume` | 0.190215 | 4.173959 | +3.983744 | 0.045572× |
| `measure.manders_overlap_coeff/future/cache` | 0.124830 | 3.378870 | +3.254040 | 0.036944× |
| `measure.manders_overlap_coeff/future/cache-F` | 0.126301 | 3.342357 | +3.216056 | 0.037788× |
| `measure.manders_overlap_coeff/future/stepped` | 0.216850 | 3.379165 | +3.162315 | 0.064173× |
| `measure.manders_overlap_coeff/future/masked` | 0.468516 | 2.750826 | +2.282310 | 0.170318× |
| `measure.manders_coloc_coeff/future/cache` | 0.079031 | 1.504262 | +1.425231 | 0.052538× |
| `measure.manders_coloc_coeff/future/cache-F` | 0.080103 | 1.459981 | +1.379879 | 0.054865× |
| `measure.manders_coloc_coeff/future/stepped` | 0.156787 | 1.469434 | +1.312647 | 0.106699× |
| `measure.intersection_coeff/future/masked` | 0.451910 | 1.567603 | +1.115694 | 0.288281× |
| `measure.manders_coloc_coeff/future/masked` | 0.448782 | 1.454930 | +1.006147 | 0.308456× |
| `measure.intersection_coeff/future/cache` | 0.070022 | 1.052831 | +0.982808 | 0.066508× |
| `measure.intersection_coeff/future/cache-F` | 0.072295 | 1.050592 | +0.978297 | 0.068814× |
| `measure.intersection_coeff/future/stepped` | 0.337671 | 1.076227 | +0.738556 | 0.313755× |
| `measure.manders_overlap_coeff/future/tiny` | 0.025449 | 0.095541 | +0.070092 | 0.266370× |
| `measure.manders_coloc_coeff/future/tiny` | 0.019321 | 0.050063 | +0.030742 | 0.385941× |
| `measure.intersection_coeff/future/tiny` | 0.026248 | 0.031657 | +0.005409 | 0.829143× |

The worst added latency is Manders overlap/memory: 3.099946 ms original versus 42.120167 ms v3, **+39.020221 ms**. Complete public-call timings include allocation, orchestration, explicit providers and output destruction; they do not isolate kernel time.

[Exact raw checks, identities and all seven paired samples](phase3-f64-v3.json) preserve all 73 workloads. Full performance/preparation archives and SHA/CRC records remain in `build/phase3-f64-results-v3/` (artifacts `12627241021` and `12627440319`). The original independent review passed this scoped measurement while withholding a keep decision; the later controlled comparison resolved that hold.

The completed [same-host ABBA result](phase3-f64-same-host.md) reuses these frozen products and preserves every workload and adverse case. This historical cross-host result still grants no whole-product acceptance, cross-profile speed claim or 12% causal code gain.
