# Modern Linux baseline for the eight additions

Run [40262149183](https://github.com/bipxx/scikit-imagexx/actions/runs/40262149183): CPython 3.13.16, NumPy 2.5.4, SciPy 1.18.1, scikit-image 0.26.0, AMD EPYC 7763. The library and all loaded upstream pools reported one thread. Producer `1d37745` supplied the Release wheel; tooling `a7c2bcca` has identical product source. This is the unoptimized baseline, separate from affine v1.

Preparation passed 785 public checks in each normal/scalar mode, 1,318 smoke checks, 13 benchmark metadata checks, and all 73 corpus checks in both modes. The fresh measurement host passed the same 73 exact parity checks before seven paired rounds. Offline wheel/package/runtime identity matched before and after measurement. Timings include complete public orchestration, allocations, explicit NumPy providers and output destruction.

| API | Cases | Faster than original | Geometric mean original/current |
|---|---:|---:|---:|
| `morphology.ellipse` | 12 | 12 | 3.342146× |
| `morphology.octagon` | 13 | 13 | 3.936323× |
| `morphology.octahedron` | 12 | 10 | 1.040573× |
| `morphology.star` | 6 | 6 | 4.349690× |
| `morphology.footprint_from_sequence` | 9 | 9 | 4.202973× |
| `measure.intersection_coeff` | 7 | 0 | 0.042916× |
| `measure.manders_coloc_coeff` | 7 | 0 | 0.040120× |
| `measure.manders_overlap_coeff` | 7 | 0 | 0.030093× |

50 of 52 footprint workloads are faster; both exceptions are large dense octahedra. All 21 colocalization workloads remain slower than scikit-image. Their fixed geometric-mean metric is **0.03728064125267×**. The worst added latency is Manders overlap on the memory workload: 2.449417 ms original versus 102.229722 ms current, **+99.780306 ms**.

## Every slower case

| Workload | Original ms | Current ms | Added ms | Original/current |
|---|---:|---:|---:|---:|
| `measure.manders_overlap_coeff/future/memory` | 2.449416 | 102.229722 | +99.780305 | 0.023960× |
| `measure.manders_coloc_coeff/future/memory` | 1.417063 | 50.098728 | +48.681665 | 0.028285× |
| `measure.intersection_coeff/future/memory` | 0.609440 | 38.449384 | +37.839944 | 0.015850× |
| `measure.manders_overlap_coeff/future/volume` | 0.502681 | 32.832247 | +32.329566 | 0.015311× |
| `measure.manders_coloc_coeff/future/volume` | 0.322180 | 17.076637 | +16.754457 | 0.018867× |
| `measure.intersection_coeff/future/volume` | 0.169814 | 11.453947 | +11.284133 | 0.014826× |
| `measure.manders_overlap_coeff/future/cache-F` | 0.101679 | 6.289418 | +6.187739 | 0.016167× |
| `measure.manders_overlap_coeff/future/cache` | 0.105208 | 6.232538 | +6.127330 | 0.016880× |
| `measure.manders_overlap_coeff/future/stepped` | 0.187764 | 6.212255 | +6.024491 | 0.030225× |
| `measure.manders_overlap_coeff/future/masked` | 0.410812 | 5.148765 | +4.737953 | 0.079788× |
| `measure.intersection_coeff/future/masked` | 0.409264 | 4.104284 | +3.695020 | 0.099716× |
| `measure.intersection_coeff/future/cache-F` | 0.048273 | 3.215341 | +3.167068 | 0.015013× |
| `measure.manders_coloc_coeff/future/cache-F` | 0.061178 | 3.122320 | +3.061142 | 0.019594× |
| `measure.manders_coloc_coeff/future/cache` | 0.066192 | 3.118135 | +3.051943 | 0.021228× |
| `measure.manders_coloc_coeff/future/stepped` | 0.136967 | 3.092169 | +2.955202 | 0.044295× |
| `measure.manders_coloc_coeff/future/masked` | 0.397957 | 3.238612 | +2.840656 | 0.122879× |
| `measure.intersection_coeff/future/cache` | 0.050506 | 2.465989 | +2.415482 | 0.020481× |
| `measure.intersection_coeff/future/stepped` | 0.293353 | 2.479891 | +2.186538 | 0.118293× |
| `morphology.octahedron/large/uint8/None` | 7.366300 | 8.753088 | +1.386788 | 0.841566× |
| `measure.manders_overlap_coeff/future/tiny` | 0.013944 | 0.150624 | +0.136680 | 0.092575× |
| `measure.manders_coloc_coeff/future/tiny` | 0.011048 | 0.079774 | +0.068726 | 0.138487× |
| `morphology.octahedron/large/float64/None` | 8.670709 | 8.713785 | +0.043076 | 0.995057× |
| `measure.intersection_coeff/future/tiny` | 0.018356 | 0.058352 | +0.039996 | 0.314578× |

[Raw checks, product identities and all seven samples](phase3-modern-initial.json) preserve every workload. Full preparation/performance ZIPs and retention hashes remain in `build/phase3-modern-results-v1/` (artifacts `12624946446` and `12625450760`).

This result covers this Linux profile and fixed corpus. The producer wheel-build step passed, while its broader CI job failed; the focused preparation does not convert that broader run into whole-product acceptance. The [minimum baseline](phase3-minimum-initial.md) ran on another host and NumPy version, so its latencies are not a paired cross-profile comparison. No isolated-kernel attribution or full-library speedup is claimed.
