# Public resize/rescale: controlled one-worker comparison

The [completed run](https://github.com/bipxx/scikit-imagexx/actions/runs/40020239743) compared the full initial C++ product (`66e5f78`) with the full current product (`36440f8`) and scikit-image 0.26.0 in one process on each runtime's measurement host. All 373 cases passed three-backend parity before 372 cases were timed with seven samples per backend. The expected endian-float16 error is retained and untimed. Both C++ backends defaulted to one worker and had independent worker and fallback state; all measurements used one worker.

Ratios above 1 favor the current product. Geometric means weight each workload equally.

| Runtime | Current / initial speedup | Current / original speedup | Slower than initial | Slower than original |
|---|---:|---:|---:|---:|
| Windows modern | 1.0119x | 1.2613x | 178/372 | 57/372 |
| Linux modern | 1.0124x | 1.5200x | 186/372 | 34/372 |
| Linux minimum | 1.0145x | 1.5567x | 179/372 | 23/372 |

The aggregate changes versus the initial product are small and mixed. This same-host comparison avoids differences between historical runners, but compares two complete products and does not isolate an individual patch. Seven samples do not establish equivalence for near-1 ratios or guarantee a speedup for every workload.

The layout cohorts show larger measured changes. Each row contains the same 12 dtype workloads on each runtime; the table reports current-versus-initial geometric means.

| Cohort | Windows modern | Linux modern | Linux minimum |
|---|---:|---:|---:|
| Broadcast | 1.1271x | 1.2015x | 1.2050x |
| Reversed | 1.0779x | 1.1135x | 1.1123x |
| Stepped | 1.0690x | 1.0694x | 1.0642x |

Other cohorts remain mixed: anisotropic rescaling measures 0.9869x/0.9884x versus initial on modern/minimum Linux; identity resizing measures 0.9859x on Windows. Windows nearest-down and up-default remain slower than original in aggregate (0.877x and 0.961x). The worst individual current-versus-original ratios are 0.428x, 0.562x and 0.566x on Windows, modern Linux and minimum Linux respectively.

These are warmed end-to-end public calls, including dispatch, layout conversion, temporary allocations and result destruction. The three backends shared each runtime's recorded CPU session, fixtures and sample ordering. Measurement jobs performed no builds or tests and started after all three preparation jobs passed.

The [recomputed summary](resize-public-controlled.json) retains every slower case versus both initial and original, with complete workload specifications, medians and samples. It links the three byte-exact raw reports and their hashes; all 372 timing rows remain unchanged. Offline verification checked pinned initial artifacts, complete current source/package/runtime wheel trees, complete SDKs, capsule hashes, sample arithmetic and the preparation barrier. No local image operations or timing calls were used for this publication.
