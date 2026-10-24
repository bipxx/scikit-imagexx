# Rank intensity algorithms: initial performance

The [completed isolated campaign](https://github.com/bipxx/scikit-imagexx/actions/runs/40034423731) measured exact source `fbdfb46a339f0b5d89dd1e5ba861f6e68ff6b113` against scikit-image 0.26.0. All fixed parity cases passed before seven paired samples per successful workload, using one native worker and one reference thread pool. Ratios above 1 favor the C++ public call.

| Runtime | Geometric mean public-call speedup | Slower cases | Worst speedup |
|---|---:|---:|---:|
| windows-modern | 2.0892x | 92/384 | 0.3908x |
| linux-modern | 2.2440x | 81/384 | 0.4358x |
| linux-minimum | 2.4022x | 81/384 | 0.4124x |

Each runtime passed all 1,428 canonical cases. Only 384 successful raster calls across four algorithms were timed; 1,044 canonical checks, four generic identity aliases and 24 policy checks remain untimed. Alias paths do not increase the algorithm or timing count. Supplied-output reset/public allocation/result release are included; image/footprint/mask fixture creation is excluded.

Measured slower examples (complete samples and workload specifications are retained in the JSON):

- windows-modern: `filters.rank.equalize`, `balanced/s1/d1/p3`, 0.3908x; workload index 372.
- linux-modern: `filters.rank.autolevel`, `footprint/3d/d0/f2`, 0.4358x; workload index 46.
- linux-minimum: `filters.rank.threshold`, `balanced/s1/d1/p2`, 0.4124x; workload index 1085.

| Public API | Windows modern | Linux modern | Linux minimum |
|---|---:|---:|---:|
| `filters.rank.autolevel` | 2.0118x | 2.0165x | 2.1098x |
| `filters.rank.enhance_contrast` | 1.9951x | 2.0180x | 2.1195x |
| `filters.rank.equalize` | 1.4306x | 2.0032x | 2.1192x |
| `filters.rank.threshold` | 3.3176x | 3.1106x | 3.5138x |

These are observed ratios on the recorded CPUs for the exact source above. Hosted CPU/run differences prevent causal before/after attribution to the named patch. Individual slower calls remain, and this campaign does not establish universal performance superiority.

Offline verification covered the archived source against Git, complete product/dependency wheels and RECORDs, installed package/library trees and all 59 SDK members. Initial worker1 is asserted by source preflight; it is not separately serialized as initial-default telemetry.

All three preparation jobs finished before fresh measurement jobs began; measurement jobs performed no builds or tests. The [summary and complete slower-case list](rank-intensity-public-initial.json) links three byte-exact raw reports, authoritative artifact IDs/hashes, all samples and source workload specifications. No image operations, timers, foreign-binary imports or source extraction were performed locally for this publication.
