# Rank neighborhood algorithms: initial one-worker performance

The [completed isolated campaign](https://github.com/bipxx/scikit-imagexx/actions/runs/40024802716) measured source `77a0e34` against scikit-image 0.26.0. These are warmed public-call ratios on the recorded hosts, with seven paired samples and one native worker. Ratios above 1 favor the C++ implementation.

All 810 canonical parity cases passed on each runtime; 480 successful raster workloads were measured across 5 algorithms. The other 330 canonical cases and 5 generic identity-alias checks are untimed. Aliases, errors and metadata do not increase the timed algorithm count.

| Runtime | Geometric mean speedup | Faster cases | Slower cases | Worst speedup |
|---|---:|---:|---:|---:|
| windows-modern | 2.0809x | 338/480 | 142 | 0.3108x |
| linux-modern | 2.2196x | 325/480 | 155 | 0.2235x |
| linux-minimum | 2.4120x | 368/480 | 112 | 0.3891x |

The geometric mean weights each measured workload equally. Individual slower cases remain; this initial campaign does not establish that every call is faster or measure a previous/current patch effect.

| Public API | Windows modern | Linux modern | Linux minimum |
|---|---:|---:|---:|
| `filters.rank.maximum` | 1.4841x | 1.5514x | 1.8226x |
| `filters.rank.mean` | 3.6239x | 3.9620x | 3.6456x |
| `filters.rank.minimum` | 1.7256x | 1.8391x | 2.1087x |
| `filters.rank.pop` | 1.1688x | 1.1790x | 1.6223x |
| `filters.rank.sum` | 3.5970x | 4.0419x | 3.5912x |

Measured bottleneck examples (all slower cases and complete specifications are retained in the JSON):

- windows-modern: `filters.rank.pop`, `balanced/s1/d0/p1`, 0.3108x. Fixture: `{"bins":256,"dtype":"u1","footprint":"disconnected","id":"balanced/s1/d0/p1","layout":"step","mask":"none","out":"f4","out_layout":"C","pattern":"ramp","shape":[512,513]}`.
- linux-modern: `filters.rank.maximum`, `footprint/3d/d0/f2`, 0.2235x. Fixture: `{"bins":256,"dtype":"u1","footprint":"sparse","id":"footprint/3d/d0/f2","layout":"C","mask":"none","out":"f4","out_layout":"C","pattern":"random","shape":[7,11,13]}`.
- linux-minimum: `filters.rank.maximum`, `balanced/s1/d1/p3`, 0.3891x. Fixture: `{"bins":256,"dtype":"u2","footprint":"sparse","id":"balanced/s1/d1/p3","layout":"step","mask":"sparse","out":"f4","out_layout":"C","pattern":"random","shape":[512,513]}`.

Input, footprint, mask and provided-output fixture construction is excluded; provided outputs are reused and overwrite/result release is included. Source-backed preflight asserted native public registration, non-original wrappers and generic alias identity before fixture creation. Every measured call enforced worker1. Initial default1 was not independently recorded by this campaign; it remains supported by prior product/source evidence.

Exact product wheels, complete Python/native packages, all 59 SDK files, source fingerprints and runtime pins/freeze text were verified offline. These rank artifacts did not archive dependency-wheel/binary closures, so no independent offline verification of those dependency bytes is claimed.

Measurement jobs performed no builds or tests and started after all three preparation jobs passed. The [summary and complete slower-case list](rank-public-initial.json) links all three byte-exact raw reports, authoritative artifact IDs and hashes, samples and source workload specifications. No local image operations, foreign-binary imports or timing calls were used for this publication.
