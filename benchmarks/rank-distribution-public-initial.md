# Rank distribution algorithms: initial one-worker performance

The [completed isolated campaign](https://github.com/bipxx/scikit-imagexx/actions/runs/40024807986) measured source `77a0e34` against scikit-image 0.26.0. These are warmed public-call ratios on the recorded hosts, with seven paired samples and one native worker. Ratios above 1 favor the C++ implementation.

All 648 canonical parity cases passed on each runtime; 384 successful raster workloads were measured across 4 algorithms. The other 264 canonical cases and 4 generic identity-alias checks are untimed. Twenty-four warning/callback policy checks are also untimed. Entropy outputs passed full raw-byte comparisons using the recorded runtime C log provider; no cross-libm or log2 equivalence is assumed. Aliases, errors and metadata do not increase the timed algorithm count.

| Runtime | Geometric mean speedup | Faster cases | Slower cases | Worst speedup |
|---|---:|---:|---:|---:|
| windows-modern | 2.5833x | 315/384 | 69 | 0.4385x |
| linux-modern | 3.2954x | 320/384 | 64 | 0.3172x |
| linux-minimum | 3.4111x | 321/384 | 63 | 0.3181x |

The geometric mean weights each measured workload equally. Individual slower cases remain; this initial campaign does not establish that every call is faster or measure a previous/current patch effect.

| Public API | Windows modern | Linux modern | Linux minimum |
|---|---:|---:|---:|
| `filters.rank.entropy` | 4.5756x | 5.0836x | 5.2964x |
| `filters.rank.gradient` | 1.9516x | 1.9144x | 2.0099x |
| `filters.rank.median` | 2.2408x | 3.3162x | 3.3841x |
| `filters.rank.modal` | 2.2256x | 3.6541x | 3.7584x |

Measured bottleneck examples (all slower cases and complete specifications are retained in the JSON):

- windows-modern: `filters.rank.modal`, `footprint/3d/d0/f2`, 0.4385x. Fixture: `{"bins":256,"dtype":"u1","footprint":"sparse","id":"footprint/3d/d0/f2","layout":"C","mask":"none","out":"f4","out_layout":"C","pattern":"random","shape":[7,11,13]}`.
- linux-modern: `filters.rank.gradient`, `footprint/3d/d0/f2`, 0.3172x. Fixture: `{"bins":256,"dtype":"u1","footprint":"sparse","id":"footprint/3d/d0/f2","layout":"C","mask":"none","out":"f4","out_layout":"C","pattern":"random","shape":[7,11,13]}`.
- linux-minimum: `filters.rank.gradient`, `footprint/3d/d0/f2`, 0.3181x. Fixture: `{"bins":256,"dtype":"u1","footprint":"sparse","id":"footprint/3d/d0/f2","layout":"C","mask":"none","out":"f4","out_layout":"C","pattern":"random","shape":[7,11,13]}`.

Input, footprint, mask and provided-output fixture construction is excluded; provided outputs are reused and overwrite/result release is included. Source-backed preflight asserted native public registration, non-original wrappers and generic alias identity before fixture creation. Every measured call enforced worker1. Initial default1 was not independently recorded by this campaign; it remains supported by prior product/source evidence.

Exact product wheels, complete Python/native packages, all 59 SDK files, source fingerprints and runtime pins/freeze text were verified offline. These rank artifacts did not archive dependency-wheel/binary closures, so no independent offline verification of those dependency bytes is claimed.

Measurement jobs performed no builds or tests and started after all three preparation jobs passed. The [summary and complete slower-case list](rank-distribution-public-initial.json) links all three byte-exact raw reports, authoritative artifact IDs and hashes, samples and source workload specifications. No local image operations, foreign-binary imports or timing calls were used for this publication.
