# Component consumers: initial one-worker performance

The [completed isolated campaign](https://github.com/bipxx/scikit-imagexx/actions/runs/40024754860) measured source `77a0e34` against scikit-image 0.26.0. These are warmed public-call ratios on the recorded hosts, with seven paired samples and one native worker. Ratios above 1 favor the C++ implementation.

All 878 parity cases passed on each runtime. Of 216 proposed raster workloads, 11 raise defined original errors and remain untimed, leaving 205 measured workloads (181 raster and 24 raster-mutation calls). The other 662 proposed contract cases are untimed. Returned ArrayMap state and metadata checks carry no dense-map or class performance claim.

| Runtime | Geometric mean speedup | Faster cases | Slower cases | Worst speedup |
|---|---:|---:|---:|---:|
| windows-modern | 0.8824x | 71/205 | 134 | 0.1830x |
| linux-modern | 1.1008x | 104/205 | 101 | 0.2140x |
| linux-minimum | 1.2302x | 112/205 | 93 | 0.2196x |

The geometric mean weights each measured workload equally. Individual slower cases remain; this initial campaign does not establish that every call is faster or measure a previous/current patch effect.

| Public API | Windows modern | Linux modern | Linux minimum |
|---|---:|---:|---:|
| `morphology.remove_small_holes` | 1.0200x | 1.3994x | 1.5870x |
| `morphology.remove_small_objects` | 0.8212x | 0.9066x | 0.9285x |
| `segmentation.clear_border` | 1.1704x | 1.4264x | 1.5815x |
| `segmentation.relabel_sequential` | 0.5151x | 0.6935x | 0.8757x |

Measured bottleneck examples (all slower cases and complete specifications are retained in the JSON):

- windows-modern: `segmentation.relabel_sequential`, `raster/3/0/2`, 0.1830x. Fixture: `{"dtype":"u1","layout":"broadcast","mask":null,"options":{"offset":127},"pattern":"checkerboard","shape":[17,129,131]}`.
- linux-modern: `segmentation.clear_border`, `raster/1/1/2`, 0.2140x. Fixture: `{"dtype":"i2","layout":"broadcast","mask":null,"options":{"bgval":0,"buffer_size":0},"pattern":"checkerboard","shape":[512,513]}`.
- linux-minimum: `segmentation.clear_border`, `raster/1/1/2`, 0.2196x. Fixture: `{"dtype":"i2","layout":"broadcast","mask":null,"options":{"bgval":0,"buffer_size":0},"pattern":"checkerboard","shape":[512,513]}`.

Calls include public allocations, conversions and result destruction. Mutating/output workloads reset working buffers inside the timed callback and retain their owners through destruction. Nonmutating fixture construction is excluded. The source preflight asserted default worker1, sole-core/shared-worker identity and strict native public registration. Full hashed dependency wheels, source/package/library trees and all 59 SDK files were verified offline.

Measurement jobs performed no builds or tests and started after all three preparation jobs passed. The [summary and complete slower-case list](component-consumers-public-initial.json) links all three byte-exact raw reports, authoritative artifact IDs and hashes, samples and source workload specifications. No local image operations, foreign-binary imports or timing calls were used for this publication.
