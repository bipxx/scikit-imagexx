# Components: public one-worker baseline

The [completed isolated run](https://github.com/bipxx/scikit-imagexx/actions/runs/39983471905) measured source `b4633193db85cd09de961fb934ef368216e34bda` against scikit-image 0.26.0. Each runtime passed 954 strict checks and timed 802 defined successful calls, with seven paired samples. All raw samples and slower workloads are retained.

| Runtime | Work | Cases | Geometric mean ratio | Slower cases |
|---|---|---:|---:|---:|
| windows-modern | cast_warning | 6 | 1.060x | 0 |
| windows-modern | empty | 18 | 2.072x | 0 |
| windows-modern | raster | 200 | 0.821x | 108 |
| windows-modern | tiny | 578 | 1.213x | 277 |
| linux-modern | cast_warning | 6 | 1.123x | 0 |
| linux-modern | empty | 18 | 1.991x | 0 |
| linux-modern | raster | 200 | 1.395x | 84 |
| linux-modern | tiny | 578 | 1.152x | 262 |
| linux-minimum | cast_warning | 6 | 1.200x | 0 |
| linux-minimum | empty | 18 | 2.406x | 0 |
| linux-minimum | raster | 200 | 1.210x | 82 |
| linux-minimum | tiny | 578 | 1.334x | 247 |

Tiny, empty and warning calls are separated from raster processing. The raster-only result retains major multivalued/high-connectivity regressions; the overall ratio across tiny and large calls does not establish that image labeling is faster. Both public label paths name the same implementation.

The [summary and all regressions](components-public-initial.json) links byte-preserved raw reports. Exact wheels, complete package/SDK seals where present, dependency logs and source fingerprints remain protected in the downloaded workflow artifacts. No local operation timings were used.
