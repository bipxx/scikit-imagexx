# Footprints: public one-worker baseline

The [completed isolated run](https://github.com/bipxx/scikit-imagexx/actions/runs/39983468427) measured source `b4633193db85cd09de961fb934ef368216e34bda` against scikit-image 0.26.0. Each runtime passed 556 strict checks and timed 527 defined successful calls, with seven paired samples. All raw samples and slower workloads are retained.

| Runtime | Work | Cases | Geometric mean ratio | Slower cases |
|---|---|---:|---:|---:|
| windows-modern | decomposition_plan | 144 | 0.693x | 96 |
| windows-modern | raster | 254 | 2.160x | 34 |
| windows-modern | view_metadata | 129 | 0.835x | 128 |
| linux-modern | decomposition_plan | 144 | 0.719x | 97 |
| linux-modern | raster | 254 | 2.323x | 32 |
| linux-modern | view_metadata | 129 | 0.935x | 124 |
| linux-minimum | decomposition_plan | 144 | 0.796x | 96 |
| linux-minimum | raster | 254 | 2.499x | 32 |
| linux-minimum | view_metadata | 129 | 0.950x | 111 |

Raster generation, decomposition construction and view metadata are reported separately. Decomposition and mirror metadata overhead must not be presented as C++ image-kernel speedup. Rectangle setup and decomposition construction retain substantial slower cases.

The [summary and all regressions](footprints-public-initial.json) links byte-preserved raw reports. Exact wheels, complete package/SDK seals where present, dependency logs and source fingerprints remain protected in the downloaded workflow artifacts. No local operation timings were used.
