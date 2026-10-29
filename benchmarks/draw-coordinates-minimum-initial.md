# Drawing coordinates: initial one-thread results

**10 wins, 2 losses; geometric mean 1.874024071× over scikit-image.** This is an initial descriptive comparison, with every loss retained. It is not a KEEP/DISCARD optimization decision or a whole-library performance claim.

[Run 40820624189](https://github.com/bipxx/scikit-imagexx/actions/runs/40820624189) used the genuine minimum-profile source `566c1fcd`, core `9d464675`, on AMD EPYC 9V74 with affinity CPU 1 and one worker/reference pool. All 168 samples across 12 workloads are retained in [the JSON evidence](draw-coordinates-minimum-initial.json), including the complete result ZIP, report, calibration journal and independent reviews.

| Workload | Original ms | Current ms | Original/current | Added current µs |
|---|---:|---:|---:|---:|
| line/tiny | 0.001777404 | 0.003484300 | 0.510118026× | +1.706896 |
| line/long | 0.197545885 | 0.098592780 | 2.003654685× | -98.953105 |
| rectangle/tiny | 0.020432315 | 0.008698549 | 2.348933658× | -11.733766 |
| rectangle/2d | 0.054314790 | 0.055090680 | 0.985916116× | +0.775891 |
| rectangle/3d | 0.053705950 | 0.039037415 | 1.375755801× | -14.668535 |
| disk/tiny | 0.033899766 | 0.023187236 | 1.462001183× | -10.712530 |
| disk/dense | 2.609138273 | 0.801391841 | 3.255758467× | -1807.746431 |
| disk/clipped | 0.481196782 | 0.160842755 | 2.991721832× | -320.354027 |
| ellipse/tiny | 0.033524454 | 0.022922517 | 1.462511897× | -10.601937 |
| ellipse/axes | 0.883443366 | 0.277653222 | 3.181822848× | -605.790144 |
| ellipse/rotated | 0.966688697 | 0.316961794 | 3.049858740× | -649.726904 |
| ellipse/clipped | 0.291103806 | 0.102109879 | 2.850887767× | -188.993926 |

The losses are **line/tiny: 0.510118026×, +1.706896 µs**, and **rectangle/2d: 0.985916116×, +0.775891 µs**. The aggregate does not remove these regressions; this single cohort does not establish their stage-level causes.

| API | Cases | Geometric mean |
|---|---:|---:|
| draw.disk | 3 | 2.423857243× |
| draw.ellipse | 4 | 2.522079733× |
| draw.line | 2 | 1.010989799× |
| draw.rectangle | 3 | 1.471467552× |

Every accepted batch lasted at least 100 ms (observed 106.721–122.505 ms). All 24 function warmup groups accumulated at least 250 ms. The full public call, allocation, conversions, declared providers and output destruction are included. The harness verified actual `skimagexx._core` returns, no whole-algorithm forwarding on admitted paths, exact coordinate words/container/layout/base ownership and aliases, independent oracles, input preservation, and unchanged worker/NumPy error-policy state. Product, runtime, core and SDK identities matched before and after. No hardware floating-environment equivalence is claimed.

The earlier [run 40819506015](https://github.com/bipxx/scikit-imagexx/actions/runs/40819506015) remains a **bootstrap failure with zero image calls, zero samples and no performance data**. `runpy.run_path` did not add the script directory, so `from provenance import source_fingerprint` failed before NumPy or target calls. V2 added one workflow import-path line; production source, driver, contracts, sampler and all 12 geometries stayed unchanged. Both attempts retained the original clock: **2026-10-29 04:45:15.223425 UTC to 06:45:15.223425 UTC**. The failed ZIP, traceback and forensic receipts are embedded separately; V1 is not relabelled as successful.

Native ownership covers ordinary Bresenham traversal, rectangle xy grids, and disk/ellipse mask, prefix and emission. NumPy rounding, clipping, bounding-box construction, scalar modulo/trig, and result view/container construction remain explicit providers. The dtype labels describe coordinate parameters and `intp` outputs; they do not imply pixel-image dtype support or full-domain native coverage.

Measured source `566c1fcd` is distinct from its public integration `bb9825b6`. Genuine preparation covered 251 partial image APIs plus 3 metadata APIs, 178 product members and 84 SDK members. One installed drawing SDK client family passed two mode checks, each using workers 1/4; this is separate from the declared 27-family campaign. **Full supported-profile qualification remains the separate 247-API baseline.**
