# Polygon classification: initial minimum-profile Release

All 12 workloads were faster than scikit-image 0.26.0. The equal-case geometric mean of original/current median ratios is **2.037704761×**: **2.311963789×** for six point cases and **1.795979987×** for six grid cases. There are no measured losses.

[Exact raw report](polygon-minimum-initial.json) · [Release 40439586172](https://github.com/bipxx/scikit-imagexx/actions/runs/40439586172) · [measured source 24299acf](https://github.com/bipxx/scikit-imagexx/commit/24299acf5bc6ad0aeabe9cf8d0606b046a571162)

The fresh artifact-only measurement used one global worker, seven paired samples per case, and 168 raw samples. Inputs are actual float64 point/vertex arrays: 4,096 or 65,536 points, or square grids of side 256 or 1,024, with 4, 32 or 128 vertices. Preparation is outside timing; grid calls use default `binarize=True`.

| Case | Original ms | Current ms | Original/current | Current − original ms | Max error |
|---|---:|---:|---:|---:|---:|
| points:4096:v4 | 0.115315687 | 0.038323843 | 3.008980249× | -0.076991844 | 0.0 |
| points:4096:v32 | 0.482176808 | 0.224433778 | 2.148414612× | -0.257743030 | 0.0 |
| points:4096:v128 | 1.388261750 | 0.866805467 | 1.601583981× | -0.521456283 | 0.0 |
| points:65536:v4 | 2.374214167 | 0.562752875 | 4.218928542× | -1.811461292 | 0.0 |
| points:65536:v32 | 7.667731500 | 3.535663750 | 2.168682330× | -4.132067750 | 0.0 |
| points:65536:v128 | 22.185059000 | 13.761307000 | 1.612133135× | -8.423752000 | 0.0 |
| grid:256:v4 | 1.779313125 | 0.657943611 | 2.704355046× | -1.121369514 | 0.0 |
| grid:256:v32 | 5.579565500 | 3.626781750 | 1.538434316× | -1.952783750 | 0.0 |
| grid:256:v128 | 20.342157000 | 13.873329000 | 1.466277993× | -6.468828000 | 0.0 |
| grid:1024:v4 | 27.986796000 | 10.543472000 | 2.654419341× | -17.443324000 | 0.0 |
| grid:1024:v32 | 84.815787000 | 57.088237000 | 1.485696379× | -27.727550000 | 0.0 |
| grid:1024:v128 | 304.886904000 | 218.568914000 | 1.394923452× | -86.317990000 | 0.0 |

Every untimed original/current comparison passed exact output metadata and raw-byte checks; zero error is the exact helper result. Point outputs retain noncanonical boolean labels and a nonowning memoryview base. Grid outputs are owning boolean arrays. Output hashes and input hashes are in the raw report; full output arrays are not retained there.

Timed calls include NumPy coordinate-column conversion, allocation, owned C++ ordered dual-ray classification, and the source memoryview/binarize adaptation. Coordinate and protocol handling remain explicit providers. This first original/current comparison does not measure a gain over a prior C++ version.

The producer passed 3,886 focused public checks across 17 normal/scalar pairs, 3,696 full color checks, all 1,612 existing matching checks, and 1,401 smoke checks with registry 235. Separate controls passed: four polygon intake guards, four Euler intake guards, and nine integer-width simulations. The private two-profile component passed 344 public, 176 shared and eight intake checks (528 total), with frozen expectations unchanged. Native controls passed 3,344 each in normal/scalar and both sanitizer modes, and 1,684 each in portable normal/scalar modes.

The review authenticated 59 archive members, 415 committed source files, 14 wheels including 13 runtime dependencies, the installed package/core/SDK, and identical before/after measurement identities. Only the new polygon timing ran; the 15 earlier timing steps were skipped.

This is one minimum Linux x86_64 Release host, with NumPy 1.24.4 and SciPy 1.15.3. The CPU field records only `x86_64`, so the processor model is unknown. The last complete profile qualification remains **203 image APIs**. These results do not establish other platforms or a cumulative cross-host speedup.

Raw JSON SHA-256: `8cf26dfff8e9c648091dc3700af4ff32238e6aa7e99232b5b6ea67c2a2842689`. The single retained archive is `12705888083.zip` (84,612,837 bytes), SHA-256 `f38dcbb8d237129985b6fdb29bcdccd9083c2ad4b6c2aa9d1cc3e02bf1bc701e`; it remains in the ignored local evidence directory. Independent review: `build/phase3-polygon-first-release-v1/final-review.json`, SHA-256 `0644046e5d156ff30b8b97b0202e864ecad5a75818f95642c4d6c2c24878376d`.
