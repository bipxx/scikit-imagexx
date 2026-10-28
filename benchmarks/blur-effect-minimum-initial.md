# Blur effect: first minimum-profile Release comparison

All 24 measured cases are faster than scikit-image 0.26.0: equal-case geometric mean **1.539529955×**, range **1.184230484–1.854105426×**, with no losses and zero observed error. The [byte-exact JSON](blur-effect-minimum-initial.json) retains all **336 raw latency samples**.

Inputs are positive 1D, 2D and 3D C-order images in uint8, float32 and float64. Each fixture starts with seed-2026 random uint8 values in [0, 255]; floating inputs are normalized by 255 and cast before timing. Calls use `channel_axis=None`, the default `np.max` reducer, and the listed `h_size`.

| Shape | Input dtype | h_size | Original ms | Current ms | Original/current | Current − original ms | Max error |
|---|---|---:|---:|---:|---:|---:|---:|
| 4,096 | uint8 | 11 | 0.116840017 | 0.089491636 | 1.305597059× | -0.027348381 | 0.0 |
| 4,096 | float32 | 11 | 0.099597458 | 0.084103103 | 1.184230484× | -0.015494355 | 0.0 |
| 4,096 | float64 | 11 | 0.108859800 | 0.087461060 | 1.244665912× | -0.021398740 | 0.0 |
| 65,536 | uint8 | 11 | 1.008308875 | 0.656530429 | 1.535814383× | -0.351778446 | 0.0 |
| 65,536 | float32 | 11 | 0.867025105 | 0.604503190 | 1.434277137× | -0.262521915 | 0.0 |
| 65,536 | float64 | 11 | 0.971616250 | 0.636808043 | 1.525760015× | -0.334808207 | 0.0 |
| 256 × 256 | uint8 | 11 | 2.969086000 | 1.813421125 | 1.637284335× | -1.155664875 | 0.0 |
| 256 × 256 | float32 | 11 | 2.457185143 | 1.422841636 | 1.726956170× | -1.034343506 | 0.0 |
| 256 × 256 | float64 | 11 | 2.931053333 | 1.599309400 | 1.832699372× | -1.331743933 | 0.0 |
| 1,024 × 1,024 | uint8 | 11 | 57.549962000 | 41.926216000 | 1.372648607× | -15.623746000 | 0.0 |
| 1,024 × 1,024 | float32 | 11 | 45.698997000 | 25.360642000 | 1.801965305× | -20.338355000 | 0.0 |
| 1,024 × 1,024 | float64 | 11 | 53.736648000 | 29.819823000 | 1.802044499× | -23.916825000 | 0.0 |
| 32 × 32 × 32 | uint8 | 11 | 3.741709500 | 2.912935000 | 1.284515274× | -0.828774500 | 0.0 |
| 32 × 32 × 32 | float32 | 11 | 3.365493000 | 2.398498429 | 1.403166648× | -0.966994571 | 0.0 |
| 32 × 32 × 32 | float64 | 11 | 3.726261250 | 2.524774571 | 1.475878794× | -1.201486679 | 0.0 |
| 64 × 64 × 64 | uint8 | 11 | 28.523341000 | 18.822686000 | 1.515370389× | -9.700655000 | 0.0 |
| 64 × 64 × 64 | float32 | 11 | 23.864179000 | 16.343610000 | 1.460153479× | -7.520569000 | 0.0 |
| 64 × 64 × 64 | float64 | 11 | 28.352786000 | 17.893243000 | 1.584552672× | -10.459543000 | 0.0 |
| 256 × 256 | uint8 | 1 | 2.935144833 | 1.668160778 | 1.759509558× | -1.266984056 | 0.0 |
| 256 × 256 | float32 | 1 | 2.399950571 | 1.369730636 | 1.752133235× | -1.030219935 | 0.0 |
| 256 × 256 | float64 | 1 | 2.932399000 | 1.599475667 | 1.833350179× | -1.332923333 | 0.0 |
| 1,024 × 1,024 | uint8 | 21 | 54.013537000 | 40.707298000 | 1.326876006× | -13.306239000 | 0.0 |
| 1,024 × 1,024 | float32 | 21 | 44.280593000 | 27.296963000 | 1.622180204× | -16.983630000 | 0.0 |
| 1,024 × 1,024 | float64 | 21 | 55.481333000 | 29.923505000 | 1.854105426× | -25.557828000 | 0.0 |

Timings cover the entire public call: image conversion, storage preparation, owned uniform filtering and source-ordered Sobel sampling, allocations, clamps, cropped sums, ratio and final reducer. Even `h_size=1` executes the ordered rolling recurrence. NumPy clamps, cropped reductions and final reduction remain explicit providers; their installed dtype-promotion rules are preserved. Nonfinite, rank-4, channel-conversion and other primitive-provider routes are outside this cohort.

Before timing, the authenticated harness compares complete typed metadata and raw words for six intermediates per axis (`filt_im`, `im_sharp`, `im_blur`, `T`, `M1`, `M2`) and the public scalar. It verifies **48 accepted owned uniform stages and 96 owned sparse stages** across the 24 cases. Trace/profile callbacks are restored before measurement. The JSON exports intermediate metadata and SHA-256 values, plus public scalar words; full intermediate payloads are not exported.

The fresh Linux x86_64 measurement host used one global worker, all four reference pool limits set to one, and seven paired rounds per implementation: Python 3.11.17, NumPy 1.24.4 and SciPy 1.15.3. The CPU field records only `x86_64`; the processor model is unknown. This is a first original/current comparison, with no prior-C++ or cumulative cross-host gain claim. The last complete supported-profile qualification remains **203 image APIs**.

Validation passed: 20 focused normal/scalar XML pairs totaling **4,492 checks**, **3,696 color checks**, the entire **1,612-check matching file**, and **1,413 smoke checks for 238 registered APIs**. Separate controls passed: four Blur, four Polygon, four Euler and nine integer-width checks. The bounded private Blur component passed 484 public, 136 shared and eight intake checks across the pinned profiles, with frozen expectations unchanged.

[Release run 40645607032](https://github.com/bipxx/scikit-imagexx/actions/runs/40645607032) measured source [`e190a76500ebd33f3a2d3cb2aee26668a486b72f`](https://github.com/bipxx/scikit-imagexx/commit/e190a76500ebd33f3a2d3cb2aee26668a486b72f). Review authenticated all **1,647 committed source blobs**, all **66 artifact members**, **14 wheels including 13 runtime dependencies**, the complete installed package/core/SDK, and identical before/after measurement identities. Only this new 24-case timing ran; **18 earlier timing steps were skipped**.

Raw JSON SHA-256: `c28aecc1571f7e1f0c545a0ddeebe2becbacce291e1f10f45b49fc36aba0a744`. The sole retained performance artifact is `12798011912.zip` (84,724,656 bytes), SHA-256 `8aa52e2ac8f1466c4be0f7fe59f5d457367f77becdbbd9d45276b4c0a7135e88`; its archive remains in ignored local evidence. Review SHA-256: `9d045783aa509dcb443ae01185f9a40d385f893b182b744c23fe1a306a933873`.
