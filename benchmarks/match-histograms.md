# Native histogram matching: parity and measured performance

Retained source: `a5902e5fe0a8aaaedbece9ae030ef9e6034ce1ab`; source fingerprint `8d7f537bca4067c8e0ce89388b4a4d6bdb1e1446dd4cae162751665ac545b186`. Reference: scikit-image 0.26.0. Hardware: Intel Core i7-12700K, 12 cores/20 logical processors, verified after timing.

The new `exposure.match_histograms(image, reference, *, channel_axis=None)` implements empirical counts, normalized CDFs, interpolation and output casting in C++. Dense value ranges use bounded direct histograms and AVX2 lookup gathers; repeated sparse values use bounded hashing; high-cardinality values use radix ranks with packed 32-bit source keys where possible. SIMD key extraction retains typed ordering, merged zeros and a single NaN group. Scalar tails and a portable path share the same contract. Optional OpenMP work obeys the global worker limit, which defaults to one; radix scattering remains correct when dynamic teams are smaller than requested.

Grayscale output is floating; channel-wise output first follows the source dtype, including exact binary64-to-binary16 storage rounding before float32 promotion. NumPy only prepares/copies layout and dtype storage, validates tiny dtype metadata, and restores axes; it does not compute image histograms or interpolation. Source/reference spatial shapes and real dtypes can differ. Unusual subclasses, complex/object types and callback floating policies retain explicit fallback. Sparse unsigned ranges use bounded native storage rather than reproducing allocation failures of upstream full-range histograms.

Validation: 1,478 dedicated cases, plus per-function signatures, reference parity and dtype-metadata checks (5,518 distinct scoped cases in each SIMD mode on NumPy 1.24.4 and Windows/NumPy 2.5.4). Portable and AVX2 address/undefined-behavior sanitizer checks passed, including dynamically limited OpenMP teams, packet/tail boundaries and large independent rank oracles. An early test fixture requested a 34 GB upstream bincount allocation; that specific process was stopped, the unsigned fixture was bounded, and large-range casting checks remain on the signed sorting path. The original fixture and stopped-run log are retained. No numerical comparison tolerances were relaxed.

Source [CI 39736601020](https://github.com/bipxx/scikit-imagexx/actions/runs/39736601020): minimum-NumPy checks passed; full Windows/Linux checks were running when published.

The fixed one-worker tuning suite contains 24 cases: uint8/uint16/int16/float32/float64 across grayscale, interleaved color, planar color and signed strides, plus quantized float32/float64 controls. Base source size is 512×513×3 and reference size 515×511×3 before layout transforms. Seed 904, seven warmed shuffled repetitions, 15 ms target batches and 20 ms cooldowns are fixed. Reference pools use one thread. Calls include input preparation, allocation and output destruction. No local builds/tests overlapped timing. The independent Windows confirmation pairs exact saved initial/current packages at one/four workers; Linux/NumPy 1.24 independently compares against original scikit-image.

| Experiment | Original/C++ geometric mean (one worker) | Initial/current paired mean | Faster than original |
|---|---:|---:|---:|
| Initial native source | 0.766× | — | 4/24 |
| SIMD/hash/radix candidate | 1.639× | 2.314× | 12/24 |
| Independent Windows-11-10.0.26200-SP0, NumPy 2.5.4 | 1.498× | 2.117× | 11/24 |
| Independent Linux-6.18.54.1-microsoft-standard-WSL2-x86_64-with-glibc2.39, NumPy 1.24.4 | 2.785× | — | 21/24 |

The keep decision uses the paired 2.314× initial/current improvement; the independent one-worker confirmation records 2.117×. Absolute times from different sessions are not ratios to chain. The independent Windows one-worker suite still has 13 cases slower than scikit-image; NumPy 1.24 has three. Planar/strided preparation, continuous float64 ranks and per-pixel casting/storage remain optimization targets. This milestone adds native function coverage and improves the initial C++ kernel; it does not establish universal speedups or full-library parity.

All final rows follow, including adverse observations. Raw initial/tuning reports and samples are embedded in `match-histogram-tuning.json`. Exact packages and wheels are retained in `build/checkpoints/match-initial`, `match-tuned` and `match-histogram-validation`. The older 21-report whole-library checkpoint remains separate and historical.

| NumPy | Input | Layout | Workers | Original ms | C++ ms | Original/C++ | Initial/current |
|---|---|---|---:|---:|---:|---:|---:|
| 2.5.4 | uint8 | mono | 1 | 1.2899 | 0.5272 | 2.447× | 5.113× |
| 2.5.4 | uint8 | mono | 4 | 1.0991 | 0.4900 | 2.243× | 2.697× |
| 2.5.4 | uint8 | color | 1 | 3.7822 | 3.6120 | 1.047× | 1.679× |
| 2.5.4 | uint8 | color | 4 | 3.9894 | 2.2150 | 1.801× | 1.553× |
| 2.5.4 | uint8 | planar | 1 | 4.1627 | 5.7274 | 0.727× | 1.353× |
| 2.5.4 | uint8 | planar | 4 | 3.6887 | 3.5937 | 1.026× | 1.673× |
| 2.5.4 | uint8 | strided | 1 | 2.7405 | 3.0481 | 0.899× | 1.304× |
| 2.5.4 | uint8 | strided | 4 | 3.5225 | 3.5154 | 1.002× | 1.263× |
| 2.5.4 | uint16 | mono | 1 | 2.7680 | 3.2968 | 0.840× | 1.677× |
| 2.5.4 | uint16 | mono | 4 | 2.7388 | 3.5755 | 0.766× | 1.171× |
| 2.5.4 | uint16 | color | 1 | 9.2484 | 10.5541 | 0.876× | 1.269× |
| 2.5.4 | uint16 | color | 4 | 11.4923 | 12.0380 | 0.955× | 1.257× |
| 2.5.4 | uint16 | planar | 1 | 8.8145 | 14.0806 | 0.626× | 0.957× |
| 2.5.4 | uint16 | planar | 4 | 8.8801 | 11.8915 | 0.747× | 1.005× |
| 2.5.4 | uint16 | strided | 1 | 8.9964 | 8.9261 | 1.008× | 1.027× |
| 2.5.4 | uint16 | strided | 4 | 7.9081 | 9.7594 | 0.810× | 1.276× |
| 2.5.4 | int16 | mono | 1 | 25.7169 | 0.8771 | 29.320× | 3.086× |
| 2.5.4 | int16 | mono | 4 | 24.5735 | 0.7403 | 33.192× | 2.076× |
| 2.5.4 | int16 | color | 1 | 75.2806 | 4.9101 | 15.332× | 1.317× |
| 2.5.4 | int16 | color | 4 | 73.0091 | 2.8035 | 26.042× | 1.744× |
| 2.5.4 | int16 | planar | 1 | 79.9341 | 6.8183 | 11.723× | 1.321× |
| 2.5.4 | int16 | planar | 4 | 72.3304 | 4.2986 | 16.827× | 1.385× |
| 2.5.4 | int16 | strided | 1 | 37.5135 | 3.4095 | 11.003× | 1.217× |
| 2.5.4 | int16 | strided | 4 | 38.3812 | 3.5091 | 10.938× | 1.184× |
| 2.5.4 | float32 | mono | 1 | 12.7825 | 14.7358 | 0.867× | 3.032× |
| 2.5.4 | float32 | mono | 4 | 13.3791 | 9.8631 | 1.356× | 4.128× |
| 2.5.4 | float32 | color | 1 | 37.6664 | 48.9175 | 0.770× | 2.769× |
| 2.5.4 | float32 | color | 4 | 36.9426 | 30.9077 | 1.195× | 4.057× |
| 2.5.4 | float32 | planar | 1 | 37.9140 | 46.4650 | 0.816× | 2.845× |
| 2.5.4 | float32 | planar | 4 | 40.0421 | 33.5474 | 1.194× | 3.732× |
| 2.5.4 | float32 | strided | 1 | 19.4352 | 24.0373 | 0.809× | 2.656× |
| 2.5.4 | float32 | strided | 4 | 18.9287 | 22.9730 | 0.824× | 2.726× |
| 2.5.4 | float64 | mono | 1 | 13.2261 | 22.1561 | 0.597× | 2.026× |
| 2.5.4 | float64 | mono | 4 | 17.9530 | 16.9565 | 1.059× | 2.662× |
| 2.5.4 | float64 | color | 1 | 80.6412 | 113.9754 | 0.708× | 1.567× |
| 2.5.4 | float64 | color | 4 | 57.5263 | 42.4398 | 1.355× | 3.256× |
| 2.5.4 | float64 | planar | 1 | 58.7261 | 86.5098 | 0.679× | 1.707× |
| 2.5.4 | float64 | planar | 4 | 46.6639 | 42.9566 | 1.086× | 3.177× |
| 2.5.4 | float64 | strided | 1 | 20.0335 | 32.6605 | 0.613× | 1.920× |
| 2.5.4 | float64 | strided | 4 | 19.3105 | 32.4524 | 0.595× | 1.972× |
| 2.5.4 | float32 | quantized-mono | 1 | 5.0864 | 4.1520 | 1.225× | 4.780× |
| 2.5.4 | float32 | quantized-mono | 4 | 4.7994 | 2.3738 | 2.022× | 7.501× |
| 2.5.4 | float32 | quantized-color | 1 | 14.3334 | 12.8553 | 1.115× | 4.655× |
| 2.5.4 | float32 | quantized-color | 4 | 14.7783 | 7.9838 | 1.851× | 6.634× |
| 2.5.4 | float64 | quantized-mono | 1 | 6.6367 | 4.0025 | 1.658× | 4.820× |
| 2.5.4 | float64 | quantized-mono | 4 | 5.9771 | 2.4963 | 2.394× | 7.286× |
| 2.5.4 | float64 | quantized-color | 1 | 27.1766 | 13.2676 | 2.048× | 4.500× |
| 2.5.4 | float64 | quantized-color | 4 | 23.5166 | 9.2417 | 2.545× | 6.271× |
| 1.24.4 | uint8 | mono | 1 | 1.1545 | 0.5454 | 2.117× | — |
| 1.24.4 | uint8 | mono | 4 | 1.1071 | 0.6570 | 1.685× | — |
| 1.24.4 | uint8 | color | 1 | 4.2413 | 3.9851 | 1.064× | — |
| 1.24.4 | uint8 | color | 4 | 3.9336 | 3.1378 | 1.254× | — |
| 1.24.4 | uint8 | planar | 1 | 3.4842 | 4.9889 | 0.698× | — |
| 1.24.4 | uint8 | planar | 4 | 4.3784 | 4.1564 | 1.053× | — |
| 1.24.4 | uint8 | strided | 1 | 2.1479 | 2.6136 | 0.822× | — |
| 1.24.4 | uint8 | strided | 4 | 1.9772 | 2.6524 | 0.745× | — |
| 1.24.4 | uint16 | mono | 1 | 2.4924 | 1.9916 | 1.251× | — |
| 1.24.4 | uint16 | mono | 4 | 2.5114 | 2.1868 | 1.148× | — |
| 1.24.4 | uint16 | color | 1 | 8.4666 | 8.3620 | 1.013× | — |
| 1.24.4 | uint16 | color | 4 | 8.7376 | 6.7858 | 1.288× | — |
| 1.24.4 | uint16 | planar | 1 | 8.0744 | 9.6615 | 0.836× | — |
| 1.24.4 | uint16 | planar | 4 | 8.6442 | 8.8320 | 0.979× | — |
| 1.24.4 | uint16 | strided | 1 | 6.4295 | 6.2178 | 1.034× | — |
| 1.24.4 | uint16 | strided | 4 | 6.3175 | 6.5091 | 0.971× | — |
| 1.24.4 | int16 | mono | 1 | 24.7077 | 0.7002 | 35.287× | — |
| 1.24.4 | int16 | mono | 4 | 25.0584 | 0.8437 | 29.699× | — |
| 1.24.4 | int16 | color | 1 | 74.5801 | 4.3500 | 17.145× | — |
| 1.24.4 | int16 | color | 4 | 74.2662 | 2.8644 | 25.928× | — |
| 1.24.4 | int16 | planar | 1 | 76.1866 | 6.3536 | 11.991× | — |
| 1.24.4 | int16 | planar | 4 | 77.8048 | 4.5529 | 17.089× | — |
| 1.24.4 | int16 | strided | 1 | 38.9888 | 3.2131 | 12.134× | — |
| 1.24.4 | int16 | strided | 4 | 39.0213 | 3.3355 | 11.699× | — |
| 1.24.4 | float32 | mono | 1 | 39.8779 | 15.0537 | 2.649× | — |
| 1.24.4 | float32 | mono | 4 | 41.0945 | 14.3440 | 2.865× | — |
| 1.24.4 | float32 | color | 1 | 132.2022 | 53.5515 | 2.469× | — |
| 1.24.4 | float32 | color | 4 | 131.8682 | 46.5531 | 2.833× | — |
| 1.24.4 | float32 | planar | 1 | 118.6091 | 49.7636 | 2.383× | — |
| 1.24.4 | float32 | planar | 4 | 125.0161 | 56.5771 | 2.210× | — |
| 1.24.4 | float32 | strided | 1 | 53.7565 | 26.4015 | 2.036× | — |
| 1.24.4 | float32 | strided | 4 | 54.5468 | 25.5220 | 2.137× | — |
| 1.24.4 | float64 | mono | 1 | 36.6912 | 20.3653 | 1.802× | — |
| 1.24.4 | float64 | mono | 4 | 38.1709 | 16.3071 | 2.341× | — |
| 1.24.4 | float64 | color | 1 | 130.3285 | 79.2848 | 1.644× | — |
| 1.24.4 | float64 | color | 4 | 131.1421 | 65.9618 | 1.988× | — |
| 1.24.4 | float64 | planar | 1 | 125.8105 | 83.6726 | 1.504× | — |
| 1.24.4 | float64 | planar | 4 | 126.2783 | 77.1046 | 1.638× | — |
| 1.24.4 | float64 | strided | 1 | 62.1392 | 32.5185 | 1.911× | — |
| 1.24.4 | float64 | strided | 4 | 69.4512 | 45.1599 | 1.538× | — |
| 1.24.4 | float32 | quantized-mono | 1 | 27.6993 | 4.4923 | 6.166× | — |
| 1.24.4 | float32 | quantized-mono | 4 | 23.6344 | 3.5439 | 6.669× | — |
| 1.24.4 | float32 | quantized-color | 1 | 66.6502 | 10.7695 | 6.189× | — |
| 1.24.4 | float32 | quantized-color | 4 | 66.7852 | 8.3687 | 7.980× | — |
| 1.24.4 | float64 | quantized-mono | 1 | 26.6249 | 4.1092 | 6.479× | — |
| 1.24.4 | float64 | quantized-mono | 4 | 24.1840 | 2.9070 | 8.319× | — |
| 1.24.4 | float64 | quantized-color | 1 | 76.1820 | 13.3653 | 5.700× | — |
| 1.24.4 | float64 | quantized-color | 4 | 74.7484 | 10.4842 | 7.130× | — |
