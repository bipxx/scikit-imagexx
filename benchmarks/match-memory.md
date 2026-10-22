# Histogram-matching rank storage and floating output tuning

Retained source: `a0740f98a8ba4d411cce0242bf35805cad9f6728`; source fingerprint `aaa1de1946731b099614b94f02163a21b2b400feb8c2e992e663a56982f50c4b`. Paired previous build: `a5902e5`. Reference: scikit-image 0.26.0.

The retained change removes decoded source values that the CDF mapper does not consume, reserves rank storage, and releases reference keys/counts/CDF buffers after their final use. Native floating output uses a typed pass: float64 values store directly; float32 values convert once per unique source group, retain conversion warning flags, then scatter without a per-pixel cast callback. Binary16 rounding and endian output retain the existing general path. The matching translation unit also disables contraction with the project’s strict floating compile settings. Public Python/C++ signatures and global optional worker limits are unchanged.

Validation: 5,590 scoped matching/API/metadata cases passed in each SIMD mode on Windows/NumPy 2.5.4 and Linux/NumPy 1.24.4. New cases compare underflow, overflow, nonfinite values and warn/raise policies against the original library. Eight further tiny sparse-range oracles passed separately in all four configurations (5,598 distinct scoped cases). AVX2 and portable address/undefined-behavior sanitizers passed, including dynamically reduced OpenMP teams. No numerical or warning comparisons were relaxed.

Source [CI 39741060736](https://github.com/bipxx/scikit-imagexx/actions/runs/39741060736) passed: Linux recorded 43,061 Python tests in each SIMD mode; Windows recorded 43,059 passed and two skipped in each mode. Native C++ SIMD, dynamically limited workers and portable checks passed on both platforms. The selected minimum-NumPy suite passed 18,885 tests. The eight later tiny-range cases were checked locally; they are not included in those CI counts.

The unchanged 24-case corpus covers uint8/uint16/int16/float32/float64 grayscale, interleaved color, planar color and signed strides, plus quantized floating controls. Source shape is 512×513×3 and reference shape 515×511×3 before transformations; seed 904, seven shuffled warmed repetitions, 15 ms target batches and 20 ms cooldowns remain fixed. Calls include preparation, allocation and output destruction. Original thread pools use one; C++ limits are one/four and small views stay serial. No local builds/tests overlapped measurement. Saved previous packages are verified by binary/Python hashes and matching ISA dispatch.

| Run | Worker limit | Previous/current geometric mean | Original/current | Faster than previous | Faster than original |
|---|---:|---:|---:|---:|---:|
| Primary Windows experiment | 1 | 1.149× | 1.754× | 22/24 | 15/24 |
| Independent Windows-11-10.0.26200-SP0, NumPy 2.5.4 | 1 | 1.124× | 1.698× | 22/24 | 16/24 |
| Independent Windows-11-10.0.26200-SP0, NumPy 2.5.4 | 4 | 1.134× | 2.114× | 20/24 | 20/24 |
| Independent Linux-6.18.54.1-microsoft-standard-WSL2-x86_64-with-glibc2.39, NumPy 1.24.4 | 1 | 1.168× | 3.313× | 21/24 | 21/24 |
| Independent Linux-6.18.54.1-microsoft-standard-WSL2-x86_64-with-glibc2.39, NumPy 1.24.4 | 4 | 1.107× | 3.549× | 19/24 | 22/24 |

The 1.149× primary previous/current ratio justified retaining this variant. Independent one-worker ratios are 1.124× on Windows and 1.168× on minimum NumPy. Ratios from separate runs are not multiplied or derived from absolute times. The primary float64-color regression (0.922×) did not reproduce in either confirmation, but all samples remain in the history. The independent Windows int16-mono case regressed to 0.548× versus the previous build; int16-color also regressed. These observations remain open optimization targets. Eight Windows one-worker cases and three minimum-NumPy cases still trail original scikit-image. This is a scoped improvement, not universal speedup or full-library completion.

A second experiment sized dense rank vectors from populated histogram bins and selected sorting for tiny sparse ranges. It passed 1,558 dedicated cases and dynamic-worker sanitizer checks, but its paired one-worker ratio was only 1.006×. A 2,000-draw resampling of per-function median samples gave a 95% interval [0.947, 1.048], which includes parity; this interval does not cover between-run effects. Integer color/planar regressions accompanied the weak aggregate result. The allocation experiment was discarded, its exact wheel/source patch and every raw sample were preserved, and the kept package was restored without rebuilding it. The added tiny-range parity tests remain.

Raw primary, independent and rejected reports are in `match-memory-tuning.json`; final platform reports are also available separately. Exact packages, wheels, source and validation evidence are preserved under `build/checkpoints/match-memory-*`. The historical 21-report whole-library checkpoint is unchanged.

| NumPy | Input | Layout | Worker limit | Original ms | Current ms | Original/current | Previous/current |
|---|---|---|---:|---:|---:|---:|---:|
| 2.5.4 | uint8 | mono | 1 | 1.2066 | 0.4929 | 2.448× | 1.152× |
| 2.5.4 | uint8 | mono | 4 | 1.1035 | 0.5147 | 2.144× | 0.977× |
| 2.5.4 | uint8 | color | 1 | 3.6764 | 3.4186 | 1.075× | 1.008× |
| 2.5.4 | uint8 | color | 4 | 4.2888 | 2.1677 | 1.979× | 0.996× |
| 2.5.4 | uint8 | planar | 1 | 3.9776 | 5.2493 | 0.758× | 1.084× |
| 2.5.4 | uint8 | planar | 4 | 3.8152 | 3.8529 | 0.990× | 0.950× |
| 2.5.4 | uint8 | strided | 1 | 2.5251 | 2.9529 | 0.855× | 1.123× |
| 2.5.4 | uint8 | strided | 4 | 2.2017 | 2.8909 | 0.762× | 1.077× |
| 2.5.4 | uint16 | mono | 1 | 2.5797 | 2.2341 | 1.155× | 1.263× |
| 2.5.4 | uint16 | mono | 4 | 2.5894 | 2.1566 | 1.201× | 1.233× |
| 2.5.4 | uint16 | color | 1 | 8.3973 | 9.2611 | 0.907× | 1.130× |
| 2.5.4 | uint16 | color | 4 | 9.5977 | 6.7694 | 1.418× | 1.203× |
| 2.5.4 | uint16 | planar | 1 | 8.4416 | 10.6942 | 0.789× | 1.102× |
| 2.5.4 | uint16 | planar | 4 | 10.2610 | 8.9189 | 1.150× | 1.315× |
| 2.5.4 | uint16 | strided | 1 | 6.8541 | 6.7731 | 1.012× | 1.197× |
| 2.5.4 | uint16 | strided | 4 | 6.9138 | 6.8788 | 1.005× | 1.193× |
| 2.5.4 | int16 | mono | 1 | 31.2667 | 1.8690 | 16.729× | 0.548× |
| 2.5.4 | int16 | mono | 4 | 40.6421 | 1.5092 | 26.930× | 1.574× |
| 2.5.4 | int16 | color | 1 | 90.4339 | 8.2322 | 10.985× | 0.853× |
| 2.5.4 | int16 | color | 4 | 75.5302 | 3.7580 | 20.099× | 1.041× |
| 2.5.4 | int16 | planar | 1 | 73.4156 | 6.4234 | 11.429× | 1.107× |
| 2.5.4 | int16 | planar | 4 | 73.1136 | 6.4881 | 11.269× | 0.933× |
| 2.5.4 | int16 | strided | 1 | 56.7334 | 7.2355 | 7.841× | 1.135× |
| 2.5.4 | int16 | strided | 4 | 43.5707 | 3.8251 | 11.391× | 1.035× |
| 2.5.4 | float32 | mono | 1 | 15.8182 | 11.9243 | 1.327× | 1.211× |
| 2.5.4 | float32 | mono | 4 | 14.9989 | 8.5049 | 1.764× | 1.404× |
| 2.5.4 | float32 | color | 1 | 43.2907 | 38.3377 | 1.129× | 1.200× |
| 2.5.4 | float32 | color | 4 | 39.2909 | 31.1280 | 1.262× | 1.182× |
| 2.5.4 | float32 | planar | 1 | 42.6828 | 41.4388 | 1.030× | 1.329× |
| 2.5.4 | float32 | planar | 4 | 41.5299 | 31.9249 | 1.301× | 1.152× |
| 2.5.4 | float32 | strided | 1 | 24.9077 | 21.2774 | 1.171× | 1.136× |
| 2.5.4 | float32 | strided | 4 | 18.2550 | 20.4473 | 0.893× | 1.094× |
| 2.5.4 | float64 | mono | 1 | 18.1439 | 20.2711 | 0.895× | 1.260× |
| 2.5.4 | float64 | mono | 4 | 14.1296 | 10.8340 | 1.304× | 1.321× |
| 2.5.4 | float64 | color | 1 | 58.8367 | 72.8569 | 0.808× | 1.223× |
| 2.5.4 | float64 | color | 4 | 45.1160 | 38.5842 | 1.169× | 1.021× |
| 2.5.4 | float64 | planar | 1 | 43.2339 | 53.4769 | 0.808× | 1.191× |
| 2.5.4 | float64 | planar | 4 | 47.2124 | 38.1852 | 1.236× | 1.249× |
| 2.5.4 | float64 | strided | 1 | 20.6386 | 25.8263 | 0.799× | 1.138× |
| 2.5.4 | float64 | strided | 4 | 21.0133 | 25.0228 | 0.840× | 1.210× |
| 2.5.4 | float32 | quantized-mono | 1 | 5.3204 | 3.0640 | 1.736× | 1.170× |
| 2.5.4 | float32 | quantized-mono | 4 | 5.3806 | 2.2536 | 2.388× | 1.125× |
| 2.5.4 | float32 | quantized-color | 1 | 15.6536 | 8.5688 | 1.827× | 1.285× |
| 2.5.4 | float32 | quantized-color | 4 | 16.2553 | 7.3129 | 2.223× | 1.061× |
| 2.5.4 | float64 | quantized-mono | 1 | 7.3602 | 3.2607 | 2.257× | 1.181× |
| 2.5.4 | float64 | quantized-mono | 4 | 6.3465 | 2.4220 | 2.620× | 1.023× |
| 2.5.4 | float64 | quantized-color | 1 | 23.3681 | 10.0968 | 2.314× | 1.296× |
| 2.5.4 | float64 | quantized-color | 4 | 25.8848 | 10.2061 | 2.536× | 1.066× |
| 1.24.4 | uint8 | mono | 1 | 1.0298 | 0.5089 | 2.024× | 1.011× |
| 1.24.4 | uint8 | mono | 4 | 1.1051 | 0.6638 | 1.665× | 0.960× |
| 1.24.4 | uint8 | color | 1 | 3.8717 | 3.6879 | 1.050× | 1.039× |
| 1.24.4 | uint8 | color | 4 | 3.7703 | 2.5212 | 1.495× | 0.976× |
| 1.24.4 | uint8 | planar | 1 | 3.7893 | 5.1240 | 0.740× | 1.019× |
| 1.24.4 | uint8 | planar | 4 | 3.5694 | 3.9716 | 0.899× | 1.017× |
| 1.24.4 | uint8 | strided | 1 | 1.9294 | 2.6077 | 0.740× | 1.012× |
| 1.24.4 | uint8 | strided | 4 | 2.0187 | 2.6378 | 0.765× | 0.983× |
| 1.24.4 | uint16 | mono | 1 | 2.6152 | 1.9964 | 1.310× | 1.227× |
| 1.24.4 | uint16 | mono | 4 | 2.6991 | 2.0844 | 1.295× | 1.118× |
| 1.24.4 | uint16 | color | 1 | 9.0264 | 7.6117 | 1.186× | 1.163× |
| 1.24.4 | uint16 | color | 4 | 8.3617 | 7.5298 | 1.110× | 0.985× |
| 1.24.4 | uint16 | planar | 1 | 8.7333 | 8.9697 | 0.974× | 1.143× |
| 1.24.4 | uint16 | planar | 4 | 8.5386 | 8.1236 | 1.051× | 1.064× |
| 1.24.4 | uint16 | strided | 1 | 6.6432 | 5.7922 | 1.147× | 1.091× |
| 1.24.4 | uint16 | strided | 4 | 6.9191 | 5.8715 | 1.178× | 1.145× |
| 1.24.4 | int16 | mono | 1 | 24.6518 | 0.6676 | 36.929× | 0.969× |
| 1.24.4 | int16 | mono | 4 | 25.3715 | 0.8387 | 30.252× | 1.025× |
| 1.24.4 | int16 | color | 1 | 76.0417 | 4.3441 | 17.504× | 1.055× |
| 1.24.4 | int16 | color | 4 | 74.4232 | 2.9630 | 25.117× | 1.026× |
| 1.24.4 | int16 | planar | 1 | 76.7039 | 6.5208 | 11.763× | 0.960× |
| 1.24.4 | int16 | planar | 4 | 77.0112 | 4.8076 | 16.019× | 1.086× |
| 1.24.4 | int16 | strided | 1 | 35.1161 | 3.1162 | 11.269× | 0.984× |
| 1.24.4 | int16 | strided | 4 | 35.1070 | 2.8523 | 12.308× | 1.010× |
| 1.24.4 | float32 | mono | 1 | 36.8898 | 10.2701 | 3.592× | 1.304× |
| 1.24.4 | float32 | mono | 4 | 36.1718 | 9.7065 | 3.727× | 1.264× |
| 1.24.4 | float32 | color | 1 | 115.3545 | 34.8812 | 3.307× | 1.239× |
| 1.24.4 | float32 | color | 4 | 116.6042 | 31.9228 | 3.653× | 1.223× |
| 1.24.4 | float32 | planar | 1 | 119.3519 | 39.2355 | 3.042× | 1.294× |
| 1.24.4 | float32 | planar | 4 | 117.6479 | 36.9027 | 3.188× | 1.257× |
| 1.24.4 | float32 | strided | 1 | 55.9533 | 15.7425 | 3.554× | 1.692× |
| 1.24.4 | float32 | strided | 4 | 54.5586 | 15.4064 | 3.541× | 1.620× |
| 1.24.4 | float64 | mono | 1 | 36.7945 | 14.5565 | 2.528× | 1.249× |
| 1.24.4 | float64 | mono | 4 | 36.3340 | 13.1347 | 2.766× | 1.326× |
| 1.24.4 | float64 | color | 1 | 115.2442 | 51.5780 | 2.234× | 1.207× |
| 1.24.4 | float64 | color | 4 | 114.9641 | 39.5250 | 2.909× | 1.318× |
| 1.24.4 | float64 | planar | 1 | 113.3685 | 53.5669 | 2.116× | 1.287× |
| 1.24.4 | float64 | planar | 4 | 126.1606 | 53.6740 | 2.350× | 1.224× |
| 1.24.4 | float64 | strided | 1 | 56.8525 | 23.7233 | 2.396× | 1.150× |
| 1.24.4 | float64 | strided | 4 | 57.0882 | 25.9199 | 2.202× | 1.148× |
| 1.24.4 | float32 | quantized-mono | 1 | 22.3155 | 2.4651 | 9.053× | 1.374× |
| 1.24.4 | float32 | quantized-mono | 4 | 22.7684 | 2.6145 | 8.709× | 0.965× |
| 1.24.4 | float32 | quantized-color | 1 | 68.1408 | 8.2792 | 8.230× | 1.282× |
| 1.24.4 | float32 | quantized-color | 4 | 70.0376 | 8.3904 | 8.347× | 1.013× |
| 1.24.4 | float64 | quantized-mono | 1 | 23.4357 | 2.6108 | 8.976× | 1.322× |
| 1.24.4 | float64 | quantized-mono | 4 | 24.9537 | 2.6217 | 9.518× | 1.018× |
| 1.24.4 | float64 | quantized-color | 1 | 76.1269 | 10.5746 | 7.199× | 1.224× |
| 1.24.4 | float64 | quantized-color | 4 | 74.7718 | 9.5074 | 7.865× | 1.035× |
