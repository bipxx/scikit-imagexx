# LPI filters: first minimum-Linux Release result

[Run 40311937607](https://github.com/bipxx/scikit-imagexx/actions/runs/40311937607) measures the exact `c895db2ae1c4debf3822a6ed43964625f5416bed` product against original scikit-image 0.26.0. **All 18 cases are faster**, with a **1.330318164× geometric mean speedup**. There are no slower LPI cases in this fixed set. This is the first original/current comparison, not a gain over an earlier C++ implementation.

Each of `filter_forward`, `filter_inverse` and `wiener` uses 256² and 1024² images in uint8, float32 and float64, one worker and seven paired rounds. The [exact raw report](lpi-minimum-initial.json) retains all **252 latency samples**. Calls include the Gaussian impulse callback, allocations, construction of a new filter and its transfer cache; this does not measure reuse of an existing `LPIFilter2D` cache.

## Results

Values are per-report medians. Gain is original/current; the final column is current minus original, so negative values are time saved. Every case is included.

| API | Size | Dtype | Original ms | Current ms | Gain | Current − original ms |
|---|---:|---|---:|---:|---:|---:|
| filters.filter_forward | 256² | uint8 | 26.129497 | 18.925498 | 1.380650× | -7.203999 |
| filters.filter_inverse | 256² | uint8 | 33.765312 | 26.453612 | 1.276397× | -7.311700 |
| filters.wiener | 256² | uint8 | 37.271004 | 29.905631 | 1.246287× | -7.365373 |
| filters.filter_forward | 256² | float32 | 14.628122 | 10.645726 | 1.374084× | -3.982396 |
| filters.filter_inverse | 256² | float32 | 16.167156 | 12.163818 | 1.329119× | -4.003338 |
| filters.wiener | 256² | float32 | 16.413916 | 12.628669 | 1.299734× | -3.785247 |
| filters.filter_forward | 256² | float64 | 22.027138 | 15.137848 | 1.455104× | -6.889290 |
| filters.filter_inverse | 256² | float64 | 31.216447 | 24.061199 | 1.297377× | -7.155248 |
| filters.wiener | 256² | float64 | 29.295531 | 22.374071 | 1.309352× | -6.921460 |
| filters.filter_forward | 1024² | uint8 | 427.529804 | 300.820511 | 1.421212× | -126.709293 |
| filters.filter_inverse | 1024² | uint8 | 579.263920 | 452.610859 | 1.279828× | -126.653061 |
| filters.wiener | 1024² | uint8 | 583.246893 | 459.071714 | 1.270492× | -124.175179 |
| filters.filter_forward | 1024² | float32 | 222.870425 | 154.539139 | 1.442162× | -68.331286 |
| filters.filter_inverse | 1024² | float32 | 301.153339 | 231.783861 | 1.299285× | -69.369478 |
| filters.wiener | 1024² | float32 | 301.442897 | 231.306255 | 1.303220× | -70.136642 |
| filters.filter_forward | 1024² | float64 | 418.845502 | 291.618651 | 1.436278× | -127.226851 |
| filters.filter_inverse | 1024² | float64 | 579.889779 | 452.642207 | 1.281122× | -127.247572 |
| filters.wiener | 1024² | float64 | 584.417388 | 459.205386 | 1.272671× | -125.212002 |

| Group | Cases | Wins / losses | Geometric mean gain |
|---|---:|---:|---:|
| filters.filter_forward | 6 | 6 / 0 | 1.417915673× |
| filters.filter_inverse | 6 | 6 / 0 | 1.293730110× |
| filters.wiener | 6 | 6 / 0 | 1.283430849× |
| All LPI cases | 18 | 18 / 0 | 1.330318164× |

## Correctness and measurement boundary

The genuine installed minimum-NumPy Release product passes **161 LPI tests in normal mode and 161 with SIMD disabled**, plus **175 Butterworth tests in each mode** and **1,334 registry smoke checks**. The existing minimum-profile regression job also completes successfully. Before timing, the benchmark checks output structure/dtype and the frozen numerical policy against a fresh original call; all 18 reported maximum absolute errors are zero. The frozen 80 LPI specifications and their policy were not changed for these measurements.

A fresh measurement host installs the exact archived wheelhouse offline after correctness finishes. It performs no builds or pytest runs. The benchmark checks the native thread getter after setting one worker, after the parity call and after timing; SciPy FFT reports one worker before and after the run. Reference pools are capped at one thread. Normal AVX2 dispatch is recorded; scalar mode has correctness coverage here, not a separate speed result. Source, wheel, installed package, runtime libraries and SDK identities match before and after measurement.

Runtime: **CPython 3.11.17, NumPy 1.24.4, SciPy 1.15.3, scikit-image 0.26.0**, Linux x86_64 (`Linux-6.17.0-1022-azure-x86_64-with-glibc2.39`). The report captures architecture but not a specific CPU model. These are one-host minimum-runtime observations, not cross-CPU or full supported-profile guarantees.

## Native and provider scope

C++ owns the FFT stages. NumPy supplies coordinates, casting, padding, products, reciprocal/gain/Wiener arithmetic, permutation, cropping and absolute value; the Gaussian impulse callback remains Python. The measured public functions are native-partial. These cases do not establish full C++ execution, arbitrary callback/storage support, persistent-cache performance or complete `LPIFilter2D` behavior.

Main source coverage is 215 native-partial image functions plus three metadata functions, with five partial classes and 253 upstream-only entries. The separately accepted combined supported-profile baseline remains **203 image APIs**. This Release result does not promote that qualification. The [earlier Butterworth result](butterworth-minimum-initial.md) and [73-case legacy regression comparison](phase3-butterworth-legacy-same-host.md), including their losses, remain unchanged.

## Exact identities and retained evidence

| Identity | Value |
|---|---|
| Source revision | `c895db2ae1c4debf3822a6ed43964625f5416bed` |
| Product-source fingerprint | `869697187dc059a10dba919a815345152d900dce3da26e774ac5210eac8de2ac` |
| Wheel | `scikit_imagexx-0.1.0-cp311-cp311-linux_x86_64.whl` |
| Wheel SHA-256 | `a30a1043249407d610612544775e116025c2d24211362e6843e1a0c73ecf8676` |
| Sole core SHA-256 | `250e214cf86ba16c2bfce4f396958402aa1b2f71782bb0ae8f173125b49b28d9` |
| Pre/post measurement manifest SHA-256 | `ce1fdca2c2b7ad2d7ac1dd86cda9440712b9f81d97b26d3e723ef7d30b30f0fd` |
| Raw LPI JSON SHA-256 | `79d710a79cc41bf86be70897f1eb243b8effbcbf99f715924246edf5c094b640` |

The published JSON is an exact 17,221-byte copy. The single retained performance artifact is `12645940673`, **84,296,678 bytes**, SHA-256 `bdb795723dab1c24a58471fbbaf4560ebf3f56aa211635fb493bdf18ce9454a4`. All 27 members pass CRC/size/SHA checks; 14 locked wheels, the complete installed package and SDK, 317 committed source blobs and both identity receipts authenticate. The artifact already contains the producer wheels, so no duplicate producer artifact was downloaded.

Local evidence is under `build/phase3-lpi-first-release-v1/`. Final review SHA-256: `6b53ddc1fd7ce0e39a46e88745af5d013247e9e5ba30f443d1ff7d6ca6355530`; handoff SHA-256: `7fc7876bbebff56d42c8f86d714d6c6823eb57ff38252feef086727f7eae4d4c`. Original logs, XML, raw samples and the initial metadata-review log-format failure remain retained. No numerical rerun or data replacement was used.
