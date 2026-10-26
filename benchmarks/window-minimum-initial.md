# Window: first minimum-Linux Release result

[Run 40317428784](https://github.com/bipxx/scikit-imagexx/actions/runs/40317428784) measures exact source `33030dce649477a828bab9295746aaf043531058` against original scikit-image 0.26.0. **Both cases are faster**, with a **7.288846162× geometric mean speedup** and no losses in this two-case set. This is the first original/current comparison, not a gain over an earlier C++ implementation.

The calls are `window("hann", (256, 256))` and `window("hann", (1024, 1024))`, using default order1 and one worker. Window consumes a shape, not image pixels; float64 is the output dtype. There are two distinct workloads, with no duplicated uint8/float32 fixture labels. The [exact raw report](window-minimum-initial.json) retains all **28 latency samples** from seven paired rounds per case.

| Shape | Output dtype | Original ms | Current ms | Gain | Current − original ms |
|---|---|---:|---:|---:|---:|
| 256² | float64 | 2.690202600 | 0.391585759 | 6.870021549× | -2.298616841 |
| 1024² | float64 | 39.051996000 | 5.049911500 | 7.733204037× | -34.002084500 |

Values are medians of warmed allocation-inclusive public calls. Gain is original/current; a negative final column is time saved. Both reported maximum absolute output errors are zero.

## Correctness and measurement boundary

The genuine installed minimum-Linux Release product passes **179 window tests in normal mode and 179 with SIMD disabled**, plus **161 LPI and 175 Butterworth tests in each mode**, and **1,338 registry smoke checks**. All older regression steps in the minimum producer also pass. The window file retains its frozen 89 specifications, including declared strict-refusal/error controls; passing those controls does not give unsupported higher orders numerical parity credit. No source case or tolerance was changed for timing.

A fresh measurement host installs the exact archived wheelhouse offline after correctness finishes, with no builds or pytest runs in the measurement job. Calls include SciPy 1D window generation, coordinate construction, sampling, clipping, output allocation and destruction. The native getter is asserted to be one after setting workers, after parity and after timing. Reference thread pools are capped before imports. Normal AVX2 dispatch is recorded; SIMD-disabled mode has correctness coverage here, not a separate speed result. This window path makes no SciPy FFT-worker claim. Source, wheel, installed package, runtime libraries and SDK identities match before and after measurement.

Runtime: **CPython 3.11.17, NumPy 1.24.4, SciPy 1.15.3, scikit-image 0.26.0**, Linux x86_64 (`Linux-6.17.0-1022-azure-x86_64-with-glibc2.39`). The report records architecture, not a specific CPU model. These are observations on one minimum-runtime host, not cross-CPU or full supported-profile guarantees.

## Native and provider scope

C++ owns order-0/1 grid-constant sampling. SciPy `signal.get_window` supplies the one-dimensional window; NumPy supplies radial coordinates and clipping. The measured function remains native-partial. Higher interpolation orders and mixed-empty shapes remain outside the current native frontier; two Hann cases do not establish complete parameter, storage or callback support.

Source coverage is **216 native-partial image functions plus three metadata functions**, with five partial classes and 252 upstream-only entries. The separately accepted combined supported-profile baseline remains **203 image APIs**. This Release result does not promote that qualification. The earlier [LPI result](lpi-minimum-initial.md), [Butterworth result](butterworth-minimum-initial.md) and [73-case legacy comparison](phase3-butterworth-legacy-same-host.md), including their losses, remain unchanged. This run did not repeat the fixed73 comparison.

## Exact identities and retained evidence

| Identity | Value |
|---|---|
| Source revision | `33030dce649477a828bab9295746aaf043531058` |
| Product-source fingerprint | `3ee814cdbcfd9af47e5d4eefe088b360a1100ebb1596445852fffd7622de69fd` |
| Wheel | `scikit_imagexx-0.1.0-cp311-cp311-linux_x86_64.whl` |
| Wheel SHA-256 | `6399074bff50ae12541660c6404148210208ed2cee7e04407062ee1ae0f7a5b4` |
| Sole core SHA-256 | `655b60b1caa4fcfa9ad26463c3d4c29992d6eaa158fe8e9698897f7cd4c7f821` |
| Pre/post measurement manifest SHA-256 | `6a0fed796d1802be37e86a7291e912b251b5fef2821adcb5fbbf1a9a34e8da1e` |
| Raw window JSON SHA-256 | `6d2e08ed3c59ea73b43b921ce9263b790067bbbc5511a9c93b62f7de3ac3505d` |

The published JSON is an exact **3,340-byte** copy. The single retained performance artifact is `12647573498`, **84,313,186 bytes**, SHA-256 `3aea65b64fca078f27666234247d12b9c04dd694d9b45ec4d2f7e4faa72b8ff9`. All 31 members pass CRC/size/SHA checks; 14 locked wheels, the complete installed package and SDK, 324 committed source blobs and both identity receipts authenticate. The artifact already includes the producer wheels, so no duplicate producer download was needed.

Local evidence is under `build/phase3-window-first-release-v1/`. Final review SHA-256: `5e0053632859f583a5ba8cdcd6a9ed80dd04e6a8a25cfc686fcae5668584fb38`; handoff SHA-256: `35d9998f3b9ec7318d93fe19a3deef6a3b8c9411c47075e6240500363ecfe50c`. Original logs, XML and raw samples remain retained. Later documentation and feature-selector changes are not attributed to this producer; its exact measured source is 33030dc.
