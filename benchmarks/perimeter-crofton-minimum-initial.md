# Crofton perimeter: first minimum-runtime comparison

All six cases were faster than scikit-image 0.26.0 on this host: **5.6737467849× geometric mean**, with no losses. The [unchanged raw JSON](perimeter-crofton-minimum-initial.json) retains all **84 samples**: seven paired rounds × two implementations × six cases. These are warmed, allocation-inclusive public calls at one thread, with fallback disabled and original thread pools limited to one thread. Each case passed its frozen parity check before timing.

| Square size | Input dtype | Original ms | Current ms | Original/current | Maximum absolute error |
|---:|---|---:|---:|---:|---:|
| 256 | uint8 | 0.711092000 | 0.097198446 | 7.315878307× | 2.2737367544e-13 |
| 256 | float32 | 0.702573267 | 0.133569188 | 5.259995062× | 2.2737367544e-13 |
| 256 | float64 | 0.699757250 | 0.151381818 | 4.622465620× | 2.2737367544e-13 |
| 1024 | uint8 | 9.575428000 | 1.267399500 | 7.555177353× | 0 |
| 1024 | float32 | 9.713928000 | 1.811175000 | 5.363329330× | 0 |
| 1024 | float64 | 9.742762000 | 2.105072333 | 4.628231461× | 0 |

Times are medians in milliseconds; the table is rounded and the JSON preserves the original precision. Every measured result is included. This is a first original/current comparison for default `directions=4`, not a measured gain over a prior C++ product or a claim of uniform performance on other machines and inputs.

The owned stages perform real-array thresholding, the zero-boundary stencil, the histogram, and the 16-term weighted sum. NumPy/Python coefficient construction, scalar boxing, input/storage preparation and allocation remain explicit parts of the public call. Special input types retain NumPy/SciPy count providers. The unchanged exact-rational `2*gamma_32*S` checker runs outside timing. The timed call includes all preparation and allocation.

The genuine [Release run 40427614006](https://github.com/bipxx/scikit-imagexx/actions/runs/40427614006) used source [`06953fe3ba4e426372cfbf448b696e73de6c42d6`](https://github.com/bipxx/scikit-imagexx/commit/06953fe3ba4e426372cfbf448b696e73de6c42d6). Its producer passed **3,310 focused checks** across 14 normal/scalar pairs, **3,696 color regression checks**, **1,612 matching checks**, and **1,385 installed-registry smoke checks** for 231 registered functions. A fresh measurement job installed the exact offline wheels. Complete source, package, sole core and SDK identities matched before and after measurement. The twelve older timing commands were skipped by `latest_only`; their performance was not remeasured here.

Runtime: Linux x86_64, CPython 3.11.17, NumPy 1.24.4, SciPy 1.15.3 and scikit-image 0.26.0. The artifact contains 14 wheels: the product plus 13 dependencies. The raw report records only `x86_64` as its CPU identity; no specific CPU model was captured. Source registration is partial native coverage. **Full supported-profile qualification remains 203 image APIs.**

Provenance:

- Raw JSON SHA-256: `23cb8dab7041df8e905392d43bbe9a9ea2ff901cefd59c16b77bf23885801371`.
- Retained artifact `12700273626`, 84,555,720 bytes, SHA-256 `05271a94a86659d3e7350e3a0ec78a1270c4657fc75c7dd67c1639cc76abcf4c`. All 51 members passed CRC/SHA verification; the large archive remains outside Git.
- Product wheel SHA-256: `3052bde93554d4ce80526b7757d53c4c2b1a4809f2d6d12a04d1f3e93a96882a`; sole core SHA-256: `417f401d07179b1aee1f2410fb70447f07441e3bc41e0634da730492957a7c44`.
- Review receipt: `build/phase3-perimeter-crofton-first-release-v1/final-review.json`, SHA-256 `e9059f8ed5f81b1832aa821406db6baf00a272d336af5d90f3d88f7046ee5e90`.
