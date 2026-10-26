# Derived moments: first minimum-Linux Release result

[Run 40350415600](https://github.com/bipxx/scikit-imagexx/actions/runs/40350415600) compares exact source `76cdd70ac85c889a68985c25964a1e0646fa7579` with original scikit-image 0.26.0. **All three Hu cases are faster; both normalized-moment cases are slower.** The five-case geometric mean original/current speedup is **0.737598829×**. This is a first original/current comparison, with no claim of improvement over an earlier C++ product.

The [exact raw report](moments-derived-minimum-initial.json) preserves all **70 latency samples**: seven paired rounds for each implementation across five cases. Every actual input is an owning **4×4 moment tensor** prepared before timing. Normalization uses default order 3 and unit spacing, positive mass 4 and zero first-order central slots. Preparation is excluded equally from both implementations; public allocation, dtype conversion, scalar operations and output construction remain inside each timed call. The dtype labels below describe the actual moment argument. The two requested image-fixture sizes do not create duplicate fixed-size timing rows.

| API | Wins / cases | Geometric mean original/current |
|---|---:|---:|
| `measure.moments_hu` | 3/3 | 1.232841204× |
| `measure.moments_normalized` | 0/2 | 0.341342083× |

| API | Input shape | Input dtype | Original µs | Current µs | Original/current | Current − original µs | Max absolute output error |
|---|---|---|---:|---:|---:|---:|---:|
| `moments_hu` | 4×4 | uint8 | 2.891651 | 2.646822 | 1.092499165× | -0.244829 | 0 |
| `moments_hu` | 4×4 | float32 | 2.279342 | 1.704032 | 1.337616899× | -0.575310 | 4.76837158e-07 |
| `moments_hu` | 4×4 | float64 | 2.339103 | 1.824235 | 1.282237760× | -0.514868 | 0 |
| `moments_normalized` | 4×4 | float32 | 46.051637 | 114.865824 | 0.400916791× | +68.814186 | 0 |
| `moments_normalized` | 4×4 | float64 | 24.551500 | 84.479747 | 0.290619950× | +59.928247 | 0 |

Values are medians of warmed allocation-inclusive calls. Speedup below 1 is slower. Normalization adds **68.814186 µs for float32** and **59.928247 µs for float64**; these losses remain unresolved. Integer normalization preserves the original NaN-assignment error and has no numerical or timing credit.

## Correctness and measurement scope

The genuine installed minimum-Linux Release wheel passes **155 derived-moments, 203 moments, 179 window, 161 LPI, 175 Butterworth and 134 immutable-storage checks in each of normal and SIMD-disabled modes**, plus **1,357 installed-registry smoke checks**. Every older regression step in the minimum producer also succeeds. The derived family retains its frozen 76 specifications, including source errors and explicit unsupported frontiers; the pass count is not a full-domain numerical claim.

Each of the five cases passes the frozen input/reference-derived cell-specific checker before timing. It checks shape/dtype, finite and nonfinite structure, required normalized low-order NaNs and the declared per-cell allowances. Hu float32 has a nonzero maximum output difference shown above; it satisfies the pre-output polynomial envelope. No tolerance or fixture was changed in response to these timings.

A fresh measurement host installs the archived 14-wheel runtime offline after the producer succeeds. It runs no build or pytest job alongside measurement. Native worker getters pass at 1 before and after timing; reference pools are capped at one thread. SIMD-disabled execution has correctness coverage here, not a separate speed result. Source, installed package, sole core, runtime and SDK identities match before and after measurement.

Runtime: **CPython 3.11.17, NumPy 1.24.4, SciPy 1.15.3 and scikit-image 0.26.0**, Linux x86_64. The retained metadata does not identify a specific CPU model. These five observations do not establish cross-CPU or full supported-profile performance.

## Native and provider boundary

For normalization, C++ owns the final scalar division and its floating-point status capture; NumPy scalar powers, the first division, promotion and output assignment remain providers. The current per-cell Python/native crossings remain a concrete optimization target, but this timing result does not isolate their causal contribution. Hu uses owned C++ polynomial arithmetic with source-ordered operations; NumPy dtype conversion and output wrapping remain part of the public call. Both APIs remain native-partial, with no whole-function upstream forwarding counted as native execution.

The separately accepted combined supported-profile baseline remains **203 image APIs**. This minimum-runtime Release result does not promote that qualification or close unassessed argument, dtype, protocol and storage frontiers. Earlier benchmark reports and their losses remain unchanged.

## Exact identities

| Identity | Value |
|---|---|
| Source revision | `76cdd70ac85c889a68985c25964a1e0646fa7579` |
| Run / artifact | `40350415600` / `12664501098` |
| Retained artifact | 84,390,034 bytes; 41 members; one download |
| Artifact SHA-256 | `ac5756ee20ed1792e49bb6d6750e71c0088b1f26f489f8812ec5fc0460783d57` |
| Raw JSON SHA-256 | `611bcd93d30a271a8ef0a59ef9f6c8afea70b3aff639acadaad6d75d8b4d57cf` |
| Product-source fingerprint | `bfbffd906118fbae3bf3889d579dd0fe523440c8c97302c488586172955971cd` |
| Product wheel SHA-256 | `da2274c9aee91eddff90f1891206b057614bd8fbd49760904349d276daa2a520` |
| Sole core SHA-256 | `d1d794ee256cf1c38d8d4974c6440fa119cf8f90e13108102667dc4f4f9210e0` |
| Pre/post measurement manifest SHA-256 | `5ccc508f191365c02c00f4ba084a465393791c1fbe4c61972c9eebfc7925ca95` |

All 41 ZIP members pass CRC/SHA checks; the 14 archived wheels, complete wheel package, 71 SDK files and 341 committed source blobs authenticate. Local evidence remains under `build/phase3-moments-derived-first-release-v1/`: final review SHA-256 `fe5865e2a64f908f33d939e0b027807460c1783671a8b5519d016844ff9ac2ad`, handoff SHA-256 `e9b34d32b6670c17f6f9268a398129c8e9854a47edca1a7ac916ef48a4eed11b`.

The first local log-fetch command rejected an unsupported gh option after the artifact had already downloaded and authenticated. That metadata failure is preserved; plain GET log capture then succeeded without another artifact download or any CI retry.
