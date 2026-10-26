# Moments float64 projection: same-host comparison

[ABBA run 40340933040](https://github.com/bipxx/scikit-imagexx/actions/runs/40340933040) compares baseline `07a3074bf0e5194d6e96a25ebe4aad4998b04706` with candidate `19464f23ab2c7d015b70da181c50143ebfd6e149` on one **AMD EPYC 9V45 96-Core Processor** host. Across the fixed 18 cases, the candidate gains **1.323055310× directly** and **1.338931833× after reference normalization**. The six 1024² uint8/float64 targets gain **1.742253016× directly** and **1.835711712× normalized**; both candidate medians beat both baseline medians for every target. Root kept the narrow source change after the complete review.

All **three direct losses** remain visible below; there are **zero normalized losses**. The [exact compact artifact](moments-f64-wide-same-host.zip) retains four complete reports and all **1,008 raw latency samples**. This measures the two named wheels. Later main combines the kept leaf with an independent immutable-storage guard, so its combined binary is not the exact measured candidate and receives no blanket timing claim here.

## Balanced method and all 18 cases

Each pass uses the median of seven warmed paired samples. Order is baseline B1, candidate C2, candidate C3, baseline B4, with a fresh Python process each time. For each case, direct gain is `GM(B1 current, B4 current) / GM(C2 current, C3 current)`; normalized gain is `GM(C2 original/current, C3 original/current) / GM(B1 original/current, B4 original/current)`. Overall values are geometric means of the 18 per-case gains. The two time columns below are the respective two-pass geometric means, in milliseconds; positive added time is a direct loss.

| API | Shape | Input dtype | Baseline ms | Candidate ms | Direct gain | Normalized gain | Added ms |
|---|---|---|---:|---:|---:|---:|---:|
| `moments` | 256² | uint8 | 0.055308860 | 0.047029902 | 1.176036048× | 1.150290911× | -0.008278958 |
| `moments` | 256² | float32 | 0.036094136 | 0.036624129 | 0.985528873× | 1.011382041× | +0.000529992 |
| `moments` | 256² | float64 | 0.050050461 | 0.041888451 | 1.194851092× | 1.180123741× | -0.008162010 |
| `moments` | 1024² | uint8 | 1.233042815 | 0.692788185 | 1.779826563× | 1.852336086× | -0.540254630 |
| `moments` | 1024² | float32 | 0.604350477 | 0.477975091 | 1.264397430× | 1.226356443× | -0.126375386 |
| `moments` | 1024² | float64 | 1.167330497 | 0.623898047 | 1.871027649× | 1.959440600× | -0.543432450 |
| `moments_central` | 256² | uint8 | 0.061593764 | 0.051790772 | 1.189280666× | 1.138185349× | -0.009802992 |
| `moments_central` | 256² | float32 | 0.042133997 | 0.042881243 | 0.982574067× | 1.017902150× | +0.000747246 |
| `moments_central` | 256² | float64 | 0.054596610 | 0.047168009 | 1.157492344× | 1.168841799× | -0.007428600 |
| `moments_central` | 1024² | uint8 | 1.168560809 | 0.687848797 | 1.698862909× | 1.709132019× | -0.480712012 |
| `moments_central` | 1024² | float32 | 0.621127231 | 0.484538730 | 1.281893876× | 1.205025241× | -0.136588501 |
| `moments_central` | 1024² | float64 | 1.168667973 | 0.689125242 | 1.695871668× | 1.850457342× | -0.479542731 |
| `centroid` | 256² | uint8 | 0.056329366 | 0.049915343 | 1.128498045× | 1.136999142× | -0.006414024 |
| `centroid` | 256² | float32 | 0.037259321 | 0.038090022 | 0.978191106× | 1.002083712× | +0.000830701 |
| `centroid` | 256² | float64 | 0.051328161 | 0.044796836 | 1.145798797× | 1.170316523× | -0.006531325 |
| `centroid` | 1024² | uint8 | 1.398454584 | 0.899668241 | 1.554411414× | 1.613882840× | -0.498786342 |
| `centroid` | 1024² | float32 | 0.789170194 | 0.549255146 | 1.436800728× | 1.364331692× | -0.239915048 |
| `centroid` | 1024² | float64 | 1.501306652 | 0.800529323 | 1.875392454× | 2.065609824× | -0.700777329 |

The direct losses are the three 256² float32 cases: raw moments add **0.000529992 ms** (0.985528873×), central moments add **0.000747246 ms** (0.982574067×), and centroid adds **0.000830701 ms** (0.978191106×). Their normalized ratios exceed one, but that does not erase their direct losses. No timing cause is inferred for these small regressions.

| API | Direct geometric mean | Normalized geometric mean |
|---|---:|---:|
| `measure.moments` | 1.341603587× | 1.352134353× |
| `measure.moments_central` | 1.307460356× | 1.314581748× |
| `measure.centroid` | 1.320326258× | 1.350417186× |

Balanced original/current scores are **0.768897058× baseline** and **1.029500748× candidate**. Only **9/18 cases beat the original in both candidate passes** on this host. The candidate's separate CI measurement reported 18/18 wins on another host; those cross-host figures are retained separately and do not establish the baseline-to-candidate gain. The [first original/current moments report](moments-minimum-initial.md) remains byte-exact and is not replaced.

## Correctness, provenance and scope

Both exact products passed 203 moments, 179 window, 161 LPI and 175 Butterworth installed checks in each normal/scalar mode, plus 1,350 registry smoke checks and the older minimum regressions. The baseline comes from [CI 40333236658](https://github.com/bipxx/scikit-imagexx/actions/runs/40333236658); the candidate comes from [CI 40338790308](https://github.com/bipxx/scikit-imagexx/actions/runs/40338790308). The only production-source difference is `src/simd_moments_avx2.cpp`; additive native controls are the only other changed source file. The frontend, cases, cell-specific bounds, benchmark, allocation behavior, reduction order, worker policy and provider stages are unchanged.

A fresh measurement job authenticated both successful CI artifacts, verified the closed source differences and identical 13 dependency wheels plus the respective product wheels, and installed both runtimes offline before any timer. There are no builds or pytest runs on the timing host. Original reference pools are capped at one thread. All eight before/after source/package/SDK/runtime identity checks pass. Every pass runs the fixed 18 cases with fallback disabled and one native worker, asserted before and after parity and after measurement. The source-defined per-cell checker completes before timing; all error summaries remain in the raw reports. This is normal AVX2 timing, with separate scalar correctness coverage rather than a scalar speed claim.

Runtime is CPython 3.11.17, NumPy 1.24.4, SciPy 1.15.3 and scikit-image 0.26.0 on Linux x86_64. CPU/platform records are included in the artifact. These are public allocation-inclusive calls with unchanged NumPy dtype/coordinate-power/layout stages and analytical central-moment/centroid-ratio providers, not isolated kernel timing. There is no SciPy FFT-worker claim for moments. One same-host result does not establish a universal CPU or supported-profile result.

Source coverage remains 219 native-partial image APIs plus 3 metadata functions, five partial classes and 249 upstream-only entries. Full supported-profile combined acceptance stays 203 image APIs. The accepted source optimization does not close the remaining moments parameter/storage/policy frontier or promote the later combined main product to whole-profile acceptance.

## Exact evidence

| Identity | Baseline | Candidate |
|---|---|
| Source | `07a3074bf0e5194d6e96a25ebe4aad4998b04706` | `19464f23ab2c7d015b70da181c50143ebfd6e149` |
| Product-source fingerprint | `472bd63a02b8697dfa2552248bcba9493c5d22f414f00df6c7026a7f275b7b88` | `5b59e6403ea4d990a888bc2258c618fcf6fadb68c92d05c04422cdc2c5c4950e` |
| Wheel SHA-256 | `0fe77c622854cc26e3fa93bae710d1ff229dc2f083e40e5bf218606883df2c08` | `d86ba5dc14efc0bc8619a779b28c86ba7913f920562b7832cd3612cf61ba85a4` |
| Sole core SHA-256 | `7399d79520729944a882b4332b6679ac7a5dc044040a75d48a21adae595e1e43` | `a764f26b1d88944a28a22e4aaa3a52fcb051b393eb050b0b8d38cbce23fc38a5` |
| CI artifact ID | `12655563212` | `12658548422` |
| CI archive SHA-256 | `1e6b6b09e4d5c388ec4400f41d284c03d907f799896654681aaaf88aba5ecf8a` | `1a12444e196a0c2643406906b69340269648b2f7ab8b78b2826dc5739ec36c11` |

The published ZIP is the exact **32,992-byte** artifact `12660346674`, SHA-256 `19b4e9bd814ac1533220dcae04aa82433a1a76d9d53e88fb446d1b737f3491e8`. All 36 members pass size/CRC/SHA verification. No prepared wheels are duplicated in this compact publication; their previously retained authenticated receipts are reused. Workflow revision: `3dde60680d260682826049bad03f2bcbfbc6223f`.

Local review: `build/phase3-moments-f64-wide-same-host-v1/review.json`, SHA-256 `5c2df4572fb7e2b4688bd2f816131080c4dd59df97bb16333beda9c607236c55`. Handoff SHA-256: `0aed394ec3e6d77433ae3b41e3766055daf2cca0a3dd0f081677566418c5c6ec`. The frozen aggregation contract and original observer/retention records remain beside it. All 18 cases, six targets, three direct losses and 1,008 samples are retained; no retry or extra experiment was used.
