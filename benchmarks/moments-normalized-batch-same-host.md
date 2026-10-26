# Normalized moments batch: same-host comparison

[ABBA run 40363464486](https://github.com/bipxx/scikit-imagexx/actions/runs/40363464486) compares baseline `76cdd70ac85c889a68985c25964a1e0646fa7579` with candidate `aecf5218563b09981bf329b6975e4e43cbf4334a` on one **AMD EPYC 7763 64-Core Processor** host. The fixed five-case geometric mean gains **4.478043258× directly** and **4.649099324× after reference normalization**. Both normalization targets improve in both candidate passes relative to both baseline passes. The change was kept after the complete review.

All **three Hu direct losses** remain visible below, including float32's loss in both metrics. The [exact compact artifact](moments-normalized-batch-same-host.zip) retains four complete reports and all **280 raw latency samples**. The [first original/current report](moments-derived-minimum-initial.md) and its [raw JSON](moments-derived-minimum-initial.json) remain unchanged.

## Balanced method and all five cases

Every actual input is an owning **C-contiguous 4×4 moment tensor** prepared outside timing. The dtype is the actual tensor dtype. Default normalization uses order 3, unit spacing, positive mass 4 and zero first-order central slots. Public eligibility checks, allocation, conversion and output construction remain inside each timed call. Integer normalization retains its source-error contract and has no numerical timing row.

The order is baseline B1, candidate C2, candidate C3, baseline B4, with a fresh Python process for each pass. Each value starts from the median of seven warmed paired samples. Direct gain is `GM(B1 current, B4 current) / GM(C2 current, C3 current)`; normalized gain is `GM(C2 original/current, C3 original/current) / GM(B1 original/current, B4 original/current)`. Aggregate gains are geometric means over the same five cases. Time columns are two-pass geometric means in microseconds; positive added time is a direct regression.

| API | Actual input dtype | Baseline µs | Candidate µs | Direct gain | Normalized gain | Added µs |
|---|---|---:|---:|---:|---:|---:|
| `moments_hu` | uint8 | 2.936930 | 2.941099 | 0.998582292× | 1.123095912× | +0.004170 |
| `moments_normalized` | float32 | 119.367795 | 2.318579 | 51.483161594× | 53.099057243× | -117.049216 |
| `moments_hu` | float32 | 2.015491 | 2.236655 | 0.901118257× | 0.914759937× | +0.221164 |
| `moments_normalized` | float64 | 89.816273 | 2.303745 | 38.987071152× | 39.224493856× | -87.512528 |
| `moments_hu` | float64 | 2.268773 | 2.275627 | 0.996988052× | 1.015023231× | +0.006854 |

| Group | Direct geometric mean | Normalized geometric mean |
|---|---:|---:|
| Both normalization targets | 44.801536628× | 45.637524523× |
| All three Hu cases | 0.964462177× | 1.014067011× |
| Fixed five cases | 4.478043258× | 4.649099324× |

Normalization float32 improves **51.483161594× directly**, saving **117.049216 µs**; float64 improves **38.987071152×**, saving **87.512528 µs**. Every normalization candidate-pass median is lower than both corresponding baseline-pass medians.

Hu uint8 adds **0.004170 µs**, float32 adds **0.221164 µs**, and float64 adds **0.006854 µs**. Float32 is slower in both candidate passes than both baseline passes and loses in both direct (**0.901118257×**) and normalized (**0.914759937×**) metrics. The uint8 and float64 normalized gains do not erase their direct losses. No causal explanation for these regressions is asserted.

Balanced original/current scores are **0.758508394× baseline** and **3.526380861× candidate**. All **5/5 cases beat the original in both candidate passes**. These same-host figures establish the named product comparison; the candidate's earlier measurement on a different host is retained separately and is not used to claim this gain.

## Correctness, identities and scope

Both genuine minimum-Linux Release products passed **155 derived-moments, 203 moments, 179 window, 161 LPI, 175 Butterworth and 134 immutable-storage checks in each normal/scalar mode**, plus **1,357 installed-registry smoke checks** and all older minimum regressions. The baseline is from [CI 40350415600](https://github.com/bipxx/scikit-imagexx/actions/runs/40350415600); the candidate is from [CI 40361265234](https://github.com/bipxx/scikit-imagexx/actions/runs/40361265234).

The frozen 76 specifications, 155 public comparators, prepared five-case inputs, cell-specific error policy and timing arithmetic are unchanged. Production changes are confined to the declared normalization batch kernel/header, binder, existing SIMD leaf, frontend and CMake wiring. The only benchmark difference is the literal provider description: a unique-assignment normalization followed by whole-AST comparison authenticates that exception. All other production source entries are fixed. Hu's mathematical source is unchanged; additive native controls are verification only.

A fresh measurement job authenticates both exact CI artifacts and their 13 identical dependency wheels plus the respective product wheels, then installs both runtimes offline before any timer. No build or pytest job runs on the measurement host. All eight before/after source, wheel, installed package, sole-core, SDK and runtime identity checks pass. Native worker getters remain at one, original reference pools are capped at one thread, and fallback is disabled. The input/reference-derived per-cell checker completes before timing. Normalization reports zero output error; Hu float32 retains the same **4.76837158203125e-7** maximum error in all four passes within its frozen polynomial envelope. No fixture or tolerance was changed in response to results.

Runtime is **CPython 3.11.17, NumPy 1.24.4, SciPy 1.15.3 and scikit-image 0.26.0**, Linux x86_64. This is normal AVX2 public-call timing with separately passed SIMD-disabled correctness gates. It is not isolated kernel timing or a scalar performance result.

**The optimized batch admission is C-contiguous only.** F-contiguous tensors, including ordinary `moments_central` outputs, still take the previous scalar route. A separate realism check saved eleven real F-order moment tensors that miss this batch eligibility; extending that route is a separate pending experiment. This fixed five-case comparison makes no whole-pipeline gain claim. NumPy providers remain in the scalar normalization fallback and in Hu conversion/wrapping; both APIs remain native-partial.

One same-host result does not establish universal CPU, dtype, parameter, storage or supported-profile performance. Full combined supported-profile acceptance stays **203 image APIs**. No additional API qualification is inferred from this optimization or from later source merges.

## Exact evidence

| Identity | Baseline | Candidate |
|---|---|---|
| Source | `76cdd70ac85c889a68985c25964a1e0646fa7579` | `aecf5218563b09981bf329b6975e4e43cbf4334a` |
| Product-source fingerprint | `bfbffd906118fbae3bf3889d579dd0fe523440c8c97302c488586172955971cd` | `174f97ff7b514adb6dab320e198f1c76bd1e9e849701b6114d8a48aa0db6a029` |
| Product wheel SHA-256 | `da2274c9aee91eddff90f1891206b057614bd8fbd49760904349d276daa2a520` | `5c0167d77bc8814eaa74b342d093e8b0440f18f3b0bf98217401c01fa0f659d1` |
| Sole core SHA-256 | `d1d794ee256cf1c38d8d4974c6440fa119cf8f90e13108102667dc4f4f9210e0` | `df710abc491e2df9f3541839526711bbe61388d78d00646709c068639d5ddb0d` |
| CI artifact ID | `12664501098` | `12669471348` |
| CI archive SHA-256 | `ac5756ee20ed1792e49bb6d6750e71c0088b1f26f489f8812ec5fc0460783d57` | `46758c7b7efc92a154cb9ad24d616107c1d7eb4037a3041d197b2837e73b39ea` |

The published ZIP is the exact **26,146-byte** artifact `12668608721`, SHA-256 `ae4349f7d28a787ea355a400f73200b02e15338e0e554e67f5e6c97e5d51071a`. All 36 members pass size/CRC/SHA verification. Prepared wheels are not duplicated in this compact artifact; their retained authenticated receipts are reused. Workflow revision: `9c83de8efe34fcefc877b2b2b2865971be3f0053`.

Local review: `build/phase3-normalized-batch-same-host-v1/review-v2.json`, SHA-256 `f3ad40e21994d861ce0469e00cefd7c2779c20538ef06482a08770d3b067f7fe`. Final handoff SHA-256: `65192e70f1bf06f2e74fdd6b3dd68c6ede54e080fc8f4d0b687e09f1ea8ba9b5`. The prior review is retained; v2 only corrects an inherited case-count wording label, without changing any check, sample, result or decision input. No retry or additional measurement was used.
