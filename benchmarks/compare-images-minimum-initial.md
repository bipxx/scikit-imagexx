# Compare images: first minimum-profile Release comparison

**Six of 18 cases are faster; 12 are slower than scikit-image 0.26.0.** The equal-case geometric mean is **0.806945489×**, so this initial implementation is slower overall. All six checkerboard cases win; every diff and blend case loses. The [byte-exact JSON](compare-images-minimum-initial.json) retains all **252 raw latency samples**, including every loss.

| Method | Cases faster | Geometric mean original/current |
|---|---:|---:|
| Diff | 0/6 | 0.535496987× |
| Blend | 0/6 | 0.453422428× |
| Checkerboard | 6/6 | 2.164076280× |

Inputs are independent C-order square images generated with seeds 2026 and 2027 in uint8, float32 and float64. Floating inputs are normalized by 255 and cast before timing. Calls use the listed method and default `(8, 8)` tiles.

| Side | Input dtype | Method | Original ms | Current ms | Original/current | Current − original ms | Max error |
|---:|---|---|---:|---:|---:|---:|---:|
| 256 | uint8 | diff | 0.115167187 | 0.186587164 | 0.617229955× | +0.071419977 | 0.0 |
| 256 | uint8 | blend | 0.109485629 | 0.165519451 | 0.661466842× | +0.056033822 | 0.0 |
| 256 | uint8 | checkerboard | 0.287976205 | 0.174556965 | 1.649754882× | -0.113419240 | 0.0 |
| 256 | float32 | diff | 0.024794131 | 0.051395426 | 0.482419011× | +0.026601296 | 0.0 |
| 256 | float32 | blend | 0.025780618 | 0.067096167 | 0.384233849× | +0.041315548 | 0.0 |
| 256 | float32 | checkerboard | 0.198082583 | 0.070074849 | 2.826728661× | -0.128007735 | 0.0 |
| 256 | float64 | diff | 0.043395768 | 0.090185809 | 0.481181786× | +0.046790040 | 0.0 |
| 256 | float64 | blend | 0.039501457 | 0.090473664 | 0.436607247× | +0.050972206 | 0.0 |
| 256 | float64 | checkerboard | 0.212460695 | 0.105877987 | 2.006655978× | -0.106582708 | 0.0 |
| 1,024 | uint8 | diff | 4.416700000 | 6.550550500 | 0.674248676× | +2.133850500 | 0.0 |
| 1,024 | uint8 | blend | 1.693911625 | 3.011061500 | 0.562562945× | +1.317149875 | 0.0 |
| 1,024 | uint8 | checkerboard | 5.187767667 | 3.348953600 | 1.549071228× | -1.838814067 | 0.0 |
| 1,024 | float32 | diff | 0.326082059 | 0.687620947 | 0.474217750× | +0.361538889 | 0.0 |
| 1,024 | float32 | blend | 0.301526262 | 0.693875952 | 0.434553555× | +0.392349690 | 0.0 |
| 1,024 | float32 | checkerboard | 2.422859600 | 0.738313778 | 3.281612335× | -1.684545822 | 0.0 |
| 1,024 | float64 | diff | 0.828955350 | 1.610524909 | 0.514711288× | +0.781569559 | 0.0 |
| 1,024 | float64 | blend | 0.525303750 | 1.639840400 | 0.320338339× | +1.114536650 | 0.0 |
| 1,024 | float64 | checkerboard | 3.696027500 | 1.711728714 | 2.159236723× | -1.984298786 | 0.0 |

The largest absolute loss is 1024² uint8 diff: **+2.133850500 ms** (0.674248676×). The lowest ratio is 1024² float64 blend: **0.320338339×**, adding **1.114536650 ms**. These measurements do not attribute the losses to a particular stage; diff and blend still require optimization.

Timings cover the entire public call: decorator, shape checks, literal image conversion, method/tile validation, storage preparation, owned arithmetic, allocation and output destruction. Literal `img_as_float` remains an explicit primitive provider. Exceptional arithmetic, non-C layouts, dtype metadata, custom protocols and NumPy-1.24 scalar-float32 blend promotion are outside this timing cohort.

Before timing, the authenticated harness compares complete converted-input metadata and raw words, verifies one accepted owned C++ call per case, and checks exact public ndarray metadata and bytes. Passive callbacks are restored before measurement. The JSON exports converted-input and output SHA-256 fingerprints; full image payloads are not exported. All 18 cases pass these exact checks with zero observed error.

The fresh Linux x86_64 host used one global worker, all four reference pool limits set to one, and seven paired rounds per implementation: Python 3.11.17, NumPy 1.24.4 and SciPy 1.15.3. The CPU field records only `x86_64`; the processor model is unknown. This is a first original/current comparison, with no prior-C++ or cumulative cross-host gain claim. The last complete supported-profile qualification remains **203 image APIs**.

Validation passed: 21 focused normal/scalar XML pairs totaling **4,710 checks**, **3,696 color checks**, the entire **1,612-check matching file**, and **1,417 smoke checks for 239 registered APIs**. Separate controls passed: four Blur, four Polygon, four Euler and nine integer-width checks. The bounded private Compare component passed 436 public and 136 shared checks across both pinned profiles and normal/scalar modes, with frozen expectations unchanged.

[Release run 40649386523](https://github.com/bipxx/scikit-imagexx/actions/runs/40649386523) measured source [`317658444d7de04c751fe92d912c60853054c0e6`](https://github.com/bipxx/scikit-imagexx/commit/317658444d7de04c751fe92d912c60853054c0e6). Review authenticated all **1,662 committed source blobs**, all **68 artifact members**, **14 wheels including 13 runtime dependencies**, the complete installed package/core/SDK, and identical before/after measurement identities. Only this new 18-case timing ran; **19 earlier timing steps were skipped**.

Raw JSON SHA-256: `7a1fcf4287215b700c785cf11a1b81f7312635bfbdeb3d6448d254daa58ac355`. The sole retained performance artifact is `12798648924.zip` (84,730,999 bytes), SHA-256 `0179be5c3b7b3232b5c005bbfc4ded321a11f34e64207913b0a6382ea684265e`; its archive remains in ignored local evidence. Review SHA-256: `33f6ff4c548a7edf20b3eede37c538e1e4d60dd1369b83b2235ab26ad5cf4422`.
