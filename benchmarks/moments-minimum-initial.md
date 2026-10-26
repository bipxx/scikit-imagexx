# Moments: first minimum-Linux Release result

[Run 40333236658](https://github.com/bipxx/scikit-imagexx/actions/runs/40333236658) measures exact source `07a3074bf0e5194d6e96a25ebe4aad4998b04706` against original scikit-image 0.26.0. **12 of 18 cases are faster**, with a **1.319698648× geometric mean speedup**. All six slower cases are retained: 1024² uint8 and float64 inputs for each API. This is the first original/current comparison, not a gain over an earlier C++ implementation.

The [exact raw report](moments-minimum-initial.json) retains all **252 latency samples** from seven paired rounds per case. Raw and automatic central moments use default order3 and unit spacing; centroid uses its order1 moments route. Calls use one worker and include allocation, dtype conversion, coordinate powers, projection and result construction.

| API | Wins / cases | Geometric mean original/current |
|---|---:|---:|
| `measure.moments` | 4/6 | 1.350012476× |
| `measure.moments_central` | 4/6 | 1.334406078× |
| `measure.centroid` | 4/6 | 1.275846780× |

| API | Shape | Input dtype | Original ms | Current ms | Gain | Current − original ms | Max absolute output error |
|---|---|---|---:|---:|---:|---:|---:|
| `moments` | 256² | uint8 | 0.293006238 | 0.163713118 | 1.789754189× | -0.129293120 | 16384 |
| `moments` | 256² | float32 | 0.235431758 | 0.113196106 | 2.079857383× | -0.122235651 | 1048576 |
| `moments` | 256² | float64 | 0.258032615 | 0.143124118 | 1.802859082× | -0.114908498 | 0.001953125 |
| `moments` | 1024² | uint8 | 1.583662833 | 2.032634500 | 0.779118348× | +0.448971667 | 1.07374182e+09 |
| `moments` | 1024² | float32 | 1.052400400 | 0.703873091 | 1.495156462× | -0.348527309 | 1.57625987e+16 |
| `moments` | 1024² | float64 | 1.294334667 | 1.671470400 | 0.774368883× | +0.377135733 | 12582912 |
| `moments_central` | 256² | uint8 | 0.303488577 | 0.178525176 | 1.699976345× | -0.124963400 | 0 |
| `moments_central` | 256² | float32 | 0.256355107 | 0.130092723 | 1.970556849× | -0.126262384 | 38048 |
| `moments_central` | 256² | float64 | 0.293670360 | 0.166670645 | 1.761980100× | -0.126999715 | 7.62939453e-05 |
| `moments_central` | 1024² | uint8 | 1.560184000 | 1.797894000 | 0.867784196× | +0.237710000 | 16 |
| `moments_central` | 1024² | float32 | 1.058554900 | 0.755513250 | 1.401106996× | -0.303041650 | 161844224 |
| `moments_central` | 1024² | float64 | 1.339030500 | 1.702073400 | 0.786705497× | +0.363042900 | 0.421875 |
| `centroid` | 256² | uint8 | 0.277707750 | 0.146867050 | 1.890878519× | -0.130840700 | 0 |
| `centroid` | 256² | float32 | 0.244738786 | 0.113418227 | 2.157843511× | -0.131320558 | 0 |
| `centroid` | 256² | float64 | 0.266365571 | 0.137158892 | 1.942021897× | -0.129206680 | 1.42108547e-14 |
| `centroid` | 1024² | uint8 | 1.528249500 | 2.581011200 | 0.592112696× | +1.052761700 | 0 |
| `centroid` | 1024² | float32 | 1.087773300 | 0.655491308 | 1.659477841× | -0.432281992 | 0.000335693359 |
| `centroid` | 1024² | float64 | 1.283488714 | 2.316938200 | 0.553958977× | +1.033449486 | 5.1159077e-13 |

Values are medians of warmed allocation-inclusive public calls. Gain is original/current; a positive penultimate column is added latency. The largest losses are centroid uint8 (+1.052761700 ms) and centroid float64 (+1.033449486 ms). All nine 256² cases and all three 1024² float32 cases are faster; the six 1024² uint8/float64 losses remain unresolved.

## Correctness and measurement boundary

The genuine installed minimum-Linux Release product passes **203 moments tests in normal mode and 203 with SIMD disabled**, plus **179 window, 161 LPI and 175 Butterworth tests in each mode**, and **1,350 registry smoke checks**. All older regression steps in the minimum producer also pass. The moments file preserves its frozen 100 specifications, including documented errors/refusals; passing those controls does not give unsupported routes numerical parity credit.

Before timing, the three moments APIs use the existing input/reference-derived per-cell checker from the pre-output policy. It verifies shape/dtype, finite and nonfinite structure, finite allowance values and source-initialized zero slots. High-order unnormalized moments have large numerical scale, so the absolute errors above are interpreted against their own weighted moment bounds; they are not zero-error claims or a new global scalar tolerance. Every old API retains its previous comparison path. No fixture, formula factor or numerical policy was changed for measurement.

A fresh measurement host installs the exact archived wheelhouse offline after correctness finishes, with no builds or pytest runs in the measurement job. The native getter is asserted to be one after setting workers, after parity and after timing. Reference thread pools are capped before imports. Normal AVX2 dispatch is recorded; SIMD-disabled mode has correctness coverage here, not a separate speed result. The moments path makes no SciPy FFT-worker claim. Source, wheel, installed package, runtime libraries and SDK identities match before and after measurement.

Runtime: **CPython 3.11.17, NumPy 1.24.4, SciPy 1.15.3, scikit-image 0.26.0**, Linux x86_64 (`Linux-6.17.0-1022-azure-x86_64-with-glibc2.39`). The report records architecture, not a specific CPU model. These are observations on one minimum-runtime host, not cross-CPU or full supported-profile guarantees.

## Native scope and next bottleneck

C++ owns typed float32/float64/complex64/complex128 per-axis projections and their floating-point status capture. NumPy provides dtype promotion, coordinate powers and layout preparation; owned Python/NumPy formulas provide automatic central moments and centroid division. The functions remain native-partial. High signed-mass conditioning, arbitrary protocols, external buffer-root storage variants, extended arithmetic and print/log FP policies remain open.

The six measured losses share float64 projection arithmetic: uint8 promotes to float64. A concrete next experiment is to improve locality in the existing real K2/K4 fused projector. Its double path handles four adjacent columns before traversing every row; processing two adjacent vectors together could consume a full adjacent input span while preserving each output’s ascending reduction order. This is a source-supported hypothesis, not evidence that cache behavior caused these timings. It needs a controlled comparison and unchanged parity gates before acceptance.

Source coverage is **219 native-partial image functions plus three metadata functions**, with five partial classes and 249 upstream-only entries. The separately accepted combined supported-profile baseline remains **203 image APIs**. This Release result does not promote that qualification. Earlier [window](window-minimum-initial.md), [LPI](lpi-minimum-initial.md), [Butterworth](butterworth-minimum-initial.md) and [73-case legacy comparison](phase3-butterworth-legacy-same-host.md) results and their losses remain unchanged. Fresh secondary window/LPI/Butterworth reports are retained in this artifact, without replacing their first reports. This run did not repeat the fixed73 comparison.

## Exact identities and retained evidence

| Identity | Value |
|---|---|
| Source revision | `07a3074bf0e5194d6e96a25ebe4aad4998b04706` |
| Product-source fingerprint | `472bd63a02b8697dfa2552248bcba9493c5d22f414f00df6c7026a7f275b7b88` |
| Wheel | `scikit_imagexx-0.1.0-cp311-cp311-linux_x86_64.whl` |
| Wheel SHA-256 | `0fe77c622854cc26e3fa93bae710d1ff229dc2f083e40e5bf218606883df2c08` |
| Sole core SHA-256 | `7399d79520729944a882b4332b6679ac7a5dc044040a75d48a21adae595e1e43` |
| Pre/post measurement manifest SHA-256 | `bc4c8ca7c9504eb6c9a16466b86873c9cad20ad2dca95b8349985834980f4c16` |
| Raw moments JSON SHA-256 | `d99c3e6e64fac5e1cbace977b09a4c649bfcb5f0533cbe7e08cf20d5061d7343` |

The published JSON is an exact **18,896-byte** copy. The single retained performance artifact is `12655563212`, **84,368,175 bytes**, SHA-256 `1e6b6b09e4d5c388ec4400f41d284c03d907f799896654681aaaf88aba5ecf8a`. All 35 members pass CRC/size/SHA checks; 14 locked wheels, the complete installed package and SDK, 333 committed source blobs and both identity receipts authenticate. The artifact already includes the producer wheels, so no duplicate producer download was needed.

Local evidence is under `build/phase3-moments-first-release-v1/`. Final review SHA-256: `190730aa0d405859c668d48bcc1ced1be457d818f439ef2e890d4b29d299a572`; handoff SHA-256: `eb8eebced72084e45af29fcd8de0662a741be3c7c514a1ec1ba829d9853b3f1c`. Original logs, XML and all raw samples remain retained. Later documentation changes are not attributed to this producer; its exact measured source is 07a3074.
