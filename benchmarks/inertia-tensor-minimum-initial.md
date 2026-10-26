# Inertia tensor: first minimum-Linux result

[Release run 40378736391](https://github.com/bipxx/scikit-imagexx/actions/runs/40378736391) measures exact source `7456bfd5c03c32dcb617ff05f8ef1b8d9b21e3aa`, merged as `37b296e`. **Three of six cases beat scikit-image; the geometric mean original/current ratio is 0.627507734x.** All three 256-square cases win and all three 1024-square cases lose. Every result and all **84 raw latency samples** remain in the [exact raw JSON](inertia-tensor-minimum-initial.json).

## All six cases

| Image size | Input dtype | Original ms | Current ms | Original/current | Added ms |
|---|---|---:|---:|---:|---:|
| 256x256 | uint8 | 0.316847 | 0.255819 | 1.238559992x | -0.061028 |
| 256x256 | float32 | 0.283966 | 0.193371 | 1.468506834x | -0.090596 |
| 256x256 | float64 | 0.294280 | 0.240461 | 1.223819511x | -0.053820 |
| 1024x1024 | uint8 | 1.609296 | 6.988372 | 0.230281890x | +5.379077 |
| 1024x1024 | float32 | 1.249462 | 2.204941 | 0.566664623x | +0.955479 |
| 1024x1024 | float64 | 1.416546 | 6.739264 | 0.210192985x | +5.322718 |

## Scope and validation

These are automatic image calls to `measure.inertia_tensor(image)` using the existing seeded image fixtures, one global worker and seven warmed paired rounds. Each timed call includes central moments, tensor preparation and allocation. The genuine original central moments used by the frozen cell-specific checker are separately prepared outside timing. Both implementations run in the same fresh offline process with reference pool caps, and source/wheel/runtime/package/SDK identities match before and after measurement. The platform is x86_64; a detailed CPU model was not captured. This first original-library comparison is not a measured gain over the previous C++ version.

Owned C++ supplies typed per-axis projections and finite float32/float64 tensor arithmetic. NumPy coordinate powers, analytical preparation, diagonal sum/indexing and exceptional/integer/complex tensor stages remain explicit providers. The tiny tensor assembly is serial; image projections use the existing SIMD and optional process-global worker budget.

The genuine installed product passes all eight public contract groups in normal and forced-scalar modes, including 115 inertia tests per mode, plus 1,361 smoke checks. Separate component validation passes 1,272 checks and 88 shared checks across both runtime profiles and dispatch modes; saved old moments records/arrays remain exact. The initial failed layout admission and its cancelled hosted run remain preserved; the correction removes six redundant NumPy-stage storage checks without changing tests, bounds or arithmetic.

All three large-image losses require follow-up. A concrete candidate is the missing K=3 fused projection: the existing SIMD fusion handles K=2/K=4, while order-2 inertia moments use K=3. This is a hypothesis to test with unchanged reduction order and a fixed six-case controlled comparison, not an attributed cause or a promised speedup.

Raw JSON SHA-256: `16f976c72d7895ad9724bd2ba36acb0841f6934d81a1f6a02059e0054be56045`. The retained parent artifact is `12677104157` with SHA-256 `e1ad53ca1cb162393a547fed7ef9ef8b6ab9876ba2f97951db4c0cc7afcc281e`. Source coverage is 222 native-partial image APIs plus three metadata functions; full supported-profile qualification stays 203 image APIs.

The subsequent [controlled K=3 comparison](inertia-k3-same-host.md) keeps all first-run losses visible and measures the fix separately: 2.415797494x direct gain, with all six candidate cases beating the original in both passes on that host.
