# RGB2YUV storage: first minimum-Linux comparison

[Release 40408342309](https://github.com/bipxx/scikit-imagexx/actions/runs/40408342309) measures exact source `e9c723705d2e1bd14b605363032e1bbb36199b77`, merged as `ab7abc0`. **All 12 cases beat scikit-image 0.26.0**, with no losses: geometric mean **3.886110367802x** overall, **3.731809863862x** for the six new storage targets, and **4.046790790973x** for the six native controls. The [byte-exact raw JSON](rgb2yuv-storage-minimum-initial.json) retains all **168 latency samples** and reports zero maximum absolute output error for these inputs.

| Case | Group | Actual input dtype.str | Original ms | Current ms | Original/current | Added ms | Max abs error |
|---|---|---|---:|---:|---:|---:|---:|
| 256:native-u8 | control | `\|u1` | 1.394459250 | 0.412537947 | 3.380196316230x | -0.981921303 | 0 |
| 256:native-f32 | control | `<f4` | 0.505660520 | 0.058351822 | 8.665719464702x | -0.447308698 | 0 |
| 256:native-f64 | control | `<f8` | 0.502868481 | 0.114668770 | 4.385400491313x | -0.388199711 | 0 |
| 256:native-f16 | target | `<f2` | 0.904718333 | 0.125080859 | 7.233067788675x | -0.779637474 | 0 |
| 256:swapped-f32 | target | `>f4` | 0.641466727 | 0.102391841 | 6.264822701115x | -0.539074887 | 0 |
| 256:swapped-f64 | target | `>f8` | 0.617564750 | 0.237989125 | 2.594928444735x | -0.379575625 | 0 |
| 1024:native-u8 | control | `\|u1` | 13.803336000 | 8.669234000 | 1.592220950548x | -5.134102000 | 0 |
| 1024:native-f32 | control | `<f4` | 7.866739000 | 1.168664556 | 6.731391794680x | -6.698074444 | 0 |
| 1024:native-f64 | control | `<f8` | 8.136728000 | 2.550640750 | 3.190072141677x | -5.586087250 | 0 |
| 1024:native-f16 | target | `<f2` | 14.255576000 | 3.111583000 | 4.581454520095x | -11.143993000 | 0 |
| 1024:swapped-f32 | target | `>f4` | 10.251872000 | 3.052432250 | 3.358591169386x | -7.199439750 | 0 |
| 1024:swapped-f64 | target | `>f8` | 12.583465000 | 8.429473000 | 1.492793796243x | -4.153992000 | 0 |

Each image has shape 256x256x3 or 1024x1024x3. `<f2` is actual binary16 storage; it is not relabeled float32. `<` and `>` denote little- and big-endian storage, and `|u1` is byte-order independent. Half and float32 inputs produce native float32 outputs; uint8 and float64 inputs produce native float64 outputs. Added time is current minus original, so every negative value above is time saved.

Seven warmed paired rounds use one global worker in a fresh offline measurement process. Input generation is outside timing; each timed public call includes preparation, allocation, matrix arithmetic, applicable FP-status capture/dispatch and axis restoration. The exact existing color-transform allowance is unchanged: float32 uses `rtol=2e-6` and `atol=2e-6 + 4*eps32*max(sum(abs(yuv_from_rgb), axis=1))*finite_input_peak`; float64 uses `rtol=atol=2e-13`. Source input hashes and worker state are checked after measurement.

C++ owns byte-order normalization, half-to-float preparation and the existing SIMD/scalar matrix arithmetic. The new status path captures and restores each worker's floating environment, combines actual exception flags, and reports source `matmul` warnings/errors through the existing Python dispatcher. The old two-argument color binding and all other transform arithmetic remain intact. NumPy array/axis/metadata handling and Python error-policy dispatch remain explicit providers. The thirteen other linear-transform half routes, swapped-integer RGB2YUV, subclasses/extended types and print/log policies remain outside this new native storage scope and retain their existing provider/refusal behavior.

The genuine installed minimum product passes **12 public normal/scalar contract pairs (2,818 checks)**, the separate full 14-operation color regressions (**1,848 normal + 1,848 scalar = 3,696 checks**), **1,377 smoke checks** at **229 registered functions**, and the full **1,612-test histogram-matching regression**. Two-profile components separately pass all 7,540 storage/color checks. Native controls pass normal/scalar/sanitizer modes and separate portable controls. Earlier failure records remain retained; the frozen 18-case storage cohort and existing comparison budgets are unchanged. Full supported-profile qualification remains **203 image APIs**.

Source, installed package/core, SDK and before/after measurement identities match; fourteen prepared wheels include thirteen authenticated dependency wheels. This closes storage partitions of an existing API and adds no API count. The measured source combines Shannon, kept float32 traversal and RGB, on one x86_64 minimum-Linux host: CPython 3.11.17, NumPy 1.24.4 and SciPy 1.15.3. A detailed CPU model was not captured. These are first original-library comparisons, not a measured gain over the previous C++ version or a full API/all-profile performance claim.

Raw JSON SHA-256: `44e254712a4770e12cdd1a1c65d67e00179f9e2517d53998a8012902b8b9e695`. Parent artifact 12693171349 SHA-256: `ddedbe1f00708911e4a39048fb9011d74e0740f9112d6addda9cb6d25c3ebc05`.

Product wheel SHA-256: `20acf35ff8fcd42f0f43a0679f094de4bb73f2db03e136c55e631c8318c090da`. Installed core SHA-256: `861391664b7d75e63baee5749aa6464f6ba7eebb47dddc3cf36e4a5231afea02`.

RGB harness canonical-LF SHA-256: `a621318a573081b070cd99ccb707ceb8f311c1346f70834d5c060a2962afd37b`; this separately matches the committed harness because the normal product fingerprint excludes benchmark files. Product fingerprint: `202d15819e4fafc50879e1b4a1117d0a1e4102e07fa79f2ee3e9cde2adad7c2c`. Final review SHA-256: `9bda74d209291359854faeb522aca5efd1dad7571e9be029785936aad395b448`.
