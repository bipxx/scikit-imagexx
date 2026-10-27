# Perimeter: first minimum-profile comparison

The six fixed `measure.perimeter` cases were faster than scikit-image 0.26.0, with a geometric mean of **12.545133822154×** and **zero observed error**. All 84 raw latencies (seven paired rounds, two implementations, six cases) are preserved in the [unchanged JSON report](perimeter-minimum-initial.json). There were no measured losses.

| Size | Actual input dtype | Original ms | Current ms | Original/current | Max absolute error |
| --- | --- | ---: | ---: | ---: | ---: |
| 256 | uint8 | 1.812465667 | 0.162185889 | 11.175237× | 0 |
| 256 | float32 | 1.814958500 | 0.173201457 | 10.478887× | 0 |
| 256 | float64 | 1.832604833 | 0.182032841 | 10.067441× | 0 |
| 1024 | uint8 | 26.589239000 | 1.657393714 | 16.042802× | 0 |
| 1024 | float32 | 26.894035000 | 1.835835833 | 14.649477× | 0 |
| 1024 | float64 | 26.775514000 | 1.903177667 | 14.068846× | 0 |

Source `4a98fc70aa9b8eeae76fd5da0caad1a8471a7efc` passed genuine [Release run 40429430891](https://github.com/bipxx/scikit-imagexx/actions/runs/40429430891). The producer passed 3,532 focused tests across 15 normal/scalar pairs, 3,696 color regression tests, 1,612 matching tests and the 1,389-node installed smoke cohort (232 registered functions). A fresh measurement host installed the exact archived wheel set offline; all 14 wheels, 13 dependencies, package/SDK files, sole core and source identity matched before and after measurement. All 53 outer artifact members passed CRC/SHA verification. The 13 older timing commands were skipped; only the six new Perimeter cases ran.

The Linux x86_64 runtime used Python 3.11.17, NumPy 1.24.4, SciPy 1.15.3 and scikit-image 0.26.0. Native and reference pools used one worker; AVX2 was available. The raw report records architecture, not a specific CPU model. Timings are warmed, allocation-inclusive public calls with seven paired rounds.

Inputs are binary masks prepared outside timing from seed-2026 gray images: `>127` for uint8 and `>0.5` for floating inputs, cast back to the same dtype. Roughly half the pixels are foreground. Public uint8 casting, allocation, binary erosion, border subtraction, modulo-256 stencil counting and the weighted sum remain inside timing. Erosion/counting/weighted arithmetic are owned C++; literal casting, coefficient construction and scalar boxing remain explicit providers. The frozen `2*gamma_100*S` parity check ran outside timing. These are default-neighborhood-4 image cases, not arbitrary labels.

This is the first comparison with the original library on one minimum-profile host. It does not establish a gain over a prior C++ product or extend the last full supported-profile qualification of **203 image APIs**.

Raw JSON SHA-256: `92863699094b8064d7aae5b8b8c44793f26e09abc6d3a9ad7c106e8f813eff5c`. Retained artifact `12702180638` (84,575,507 bytes): `6fdd567db6e3670947f884e84e19969766563ccf0ed9af87624184c2db572b65`. The complete archive remains outside Git.

Review: `build/phase3-perimeter-first-release-v1/final-review.json`, SHA-256 `e05953330471c144dff3dbe9d398b0fdd48d69782e075db0347ba73b592e851e`. Product wheel: `b706dc24bb1eaafe581833a9546c76aaf87a7b0e4637a3046229d6dcb4321739`; sole core: `73ae0fd481268a0cbf57e0b90f25d6e74a945c3c0dd1b0e62c5ccef817eae176`.
