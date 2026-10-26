# Shannon entropy: first minimum-Linux comparison

[Release40403748672](https://github.com/bipxx/scikit-imagexx/actions/runs/40403748672) measures exact source `62d127037400e16dcb8dcffd4dc63b2461cb0a30`, merged as `9f4fe3c`. **All six cases beat scikit-image**, with geometric mean **22.457450019x** and a range of **10.24x to 49.17x**. There are no losses in this cohort. The [exact raw JSON](shannon-entropy-minimum-initial.json) preserves all **84 latency samples** and reports zero output error for each case.

| Image size | Dtype | Original ms | Current ms | Original/current | Added ms |
|---|---|---:|---:|---:|---:|
| 256x256 | uint8 | 2.155488 | 0.043839 | 49.168110478x | -2.111648 |
| 256x256 | float32 | 5.092421 | 0.228758 | 22.261176316x | -4.863663 |
| 256x256 | float64 | 3.169360 | 0.238814 | 13.271256897x | -2.930546 |
| 1024x1024 | uint8 | 31.264113 | 0.687967 | 45.444182427x | -30.576146 |
| 1024x1024 | float32 | 85.224641 | 4.491464 | 18.974800794x | -80.733177 |
| 1024x1024 | float64 | 48.393187 | 4.725206 | 10.241497109x | -43.667981 |

Each public timing uses one global worker and includes input preparation, symbol counting, entropy arithmetic, allocation and scalar result construction. The six cases cover 256/1024-square images and actual uint8/float32/float64 inputs. Seven warmed paired rounds run in a fresh offline process. Original reference nats and numerical verification are prepared outside timing; the frozen entropy-only allowance is propagated through default-base conversion without changing the policy.

C++ owns sorted canonical counts for ordinary real inputs and binary64 probability/log/pairwise entropy. It reuses existing histogram/radix/hash and global worker primitives. Wide integer symbols remain distinct, NaNs share a symbol, and signed zeros compare equal. Count-only mode avoids the matching value/order output. NumPy preparation and unique for unsupported symbol types, scalar boxing and Python logarithmic base conversion remain explicit providers. Counting precedes base validation as in the source. This is native-partial support, not a fully C++ replacement of every input protocol.

The installed product passes eleven public normal/scalar contract pairs (2,744 checks), 1,377 smoke checks and the complete 1,612-test histogram-matching regression suite. Two-runtime components separately pass 244 new public and 88 shared checks. Native controls cover normal/scalar/sanitizer modes and separate portable/no-OpenMP configurations. All prior scalar policies and contextual refusals remain unchanged.

Source, wheel, installed package/core, SDK and before/after runtime identities match. Fourteen wheels include thirteen authenticated dependency wheels. The host is x86_64 minimum Linux, CPython3.11.17/NumPy1.24.4/SciPy1.15.3, comparing scikit-image0.26.0; a detailed CPU model was not captured. The measured source predates the separately kept float32 traversal change. This first original-library comparison is not a prior-C++ gain, a combined-current-main measurement, or full supported-profile qualification. Source coverage is now 226 native-partial image APIs plus three metadata functions; accepted full-profile coverage remains 203 image APIs.

Raw JSON SHA-256: `828cd7333364e82a7bda8803bb732d33cce0d6403fcfac01a33bdad33af153d5`. Parent artifact12691645770 SHA-256: `75d4c1efcb6ef3be1c2f1f39aa0fb7d56d80eb6b7c14c561e52e8d24dd7a015e`. Final review SHA-256: `13a22485b2e2bb46b8ad360c9b272d801fa53457fac3d1d91112acdb73e950ae`.
