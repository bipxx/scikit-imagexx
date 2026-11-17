# Current public-matmul VI initial comparison

Status: [root scoped performance admission](../../coverage/evidence/vi-public-matmul-minimum-performance-root-admission-v1.json) passed; main delivery is pending.

[Run43979924366](https://github.com/bipxx/scikit-imagexx/actions/runs/43979924366) measured qualified source `aaea1b060e8acb4df08e855a6574e1449416ee42` using controller `166ccd4801e618f9522832b0066bcc45e1649140` and genuine minimum product14272412077 from producer43949036485. The [whole-product root admission](../../coverage/evidence/vi-public-matmul-whole-product-root-admission-v1.json) records exact four-profile/three-stager scope.

The host was Intel Xeon6975P-C, Python3.11.17, NumPy1.24.4, SciPy1.15.3 and original scikit-image0.26.0. Both implementations used one worker/threadpool. Each case has seven accepted timed batches per implementation, with100ms targets and250ms minimum warmups:56 samples and8warmup groups overall.

| Case (shape/labels/layout) | Original median ms | Native median ms | Original/native |
|---|---:|---:|---:|
| `128x128/8/C` | 1.035786 | 0.220744 | 4.692x |
| `128x128/64/F` | 1.151482 | 0.282711 | 4.073x |
| `256x256/8/F` | 2.787085 | 0.980393 | 2.843x |
| `256x256/64/C` | 3.020109 | 0.741865 | 4.071x |

All four cases won on this host; every case and raw sample is retained. Existing slower cases in other datasets retain their original scope. These results do not close broader API, dtype, provider or platform domains.

The native public route includes NumPy log2/matmul/log2/matmul callbacks and existing NumPy/SciPy representation work. C++ owns marginals, inverses, scaling, coalescing, products and ordered sparse reductions. The timing includes those public providers.

The mandatory first cohort and each measurement preflight execute strict output-word, target-FP, caller-state and native-ownership assertions. Successful output/target-FP byte buffers were not emitted; those checks have reached-runtime assertion evidence rather than independently reconstructed raw output witnesses. The new F1 helper was source-built once on the measurement host with full32 controls; its receipt is retained below.

- [Raw report](vi.json), including all cases and samples.
- [Raw batch journal](vi.batches.jsonl).
- [First-cohort summary](first-cohort-summary.json).
- [Current-host F1 receipt](helper-receipt.json).
- [Pre-preparation capacity receipt](preparation-capacity.json).

The result artifact was authenticated and extracted once in memory. Its transport archive is unavailable locally; retained raw products and concise receipts carry the evidence. Root performance admission passed; final281 source projection and main delivery remain pending.
