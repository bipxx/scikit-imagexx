# metrics one-worker simd-load-trial

Source `6545e330759f1419b9080d5cb04fa18c069096ae` measured against scikit-image 0.26.0 in the [isolated hosted run](https://github.com/bipxx/scikit-imagexx/actions/runs/40000634063). All 1231 parity checks passed per runtime; 1212 calls have seven paired samples. Raw reports and every slower case are retained.

| Runtime | Work kind | Cases | Reference/current geometric mean | Slower cases |
|---|---|---:|---:|---:|
| windows-modern | public-metric-call | 1212 | 0.873x | 637 |
| linux-modern | public-metric-call | 1212 | 0.806x | 521 |
| linux-minimum | public-metric-call | 1212 | 0.917x | 441 |

The native implementation remains slower overall. Half-precision stepped/reversed/large inputs and signed min-max normalization remain measured bottlenecks. CPU models changed in two hosted profiles; these separate runs do not establish a causal previous/current improvement.

[Summary, raw reports and all regressions](metrics-public-simd-load-trial.json). No local operation timing was used.
