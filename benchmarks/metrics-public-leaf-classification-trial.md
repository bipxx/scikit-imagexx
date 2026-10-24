# Pairwise metrics: leaf-classification trial

The [completed isolated campaign](https://github.com/bipxx/scikit-imagexx/actions/runs/40037037771) measured exact source `0d5ab290d1c6aa97aeef120a9c567387f93ecdd6` against scikit-image 0.26.0. All fixed parity cases passed before seven paired samples per successful workload, using one native worker and one reference thread pool. Ratios above 1 favor the C++ public call.

| Runtime | Geometric mean public-call speedup | Slower cases | Worst speedup |
|---|---:|---:|---:|
| windows-modern | 1.2538x | 345/1212 | 0.2154x |
| linux-modern | 0.8070x | 620/1212 | 0.1261x |
| linux-minimum | 0.9697x | 428/1212 | 0.1551x |

Each runtime passed 1,231 cases; 1,212 successful calls were timed and all 19 expected errors remain untimed. MSE/NRMSE use relative-only error budgets, PSNR retains its recorded dB absolute budget; scalar types, dtype, signed zeros and warnings remain in the raw parity records. Linux modern/minimum remain slower than original in aggregate.

Measured slower examples (complete samples and workload specifications are retained in the JSON):

- windows-modern: `metrics.peak_signal_noise_ratio`, `stepped`, 0.2154x; workload index 989.
- linux-modern: `metrics.mean_squared_error`, `reversed`, 0.1261x; workload index 1041.
- linux-minimum: `metrics.mean_squared_error`, `reversed`, 0.1551x; workload index 984.

| Public API | Windows modern | Linux modern | Linux minimum |
|---|---:|---:|---:|
| `metrics.mean_squared_error` | 1.0948x | 0.7206x | 0.8666x |
| `metrics.normalized_root_mse` | 1.4071x | 0.8865x | 1.0512x |
| `metrics.peak_signal_noise_ratio` | 1.2348x | 0.7991x | 0.9742x |

These are observed ratios on the recorded CPUs for the exact source above. Hosted CPU/run differences prevent causal before/after attribution to the named patch. Individual slower calls remain, and this campaign does not establish universal performance superiority.

Offline verification covered complete product wheels/RECORDs, preserved package bytes, all 59 SDK members, source headers/Python and numerical runtime pins/freeze text. These older workflows did not archive a full dependency-binary closure. The metrics artifact also lacks an installed-SDK seal; SDK verification here is wheel/member/source evidence.

All three preparation jobs finished before fresh measurement jobs began; measurement jobs performed no builds or tests. The [summary and complete slower-case list](metrics-public-leaf-classification-trial.json) links three byte-exact raw reports, authoritative artifact IDs/hashes, all samples and source workload specifications. No image operations, timers, foreign-binary imports or source extraction were performed locally for this publication.
