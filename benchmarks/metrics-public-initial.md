# Metrics: public one-worker baseline

The [completed isolated run](https://github.com/bipxx/scikit-imagexx/actions/runs/39983272390) measured source `a9e3be3c69e686f6e03bd792ae1edaf5ec1210a4` against scikit-image 0.26.0. Each runtime passed 1231 strict checks and timed 1212 defined successful calls, with seven paired samples. All raw samples and slower workloads are retained.

| Runtime | Work | Cases | Geometric mean ratio | Slower cases |
|---|---|---:|---:|---:|
| windows-modern | public_metrics | 1212 | 0.847x | 658 |
| linux-modern | public_metrics | 1212 | 0.673x | 736 |
| linux-minimum | public_metrics | 1212 | 0.811x | 459 |

The baseline is slower overall on all three runtimes. Endian and stepped arrays are the most severe bottlenecks; small promotion calls also have overhead. These results do not meet the performance objective. The next trial targets typed SIMD loads while preserving reduction and callback order.

The [summary and all regressions](metrics-public-initial.json) links byte-preserved raw reports. Exact wheels, complete package/SDK seals where present, dependency logs and source fingerprints remain protected in the downloaded workflow artifacts. No local operation timings were used.
