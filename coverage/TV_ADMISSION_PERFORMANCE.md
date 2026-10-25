# TV-Chambolle one-worker comparison

The independently reviewed campaign [40174240551](https://github.com/bipxx/scikit-imagexx/actions/runs/40174240551) measures `restoration.denoise_tv_chambolle` against scikit-image 0.26.0. It covers 70 fixed fixtures on three hosted machines, with one global worker and seven paired rounds. These results are accepted within that measured scope.

| Profile | After / original speedup | After / before speedup | Slower than original | Slower than before |
|---|---:|---:|---:|---:|
| windows | 1.557x | 1.158x | 27/70 | 13/70 |
| modern | 1.276x | 1.267x | 31/70 | 5/70 |
| minimum | 1.358x | 1.236x | 28/70 | 8/70 |

Speedups are geometric means of per-case ratios; values above one are faster. Every slower case remains in the [complete 210-row table](performance/tv-default-policy-40174240551.csv), including its original, before and after median in milliseconds. The raw 4,410 intervals and all full parity/dispatch observations remain in the authenticated workflow artifacts. Seven rounds do not establish statistical significance or predict other hardware.

The actual products are `5e068a5` before and `af7b2af` after, both from the qualified 192-path stage. The controller is `28cfc58`. This campaign does not measure the current 199-path product or later source additions. The earlier campaign `40163025231` remains diagnostic because its observer did not validate the actual default-policy timed route.

The new campaign checks all 400 legacy parity cases and the exact 70 default-policy timing cases before timing on each fresh host. Normal/scalar preparation and workers 1/4 parity remain distinct from one-worker measurement. Builds and pytest run on preparation hosts, followed by a barrier; fresh measurement hosts install archived wheels offline and perform no builds or pytest. Actual accepted C++ finite-entry returns, provider identity and NumPy policy preservation are verified.

Remaining bottlenecks include one/two-iteration raster cases and Tagged layouts. Their measured overhead drives the next source-level optimization plan; no per-stage cause or new improvement is claimed yet. Aggregate gains do not mean every operation is faster.

See the [root admission and exact proof pointers](evidence/tv-default-policy-performance-admission.json) for source, runtime, CPU and evidence limits.
