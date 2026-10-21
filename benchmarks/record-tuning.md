# Fixed structured-record SIMD experiment

All 24 workloads remain fixed: 512x512 images, three record layouts, RGB/RGBA expansion and four memory layouts, one worker, seven warmed allocation-inclusive medians. The command is `python benchmarks/alpha.py --records-only --threads 1`. Metric: geometric-mean native median latency, lower is better.

| Source | Native geometric mean ms | Baseline / native | Faster than scikit-image |
|---|---:|---:|---:|
| baseline `10900f1` | 8.700697 | 1.00x | 17/24 |
| six-byte fast path `9ddd595` | 5.555889 | 1.57x | 24/24 |
| confirmation `9ddd595` | 5.775613 | 1.51x | 24/24 |

The general byte plan supports mixed-endian fields and nested subarrays, with SIMD shuffles and bounded loads/stores. The six-byte specialization keeps shuffled values in registers, uses exact four-plus-two-byte input loads and exact sixteen-plus-two/eight-byte output stores, and removes per-pixel dynamic tail copies and plan traversal. It preserves the scalar fallback and shared worker limit. Nonzero record values, unaligned/negative/stepped storage, broadcast alpha, parallel images and warning behavior are checked against scikit-image.

six-byte fast path: reference geometric-mean drift -10.7%, paired reference/native speedup improvement 1.40x.
confirmation: reference geometric-mean drift -12.1%, paired reference/native speedup improvement 1.32x.

Ordinary system load is uncontrolled, without affinity or frequency locking. All raw observations, including slower baseline cases, are retained in [JSON](record-tuning.json). These historical snapshots are excluded from the current-source aggregate.
