# Fixed edge validation experiment

The metric remains the geometric mean C++ median latency for four fixed calls:
Roberts positive/negative diagonal, float32/float64, 512×1 column views of
512×512 arrays, one worker, eleven warmed allocation-inclusive samples.
The unchanged recipe is [edge_tuning.py](edge_tuning.py).

| Source | C++ geometric mean | Baseline / C++ |
|---|---:|---:|
| Logical-dtype fix `a280a0f` | 0.010338 ms | 1.00× |
| Cached validation `3d59ca2` | 0.004039 ms | 2.56× |
| Same code, confirmation | 0.005779 ms | 1.79× |

The change caches dtype working limits, uses scalar finite checks, rejects
incompatible kinds/sizes before dtype comparison, and omits the underflow-policy
check for directional convolution without a squaring stage. Directional
subnormal inputs are compared with the reference under warn/raise/call/print/log
policies. Roberts magnitude retains its explicit underflow-policy fallback.

The reference geometric mean also changed: −43.0% in the first candidate run
and −21.5% in confirmation. The paired reference/C++ speedup improved by 1.46×
and 1.40× respectively; all four candidate cases measured faster than the
reference in both observations. Ordinary system load is uncontrolled, without
affinity or frequency locking. [All raw data](edge-metadata-tuning.json) is retained.
These historical snapshots are excluded from [current verification](STATUS.md).
