# Fixed Roberts column experiment

The metric is the geometric mean C++ median latency for four fixed calls:
`roberts_pos_diag` and `roberts_neg_diag`, float32/float64, and 512×1 views
of 512×512 arrays. Both backends use one worker. Each case checks values,
shape and dtype before eleven warmed, allocation-inclusive samples.

| Source | C++ geometric mean | Baseline / C++ | Cases faster than scikit-image |
|---|---:|---:|---:|
| Baseline `fe83cdc` | 0.014799 ms | 1.00× | 0 / 4 |
| Vector columns + shared input preparation `a8e0071` | 0.006766 ms | 2.19× | 4 / 4 |
| Same change, confirmation | 0.009599 ms | 1.54× | 4 / 4 |

The previous row loop handled one pixel per row and could not enter its SIMD
body on a single-column image. The specialized path vectorizes along the
column, retaining all reflected-border and floating-point stages. Floating
strided/unaligned input is prepared once, then reused for validation and filtering.
Large columns still partition pixels using the global worker setting.

The original-library geometric mean changed by −16.5% in the first candidate
run and +31.6% in confirmation. Measurements use ordinary load without affinity
or frequency locking. Both observations are retained in [all raw data](edge-tuning.json).
That JSON also archives the initial 57-case development pilot (53 faster cases);
the uncommitted pilot is historical evidence, separate from the fixed metric.
These historical snapshots are excluded from [current verification](STATUS.md).

Reproduce the fixed workload after building either tagged source revision:

```powershell
python -m pip install . --no-build-isolation
python benchmarks/edge_tuning.py --output build/edge-tuning.json
```
