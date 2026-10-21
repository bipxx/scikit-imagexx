# Fixed grayscale expansion experiment

The workload is fixed at 48 allocation-inclusive Python API calls: `gray2rgb`
and `gray2rgba`, 512×512 images, eight dtypes (`uint8`, `int8`, `uint16`,
`int16`, `float16`, `>u2`, `>f4`, `>f8`), and contiguous, reversed and
two-step layouts. Both implementations use one worker. Every case requires
exact shape, dtype and value parity before seven warmed timing samples.
The metric is the geometric mean of the 48 C++ median latencies; lower is better.

| Source | C++ geometric mean | Improvement over baseline | Cases faster than scikit-image |
|---|---:|---:|---:|
| Baseline `cc845a5` | 0.718469 ms | 1.00× | 30 / 48 |
| Vector loads and fixed endian masks `9cb88a5` | 0.391534 ms | 1.84× | 48 / 48 |
| Same change, confirmation | 0.444734 ms | 1.62× | 47 / 48 |

The change replaces scalar gathering for unit reverse and two-step views with
vector loads and shuffles. Overlapping loads end at the final selected element,
so they do not require readable padding after a strided allocation. Constant
2/4/8-byte shuffle masks replace runtime mask generation with integer divisions.
Native-order 256-bit blocks skip the redundant identity byte shuffle.

The original-library latency geometric mean changed by −5.4% in the first
candidate run and +4.2% in confirmation. Timings use ordinary system load,
without affinity or frequency locking. Both observations and the remaining
slower confirmation case are retained in [all raw snapshots](alpha-tuning.json).
The separate [initial full alpha matrix](alpha-initial.json) retains all 368
comparisons, including its 18 slower one-worker cases. These historical source
snapshots are excluded from the current-source aggregate in [STATUS](STATUS.md).

Reproduce each source snapshot after building that revision:

```powershell
python -m pip install . --no-build-isolation
python benchmarks/alpha_tuning.py --output build/alpha-tuning.json
```

The recipe is available on the feature branch after the experiment; when
checking out the baseline revision, preserve a copy of the unchanged recipe.
