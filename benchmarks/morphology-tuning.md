# One-worker boolean rectangle optimization

Historical experiment on the same fixed 18-case workload: six grayscale operations, 512² boolean images, a 31×31 rectangular footprint, reflected borders, and seeded random, all-zero and all-one inputs. Both libraries use one worker; fallback is disabled. Values and dtype are checked before timing. Seven warmed samples include allocation and Python dispatch. Inputs, imports and I/O are excluded.

Baseline kernel: `3b4355a0c266353da8cff53d07bbbcba4d4b1f6b`. Packed kernel: `2795b771ca5b74d3cfca83327435895c6d33abcb`. These snapshots predate later Luv work and are separate from the [current-source verification](STATUS.md). Full samples are in [the JSON](morphology-tuning.json).

Geometric-mean native latency: **3.6556 ms → 0.1891 ms**, a **19.34×** improvement over the previous C++ code. All 18 packed cases also measured faster than original scikit-image in this run.

| API | Input | Previous C++ ms | Packed C++ ms | C++ improvement | Original ms | Original / packed |
|---|---|---:|---:|---:|---:|---:|
| erosion | random | 2.0525 | 0.1508 | 13.61× | 2.3304 | 15.46× |
| erosion | zeros | 2.0876 | 0.0987 | 21.16× | 1.2069 | 12.23× |
| erosion | ones | 2.1550 | 0.1044 | 20.64× | 1.1645 | 11.15× |
| dilation | random | 2.2516 | 0.1702 | 13.23× | 2.5502 | 14.98× |
| dilation | zeros | 2.3032 | 0.0971 | 23.72× | 1.2319 | 12.69× |
| dilation | ones | 2.3085 | 0.1252 | 18.44× | 1.3781 | 11.01× |
| opening | random | 4.7638 | 0.2809 | 16.96× | 3.7705 | 13.43× |
| opening | zeros | 4.5225 | 0.1880 | 24.05× | 2.6826 | 14.27× |
| opening | ones | 4.3853 | 0.1893 | 23.17× | 2.8762 | 15.20× |
| closing | random | 4.6783 | 0.2739 | 17.08× | 3.9157 | 14.30× |
| closing | zeros | 4.8965 | 0.2427 | 20.18× | 2.7239 | 11.23× |
| closing | ones | 4.6398 | 0.2470 | 18.79× | 2.7047 | 10.95× |
| white_tophat | random | 4.7797 | 0.2850 | 16.77× | 3.8285 | 13.43× |
| white_tophat | zeros | 5.7789 | 0.2180 | 26.51× | 2.6599 | 12.20× |
| white_tophat | ones | 4.8865 | 0.1930 | 25.32× | 2.6825 | 13.90× |
| black_tophat | random | 4.3322 | 0.3036 | 14.27× | 3.9801 | 13.11× |
| black_tophat | zeros | 4.5985 | 0.2386 | 19.27× | 2.8222 | 11.83× |
| black_tophat | ones | 4.5549 | 0.2078 | 21.92× | 2.6869 | 12.93× |

The packed path processes 64 boolean pixels per word and uses SIMD for packing, word reductions and unpacking. It reuses the general border maps and preserves a portable path. Noncanonical bool bytes and unsupported packed-path constants retain the general native grayscale kernel.

Reproduce the fixed workload with `python benchmarks/tune_morphology.py --output build/tuning.json`. For historical comparisons, keep the harness and its `run.py` helper available while installing each kernel revision. Measurements use ordinary system load without affinity or frequency locking; the ratios are not universal guarantees.
