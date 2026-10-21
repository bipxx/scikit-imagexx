# Exposure correction with non-default parameters

1024² images; 1 C++ worker(s). Strict native dispatch. Medians of seven warmed, allocation-inclusive batches against original scikit-image. Speedup = reference/current; below one means slower. General transcendental functions may use scalar libm even in AVX2 builds.

| API | Options | Dtype | Original ms | Native ms | Speedup |
|---|---|---|---:|---:|---:|
| exposure.adjust_gamma | `{'gamma': 0.7, 'gain': 1.3}` | uint8 | 2.2026 | 0.1674 | 13.16x |
| exposure.adjust_gamma | `{'gamma': 0.5, 'gain': 1.3}` | uint8 | 2.0774 | 0.1454 | 14.29x |
| exposure.adjust_gamma | `{'gamma': 2, 'gain': 1.3}` | uint8 | 2.0989 | 0.1479 | 14.19x |
| exposure.adjust_log | `{'gain': 0.7}` | uint8 | 17.4896 | 0.1472 | 118.81x |
| exposure.adjust_log | `{'gain': 0.7, 'inv': True}` | uint8 | 22.3455 | 0.1501 | 148.85x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.3}` | uint8 | 18.0952 | 0.1491 | 121.32x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.7, 'inv': True}` | uint8 | 24.5152 | 0.1496 | 163.88x |
| exposure.rescale_intensity | `{'in_range': (0, 0.7), 'out_range': (-1, 2)}` | uint8 | 15.1382 | 2.1261 | 7.12x |
| exposure.adjust_gamma | `{'gamma': 0.7, 'gain': 1.3}` | float32 | 9.7324 | 6.3610 | 1.53x |
| exposure.adjust_gamma | `{'gamma': 0.5, 'gain': 1.3}` | float32 | 2.5865 | 1.2231 | 2.11x |
| exposure.adjust_gamma | `{'gamma': 2, 'gain': 1.3}` | float32 | 3.7607 | 1.4781 | 2.54x |
| exposure.adjust_log | `{'gain': 0.7}` | float32 | 6.1852 | 2.4973 | 2.48x |
| exposure.adjust_log | `{'gain': 0.7, 'inv': True}` | float32 | 8.8371 | 3.6240 | 2.44x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.3}` | float32 | 3.9523 | 0.6167 | 6.41x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.7, 'inv': True}` | float32 | 3.4810 | 0.5627 | 6.19x |
| exposure.rescale_intensity | `{'in_range': (0, 0.7), 'out_range': (-1, 2)}` | float32 | 4.3111 | 1.0481 | 4.11x |
| exposure.adjust_gamma | `{'gamma': 0.7, 'gain': 1.3}` | float64 | 22.3893 | 6.7565 | 3.31x |
| exposure.adjust_gamma | `{'gamma': 0.5, 'gain': 1.3}` | float64 | 13.6063 | 2.7476 | 4.95x |
| exposure.adjust_gamma | `{'gamma': 2, 'gain': 1.3}` | float64 | 13.7359 | 2.5076 | 5.48x |
| exposure.adjust_log | `{'gain': 0.7}` | float64 | 21.9863 | 3.4107 | 6.45x |
| exposure.adjust_log | `{'gain': 0.7, 'inv': True}` | float64 | 21.6004 | 3.3137 | 6.52x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.3}` | float64 | 22.4937 | 3.3825 | 6.65x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.7, 'inv': True}` | float64 | 21.6089 | 3.4139 | 6.33x |
| exposure.rescale_intensity | `{'in_range': (0, 0.7), 'out_range': (-1, 2)}` | float64 | 13.6412 | 2.3922 | 5.70x |
