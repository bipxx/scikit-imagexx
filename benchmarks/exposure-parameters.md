# Exposure correction with non-default parameters

1024² images; 1 C++ worker(s). Strict native dispatch. Medians of seven warmed, allocation-inclusive batches against original scikit-image. Speedup = reference/current; below one means slower. General transcendental functions may use scalar libm even in AVX2 builds.

| API | Options | Dtype | Original ms | Native ms | Speedup |
|---|---|---|---:|---:|---:|
| exposure.adjust_gamma | `{'gamma': 0.7, 'gain': 1.3}` | uint8 | 2.0422 | 0.1646 | 12.41x |
| exposure.adjust_gamma | `{'gamma': 0.5, 'gain': 1.3}` | uint8 | 2.1297 | 0.1594 | 13.36x |
| exposure.adjust_gamma | `{'gamma': 2, 'gain': 1.3}` | uint8 | 2.1604 | 0.1480 | 14.59x |
| exposure.adjust_log | `{'gain': 0.7}` | uint8 | 15.7876 | 0.1639 | 96.32x |
| exposure.adjust_log | `{'gain': 0.7, 'inv': True}` | uint8 | 21.8236 | 0.1542 | 141.54x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.3}` | uint8 | 20.1618 | 0.1671 | 120.69x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.7, 'inv': True}` | uint8 | 22.3677 | 0.1750 | 127.79x |
| exposure.rescale_intensity | `{'in_range': (0, 0.7), 'out_range': (-1, 2)}` | uint8 | 13.8723 | 2.3215 | 5.98x |
| exposure.adjust_gamma | `{'gamma': 0.7, 'gain': 1.3}` | float32 | 9.5468 | 6.8261 | 1.40x |
| exposure.adjust_gamma | `{'gamma': 0.5, 'gain': 1.3}` | float32 | 2.3121 | 1.0443 | 2.21x |
| exposure.adjust_gamma | `{'gamma': 2, 'gain': 1.3}` | float32 | 3.0877 | 1.1427 | 2.70x |
| exposure.adjust_log | `{'gain': 0.7}` | float32 | 5.1336 | 2.5360 | 2.02x |
| exposure.adjust_log | `{'gain': 0.7, 'inv': True}` | float32 | 8.8474 | 3.7918 | 2.33x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.3}` | float32 | 3.3274 | 0.5955 | 5.59x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.7, 'inv': True}` | float32 | 3.5121 | 0.5818 | 6.04x |
| exposure.rescale_intensity | `{'in_range': (0, 0.7), 'out_range': (-1, 2)}` | float32 | 4.6147 | 1.0573 | 4.36x |
| exposure.adjust_gamma | `{'gamma': 0.7, 'gain': 1.3}` | float64 | 20.5513 | 6.5922 | 3.12x |
| exposure.adjust_gamma | `{'gamma': 0.5, 'gain': 1.3}` | float64 | 10.8625 | 2.2508 | 4.83x |
| exposure.adjust_gamma | `{'gamma': 2, 'gain': 1.3}` | float64 | 11.2727 | 2.3899 | 4.72x |
| exposure.adjust_log | `{'gain': 0.7}` | float64 | 17.5698 | 3.2050 | 5.48x |
| exposure.adjust_log | `{'gain': 0.7, 'inv': True}` | float64 | 22.1618 | 3.4154 | 6.49x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.3}` | float64 | 22.5618 | 3.3361 | 6.76x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.7, 'inv': True}` | float64 | 21.8997 | 3.8242 | 5.73x |
| exposure.rescale_intensity | `{'in_range': (0, 0.7), 'out_range': (-1, 2)}` | float64 | 12.6268 | 2.4020 | 5.26x |
