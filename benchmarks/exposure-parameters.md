# Exposure correction with non-default parameters

1024² images; 1 C++ worker(s). Strict native dispatch. Medians of seven warmed, allocation-inclusive batches against original scikit-image. Speedup = reference/current; below one means slower. General transcendental functions may use scalar libm even in AVX2 builds.

| API | Options | Dtype | Original ms | Native ms | Speedup |
|---|---|---|---:|---:|---:|
| exposure.adjust_gamma | `{'gamma': 0.7, 'gain': 1.3}` | uint8 | 2.1662 | 0.1541 | 14.05x |
| exposure.adjust_gamma | `{'gamma': 0.5, 'gain': 1.3}` | uint8 | 2.1103 | 0.1548 | 13.63x |
| exposure.adjust_gamma | `{'gamma': 2, 'gain': 1.3}` | uint8 | 2.2583 | 0.1529 | 14.77x |
| exposure.adjust_log | `{'gain': 0.7}` | uint8 | 20.0114 | 0.1501 | 133.32x |
| exposure.adjust_log | `{'gain': 0.7, 'inv': True}` | uint8 | 24.5009 | 0.1579 | 155.13x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.3}` | uint8 | 23.9804 | 0.3125 | 76.74x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.7, 'inv': True}` | uint8 | 30.4881 | 0.1679 | 181.63x |
| exposure.rescale_intensity | `{'in_range': (0, 0.7), 'out_range': (-1, 2)}` | uint8 | 16.8017 | 2.3921 | 7.02x |
| exposure.adjust_gamma | `{'gamma': 0.7, 'gain': 1.3}` | float32 | 12.1424 | 8.0955 | 1.50x |
| exposure.adjust_gamma | `{'gamma': 0.5, 'gain': 1.3}` | float32 | 2.5819 | 1.2194 | 2.12x |
| exposure.adjust_gamma | `{'gamma': 2, 'gain': 1.3}` | float32 | 3.7950 | 1.5411 | 2.46x |
| exposure.adjust_log | `{'gain': 0.7}` | float32 | 5.9347 | 2.9304 | 2.03x |
| exposure.adjust_log | `{'gain': 0.7, 'inv': True}` | float32 | 12.4295 | 4.5673 | 2.72x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.3}` | float32 | 6.8559 | 0.7817 | 8.77x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.7, 'inv': True}` | float32 | 7.7912 | 1.0165 | 7.66x |
| exposure.rescale_intensity | `{'in_range': (0, 0.7), 'out_range': (-1, 2)}` | float32 | 5.2921 | 0.4560 | 11.60x |
| exposure.adjust_gamma | `{'gamma': 0.7, 'gain': 1.3}` | float64 | 26.1273 | 7.8980 | 3.31x |
| exposure.adjust_gamma | `{'gamma': 0.5, 'gain': 1.3}` | float64 | 18.1759 | 3.7843 | 4.80x |
| exposure.adjust_gamma | `{'gamma': 2, 'gain': 1.3}` | float64 | 20.8759 | 4.1281 | 5.06x |
| exposure.adjust_log | `{'gain': 0.7}` | float64 | 27.1204 | 4.6044 | 5.89x |
| exposure.adjust_log | `{'gain': 0.7, 'inv': True}` | float64 | 28.9137 | 5.1474 | 5.62x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.3}` | float64 | 43.2705 | 6.0665 | 7.13x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.7, 'inv': True}` | float64 | 30.8779 | 4.7458 | 6.51x |
| exposure.rescale_intensity | `{'in_range': (0, 0.7), 'out_range': (-1, 2)}` | float64 | 21.7925 | 4.0075 | 5.44x |
