# Exposure correction with non-default parameters

1024² images; 1 C++ worker(s). Strict native dispatch. Medians of seven warmed, allocation-inclusive batches against original scikit-image. Speedup = reference/current; below one means slower. General transcendental functions may use scalar libm even in AVX2 builds.

| API | Options | Dtype | Original ms | Native ms | Speedup |
|---|---|---|---:|---:|---:|
| exposure.adjust_gamma | `{'gamma': 0.7, 'gain': 1.3}` | uint8 | 2.6088 | 0.1769 | 14.75x |
| exposure.adjust_gamma | `{'gamma': 0.5, 'gain': 1.3}` | uint8 | 2.4865 | 0.1758 | 14.14x |
| exposure.adjust_gamma | `{'gamma': 2, 'gain': 1.3}` | uint8 | 2.1794 | 0.1711 | 12.74x |
| exposure.adjust_log | `{'gain': 0.7}` | uint8 | 18.7735 | 0.1612 | 116.46x |
| exposure.adjust_log | `{'gain': 0.7, 'inv': True}` | uint8 | 23.9339 | 0.1586 | 150.91x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.3}` | uint8 | 22.2357 | 0.1462 | 152.04x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.7, 'inv': True}` | uint8 | 26.0686 | 0.1599 | 163.00x |
| exposure.rescale_intensity | `{'in_range': (0, 0.7), 'out_range': (-1, 2)}` | uint8 | 16.8939 | 2.5932 | 6.51x |
| exposure.adjust_gamma | `{'gamma': 0.7, 'gain': 1.3}` | float32 | 10.3695 | 6.2970 | 1.65x |
| exposure.adjust_gamma | `{'gamma': 0.5, 'gain': 1.3}` | float32 | 2.8785 | 1.2438 | 2.31x |
| exposure.adjust_gamma | `{'gamma': 2, 'gain': 1.3}` | float32 | 3.3979 | 1.1557 | 2.94x |
| exposure.adjust_log | `{'gain': 0.7}` | float32 | 6.2737 | 2.6334 | 2.38x |
| exposure.adjust_log | `{'gain': 0.7, 'inv': True}` | float32 | 9.8896 | 3.5471 | 2.79x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.3}` | float32 | 4.2119 | 0.6686 | 6.30x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.7, 'inv': True}` | float32 | 4.2323 | 0.7460 | 5.67x |
| exposure.rescale_intensity | `{'in_range': (0, 0.7), 'out_range': (-1, 2)}` | float32 | 4.8361 | 0.3476 | 13.91x |
| exposure.adjust_gamma | `{'gamma': 0.7, 'gain': 1.3}` | float64 | 23.2517 | 7.0268 | 3.31x |
| exposure.adjust_gamma | `{'gamma': 0.5, 'gain': 1.3}` | float64 | 13.8168 | 2.8697 | 4.81x |
| exposure.adjust_gamma | `{'gamma': 2, 'gain': 1.3}` | float64 | 15.5657 | 2.9268 | 5.32x |
| exposure.adjust_log | `{'gain': 0.7}` | float64 | 21.3997 | 3.2723 | 6.54x |
| exposure.adjust_log | `{'gain': 0.7, 'inv': True}` | float64 | 25.5404 | 3.8788 | 6.58x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.3}` | float64 | 24.9152 | 3.4175 | 7.29x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.7, 'inv': True}` | float64 | 23.6212 | 3.6239 | 6.52x |
| exposure.rescale_intensity | `{'in_range': (0, 0.7), 'out_range': (-1, 2)}` | float64 | 16.6619 | 3.1462 | 5.30x |
