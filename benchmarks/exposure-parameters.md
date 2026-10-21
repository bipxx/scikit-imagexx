# Exposure correction with non-default parameters

1024² images; 1 C++ worker(s). Strict native dispatch. Medians of seven warmed, allocation-inclusive batches against original scikit-image. Speedup = reference/current; below one means slower. General transcendental functions may use scalar libm even in AVX2 builds.

| API | Options | Dtype | Original ms | Native ms | Speedup |
|---|---|---|---:|---:|---:|
| exposure.adjust_gamma | `{'gamma': 0.7, 'gain': 1.3}` | uint8 | 2.0143 | 0.1403 | 14.35x |
| exposure.adjust_gamma | `{'gamma': 0.5, 'gain': 1.3}` | uint8 | 2.0376 | 0.1433 | 14.22x |
| exposure.adjust_gamma | `{'gamma': 2, 'gain': 1.3}` | uint8 | 2.3235 | 0.1835 | 12.66x |
| exposure.adjust_log | `{'gain': 0.7}` | uint8 | 16.6217 | 0.1676 | 99.18x |
| exposure.adjust_log | `{'gain': 0.7, 'inv': True}` | uint8 | 20.4769 | 0.1473 | 139.00x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.3}` | uint8 | 19.2118 | 0.1572 | 122.20x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.7, 'inv': True}` | uint8 | 23.0050 | 0.1682 | 136.73x |
| exposure.rescale_intensity | `{'in_range': (0, 0.7), 'out_range': (-1, 2)}` | uint8 | 13.0556 | 2.1042 | 6.20x |
| exposure.adjust_gamma | `{'gamma': 0.7, 'gain': 1.3}` | float32 | 9.2618 | 6.7722 | 1.37x |
| exposure.adjust_gamma | `{'gamma': 0.5, 'gain': 1.3}` | float32 | 2.4059 | 1.1366 | 2.12x |
| exposure.adjust_gamma | `{'gamma': 2, 'gain': 1.3}` | float32 | 2.5262 | 1.1403 | 2.22x |
| exposure.adjust_log | `{'gain': 0.7}` | float32 | 5.3964 | 3.6003 | 1.50x |
| exposure.adjust_log | `{'gain': 0.7, 'inv': True}` | float32 | 8.9713 | 4.7091 | 1.91x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.3}` | float32 | 2.8981 | 0.4951 | 5.85x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.7, 'inv': True}` | float32 | 3.3331 | 0.5557 | 6.00x |
| exposure.rescale_intensity | `{'in_range': (0, 0.7), 'out_range': (-1, 2)}` | float32 | 5.1189 | 1.2813 | 3.99x |
| exposure.adjust_gamma | `{'gamma': 0.7, 'gain': 1.3}` | float64 | 21.4600 | 7.1660 | 2.99x |
| exposure.adjust_gamma | `{'gamma': 0.5, 'gain': 1.3}` | float64 | 11.8005 | 2.3734 | 4.97x |
| exposure.adjust_gamma | `{'gamma': 2, 'gain': 1.3}` | float64 | 10.5603 | 2.1609 | 4.89x |
| exposure.adjust_log | `{'gain': 0.7}` | float64 | 18.5680 | 3.1730 | 5.85x |
| exposure.adjust_log | `{'gain': 0.7, 'inv': True}` | float64 | 21.4956 | 3.5163 | 6.11x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.3}` | float64 | 20.2162 | 3.3726 | 5.99x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.7, 'inv': True}` | float64 | 21.1928 | 3.3561 | 6.31x |
| exposure.rescale_intensity | `{'in_range': (0, 0.7), 'out_range': (-1, 2)}` | float64 | 13.2222 | 2.3188 | 5.70x |
