# Exposure correction with non-default parameters

1024² images; 1 C++ worker(s). Strict native dispatch. Medians of seven warmed, allocation-inclusive batches against original scikit-image. Speedup = reference/current; below one means slower. General transcendental functions may use scalar libm even in AVX2 builds.

| API | Options | Dtype | Original ms | Native ms | Speedup |
|---|---|---|---:|---:|---:|
| exposure.adjust_gamma | `{'gamma': 0.7, 'gain': 1.3}` | uint8 | 2.0798 | 0.1485 | 14.00x |
| exposure.adjust_gamma | `{'gamma': 0.5, 'gain': 1.3}` | uint8 | 2.0141 | 0.1452 | 13.87x |
| exposure.adjust_gamma | `{'gamma': 2, 'gain': 1.3}` | uint8 | 2.0692 | 0.1471 | 14.07x |
| exposure.adjust_log | `{'gain': 0.7}` | uint8 | 17.9254 | 0.1528 | 117.31x |
| exposure.adjust_log | `{'gain': 0.7, 'inv': True}` | uint8 | 21.1590 | 0.1458 | 145.16x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.3}` | uint8 | 21.1903 | 0.1621 | 130.71x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.7, 'inv': True}` | uint8 | 21.3599 | 0.1501 | 142.33x |
| exposure.rescale_intensity | `{'in_range': (0, 0.7), 'out_range': (-1, 2)}` | uint8 | 13.0991 | 2.1953 | 5.97x |
| exposure.adjust_gamma | `{'gamma': 0.7, 'gain': 1.3}` | float32 | 9.1761 | 6.2359 | 1.47x |
| exposure.adjust_gamma | `{'gamma': 0.5, 'gain': 1.3}` | float32 | 2.1740 | 1.0665 | 2.04x |
| exposure.adjust_gamma | `{'gamma': 2, 'gain': 1.3}` | float32 | 2.6091 | 1.1452 | 2.28x |
| exposure.adjust_log | `{'gain': 0.7}` | float32 | 4.6161 | 2.5463 | 1.81x |
| exposure.adjust_log | `{'gain': 0.7, 'inv': True}` | float32 | 7.7717 | 3.3364 | 2.33x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.3}` | float32 | 2.9750 | 0.5256 | 5.66x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.7, 'inv': True}` | float32 | 3.4586 | 0.5900 | 5.86x |
| exposure.rescale_intensity | `{'in_range': (0, 0.7), 'out_range': (-1, 2)}` | float32 | 6.0864 | 1.2549 | 4.85x |
| exposure.adjust_gamma | `{'gamma': 0.7, 'gain': 1.3}` | float64 | 23.2258 | 7.3299 | 3.17x |
| exposure.adjust_gamma | `{'gamma': 0.5, 'gain': 1.3}` | float64 | 11.2383 | 2.4325 | 4.62x |
| exposure.adjust_gamma | `{'gamma': 2, 'gain': 1.3}` | float64 | 11.1753 | 2.3268 | 4.80x |
| exposure.adjust_log | `{'gain': 0.7}` | float64 | 17.7057 | 3.0006 | 5.90x |
| exposure.adjust_log | `{'gain': 0.7, 'inv': True}` | float64 | 20.6676 | 3.2500 | 6.36x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.3}` | float64 | 20.9139 | 3.5036 | 5.97x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.7, 'inv': True}` | float64 | 24.3790 | 4.0712 | 5.99x |
| exposure.rescale_intensity | `{'in_range': (0, 0.7), 'out_range': (-1, 2)}` | float64 | 12.4065 | 2.1422 | 5.79x |
