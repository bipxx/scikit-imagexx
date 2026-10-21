# Exposure correction with non-default parameters

1024² images; 1 C++ worker(s). Strict native dispatch. Medians of seven warmed, allocation-inclusive batches against original scikit-image. Speedup = reference/current; below one means slower. General transcendental functions may use scalar libm even in AVX2 builds.

| API | Options | Dtype | Original ms | Native ms | Speedup |
|---|---|---|---:|---:|---:|
| exposure.adjust_gamma | `{'gamma': 0.7, 'gain': 1.3}` | uint8 | 2.0177 | 0.1443 | 13.98x |
| exposure.adjust_gamma | `{'gamma': 0.5, 'gain': 1.3}` | uint8 | 2.0046 | 0.1411 | 14.21x |
| exposure.adjust_gamma | `{'gamma': 2, 'gain': 1.3}` | uint8 | 2.3374 | 0.1623 | 14.40x |
| exposure.adjust_log | `{'gain': 0.7}` | uint8 | 19.1814 | 0.1692 | 113.40x |
| exposure.adjust_log | `{'gain': 0.7, 'inv': True}` | uint8 | 21.3210 | 0.1441 | 147.92x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.3}` | uint8 | 18.2160 | 0.1474 | 123.54x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.7, 'inv': True}` | uint8 | 22.0558 | 0.1611 | 136.89x |
| exposure.rescale_intensity | `{'in_range': (0, 0.7), 'out_range': (-1, 2)}` | uint8 | 13.5711 | 2.1757 | 6.24x |
| exposure.adjust_gamma | `{'gamma': 0.7, 'gain': 1.3}` | float32 | 9.1915 | 6.8586 | 1.34x |
| exposure.adjust_gamma | `{'gamma': 0.5, 'gain': 1.3}` | float32 | 2.1327 | 1.0153 | 2.10x |
| exposure.adjust_gamma | `{'gamma': 2, 'gain': 1.3}` | float32 | 2.0970 | 1.0501 | 2.00x |
| exposure.adjust_log | `{'gain': 0.7}` | float32 | 6.3244 | 3.0816 | 2.05x |
| exposure.adjust_log | `{'gain': 0.7, 'inv': True}` | float32 | 8.9512 | 3.8087 | 2.35x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.3}` | float32 | 3.7784 | 0.6596 | 5.73x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.7, 'inv': True}` | float32 | 2.9942 | 0.6353 | 4.71x |
| exposure.rescale_intensity | `{'in_range': (0, 0.7), 'out_range': (-1, 2)}` | float32 | 4.3892 | 1.0826 | 4.05x |
| exposure.adjust_gamma | `{'gamma': 0.7, 'gain': 1.3}` | float64 | 20.4882 | 7.1243 | 2.88x |
| exposure.adjust_gamma | `{'gamma': 0.5, 'gain': 1.3}` | float64 | 12.6808 | 2.8987 | 4.37x |
| exposure.adjust_gamma | `{'gamma': 2, 'gain': 1.3}` | float64 | 11.8375 | 2.1742 | 5.44x |
| exposure.adjust_log | `{'gain': 0.7}` | float64 | 18.4712 | 3.2705 | 5.65x |
| exposure.adjust_log | `{'gain': 0.7, 'inv': True}` | float64 | 20.5502 | 3.5366 | 5.81x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.3}` | float64 | 22.9782 | 3.7412 | 6.14x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.7, 'inv': True}` | float64 | 22.1225 | 4.7574 | 4.65x |
| exposure.rescale_intensity | `{'in_range': (0, 0.7), 'out_range': (-1, 2)}` | float64 | 14.7284 | 2.7692 | 5.32x |
