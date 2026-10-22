# Exposure correction with non-default parameters

1024² images; 1 C++ worker(s). Strict native dispatch. Medians of seven warmed, allocation-inclusive batches against original scikit-image. Speedup = reference/current; below one means slower. General transcendental functions may use scalar libm even in AVX2 builds.

| API | Options | Dtype | Original ms | Native ms | Speedup |
|---|---|---|---:|---:|---:|
| exposure.adjust_gamma | `{'gamma': 0.7, 'gain': 1.3}` | uint8 | 2.1417 | 0.1617 | 13.25x |
| exposure.adjust_gamma | `{'gamma': 0.5, 'gain': 1.3}` | uint8 | 2.0742 | 0.1494 | 13.88x |
| exposure.adjust_gamma | `{'gamma': 2, 'gain': 1.3}` | uint8 | 2.1213 | 0.1492 | 14.22x |
| exposure.adjust_log | `{'gain': 0.7}` | uint8 | 16.3281 | 0.1649 | 99.02x |
| exposure.adjust_log | `{'gain': 0.7, 'inv': True}` | uint8 | 22.2482 | 0.1485 | 149.83x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.3}` | uint8 | 19.8754 | 0.1581 | 125.70x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.7, 'inv': True}` | uint8 | 23.4679 | 0.1495 | 156.98x |
| exposure.rescale_intensity | `{'in_range': (0, 0.7), 'out_range': (-1, 2)}` | uint8 | 12.5322 | 2.0018 | 6.26x |
| exposure.adjust_gamma | `{'gamma': 0.7, 'gain': 1.3}` | float32 | 9.1711 | 7.0248 | 1.31x |
| exposure.adjust_gamma | `{'gamma': 0.5, 'gain': 1.3}` | float32 | 2.1346 | 1.1653 | 1.83x |
| exposure.adjust_gamma | `{'gamma': 2, 'gain': 1.3}` | float32 | 2.2652 | 1.0424 | 2.17x |
| exposure.adjust_log | `{'gain': 0.7}` | float32 | 4.7871 | 2.6251 | 1.82x |
| exposure.adjust_log | `{'gain': 0.7, 'inv': True}` | float32 | 8.8189 | 3.8369 | 2.30x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.3}` | float32 | 3.0694 | 0.5621 | 5.46x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.7, 'inv': True}` | float32 | 3.2474 | 0.6680 | 4.86x |
| exposure.rescale_intensity | `{'in_range': (0, 0.7), 'out_range': (-1, 2)}` | float32 | 4.3042 | 1.0273 | 4.19x |
| exposure.adjust_gamma | `{'gamma': 0.7, 'gain': 1.3}` | float64 | 20.8888 | 6.8467 | 3.05x |
| exposure.adjust_gamma | `{'gamma': 0.5, 'gain': 1.3}` | float64 | 10.8436 | 2.3587 | 4.60x |
| exposure.adjust_gamma | `{'gamma': 2, 'gain': 1.3}` | float64 | 13.2997 | 2.9261 | 4.55x |
| exposure.adjust_log | `{'gain': 0.7}` | float64 | 19.5787 | 3.1724 | 6.17x |
| exposure.adjust_log | `{'gain': 0.7, 'inv': True}` | float64 | 21.9595 | 3.4125 | 6.44x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.3}` | float64 | 23.8337 | 3.8933 | 6.12x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.7, 'inv': True}` | float64 | 20.9044 | 3.5933 | 5.82x |
| exposure.rescale_intensity | `{'in_range': (0, 0.7), 'out_range': (-1, 2)}` | float64 | 13.1270 | 2.2188 | 5.92x |
