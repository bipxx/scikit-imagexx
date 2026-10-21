# Exposure correction with non-default parameters

1024² images; 1 C++ worker(s). Strict native dispatch. Medians of seven warmed, allocation-inclusive batches against original scikit-image. Speedup = reference/current; below one means slower. General transcendental functions may use scalar libm even in AVX2 builds.

| API | Options | Dtype | Original ms | Native ms | Speedup |
|---|---|---|---:|---:|---:|
| exposure.adjust_gamma | `{'gamma': 0.7, 'gain': 1.3}` | uint8 | 2.1004 | 0.1594 | 13.18x |
| exposure.adjust_gamma | `{'gamma': 0.5, 'gain': 1.3}` | uint8 | 2.1189 | 0.1656 | 12.79x |
| exposure.adjust_gamma | `{'gamma': 2, 'gain': 1.3}` | uint8 | 2.0594 | 0.1555 | 13.25x |
| exposure.adjust_log | `{'gain': 0.7}` | uint8 | 18.4928 | 0.1667 | 110.92x |
| exposure.adjust_log | `{'gain': 0.7, 'inv': True}` | uint8 | 22.0424 | 0.1462 | 150.77x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.3}` | uint8 | 19.1563 | 0.1438 | 133.20x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.7, 'inv': True}` | uint8 | 21.8006 | 0.1528 | 142.64x |
| exposure.rescale_intensity | `{'in_range': (0, 0.7), 'out_range': (-1, 2)}` | uint8 | 13.3741 | 2.0929 | 6.39x |
| exposure.adjust_gamma | `{'gamma': 0.7, 'gain': 1.3}` | float32 | 9.1541 | 6.2508 | 1.46x |
| exposure.adjust_gamma | `{'gamma': 0.5, 'gain': 1.3}` | float32 | 2.4173 | 1.0130 | 2.39x |
| exposure.adjust_gamma | `{'gamma': 2, 'gain': 1.3}` | float32 | 2.5866 | 1.0278 | 2.52x |
| exposure.adjust_log | `{'gain': 0.7}` | float32 | 4.6315 | 2.5206 | 1.84x |
| exposure.adjust_log | `{'gain': 0.7, 'inv': True}` | float32 | 8.2233 | 3.3396 | 2.46x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.3}` | float32 | 3.2022 | 0.5500 | 5.82x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.7, 'inv': True}` | float32 | 3.2524 | 0.5317 | 6.12x |
| exposure.rescale_intensity | `{'in_range': (0, 0.7), 'out_range': (-1, 2)}` | float32 | 4.5022 | 1.0423 | 4.32x |
| exposure.adjust_gamma | `{'gamma': 0.7, 'gain': 1.3}` | float64 | 20.1404 | 6.5347 | 3.08x |
| exposure.adjust_gamma | `{'gamma': 0.5, 'gain': 1.3}` | float64 | 11.2368 | 2.3669 | 4.75x |
| exposure.adjust_gamma | `{'gamma': 2, 'gain': 1.3}` | float64 | 12.1229 | 2.2420 | 5.41x |
| exposure.adjust_log | `{'gain': 0.7}` | float64 | 18.4971 | 2.8797 | 6.42x |
| exposure.adjust_log | `{'gain': 0.7, 'inv': True}` | float64 | 20.7167 | 3.3209 | 6.24x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.3}` | float64 | 24.1623 | 3.7651 | 6.42x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.7, 'inv': True}` | float64 | 20.6862 | 3.2977 | 6.27x |
| exposure.rescale_intensity | `{'in_range': (0, 0.7), 'out_range': (-1, 2)}` | float64 | 12.3594 | 2.1766 | 5.68x |
