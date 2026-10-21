# Exposure correction with non-default parameters

1024² images; 1 C++ worker(s). Strict native dispatch. Medians of seven warmed, allocation-inclusive batches against original scikit-image. Speedup = reference/current; below one means slower. General transcendental functions may use scalar libm even in AVX2 builds.

| API | Options | Dtype | Original ms | Native ms | Speedup |
|---|---|---|---:|---:|---:|
| exposure.adjust_gamma | `{'gamma': 0.7, 'gain': 1.3}` | uint8 | 2.1992 | 0.1678 | 13.11x |
| exposure.adjust_gamma | `{'gamma': 0.5, 'gain': 1.3}` | uint8 | 2.1580 | 0.1727 | 12.49x |
| exposure.adjust_gamma | `{'gamma': 2, 'gain': 1.3}` | uint8 | 2.2647 | 0.1742 | 13.00x |
| exposure.adjust_log | `{'gain': 0.7}` | uint8 | 19.6776 | 0.1673 | 117.60x |
| exposure.adjust_log | `{'gain': 0.7, 'inv': True}` | uint8 | 28.0150 | 0.1529 | 183.21x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.3}` | uint8 | 29.0993 | 0.3292 | 88.40x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.7, 'inv': True}` | uint8 | 34.6833 | 0.3827 | 90.62x |
| exposure.rescale_intensity | `{'in_range': (0, 0.7), 'out_range': (-1, 2)}` | uint8 | 20.9540 | 3.4266 | 6.12x |
| exposure.adjust_gamma | `{'gamma': 0.7, 'gain': 1.3}` | float32 | 12.5881 | 9.1079 | 1.38x |
| exposure.adjust_gamma | `{'gamma': 0.5, 'gain': 1.3}` | float32 | 5.2762 | 1.6160 | 3.26x |
| exposure.adjust_gamma | `{'gamma': 2, 'gain': 1.3}` | float32 | 5.5357 | 1.8504 | 2.99x |
| exposure.adjust_log | `{'gain': 0.7}` | float32 | 15.4404 | 4.0322 | 3.83x |
| exposure.adjust_log | `{'gain': 0.7, 'inv': True}` | float32 | 18.6519 | 6.2549 | 2.98x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.3}` | float32 | 14.3260 | 1.8183 | 7.88x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.7, 'inv': True}` | float32 | 6.2581 | 1.2694 | 4.93x |
| exposure.rescale_intensity | `{'in_range': (0, 0.7), 'out_range': (-1, 2)}` | float32 | 7.9775 | 1.4089 | 5.66x |
| exposure.adjust_gamma | `{'gamma': 0.7, 'gain': 1.3}` | float64 | 29.3311 | 7.4822 | 3.92x |
| exposure.adjust_gamma | `{'gamma': 0.5, 'gain': 1.3}` | float64 | 14.9004 | 3.8665 | 3.85x |
| exposure.adjust_gamma | `{'gamma': 2, 'gain': 1.3}` | float64 | 18.1236 | 3.9517 | 4.59x |
| exposure.adjust_log | `{'gain': 0.7}` | float64 | 28.9452 | 4.3737 | 6.62x |
| exposure.adjust_log | `{'gain': 0.7, 'inv': True}` | float64 | 30.4199 | 4.1899 | 7.26x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.3}` | float64 | 28.6981 | 5.3490 | 5.37x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.7, 'inv': True}` | float64 | 26.4941 | 5.4636 | 4.85x |
| exposure.rescale_intensity | `{'in_range': (0, 0.7), 'out_range': (-1, 2)}` | float64 | 19.8701 | 4.1189 | 4.82x |
