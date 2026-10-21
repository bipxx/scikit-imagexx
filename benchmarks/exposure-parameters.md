# Exposure correction with non-default parameters

1024² images; 1 C++ worker(s). Strict native dispatch. Medians of seven warmed, allocation-inclusive batches against original scikit-image. Speedup = reference/current; below one means slower. General transcendental functions may use scalar libm even in AVX2 builds.

| API | Options | Dtype | Original ms | Native ms | Speedup |
|---|---|---|---:|---:|---:|
| exposure.adjust_gamma | `{'gamma': 0.7, 'gain': 1.3}` | uint8 | 2.1595 | 0.1590 | 13.58x |
| exposure.adjust_gamma | `{'gamma': 0.5, 'gain': 1.3}` | uint8 | 2.0865 | 0.1465 | 14.24x |
| exposure.adjust_gamma | `{'gamma': 2, 'gain': 1.3}` | uint8 | 2.0712 | 0.1495 | 13.85x |
| exposure.adjust_log | `{'gain': 0.7}` | uint8 | 16.5363 | 0.1449 | 114.12x |
| exposure.adjust_log | `{'gain': 0.7, 'inv': True}` | uint8 | 22.5970 | 0.1527 | 147.99x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.3}` | uint8 | 21.2316 | 0.1762 | 120.50x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.7, 'inv': True}` | uint8 | 23.5558 | 0.1598 | 147.37x |
| exposure.rescale_intensity | `{'in_range': (0, 0.7), 'out_range': (-1, 2)}` | uint8 | 15.3350 | 2.8348 | 5.41x |
| exposure.adjust_gamma | `{'gamma': 0.7, 'gain': 1.3}` | float32 | 10.0876 | 6.8175 | 1.48x |
| exposure.adjust_gamma | `{'gamma': 0.5, 'gain': 1.3}` | float32 | 3.0189 | 1.2447 | 2.43x |
| exposure.adjust_gamma | `{'gamma': 2, 'gain': 1.3}` | float32 | 3.8168 | 1.3652 | 2.80x |
| exposure.adjust_log | `{'gain': 0.7}` | float32 | 7.6673 | 2.7768 | 2.76x |
| exposure.adjust_log | `{'gain': 0.7, 'inv': True}` | float32 | 15.9884 | 5.6021 | 2.85x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.3}` | float32 | 6.2289 | 1.0201 | 6.11x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.7, 'inv': True}` | float32 | 6.9909 | 1.1300 | 6.19x |
| exposure.rescale_intensity | `{'in_range': (0, 0.7), 'out_range': (-1, 2)}` | float32 | 5.9908 | 1.5693 | 3.82x |
| exposure.adjust_gamma | `{'gamma': 0.7, 'gain': 1.3}` | float64 | 29.6304 | 9.1836 | 3.23x |
| exposure.adjust_gamma | `{'gamma': 0.5, 'gain': 1.3}` | float64 | 17.5192 | 4.1630 | 4.21x |
| exposure.adjust_gamma | `{'gamma': 2, 'gain': 1.3}` | float64 | 23.2700 | 4.6896 | 4.96x |
| exposure.adjust_log | `{'gain': 0.7}` | float64 | 23.7353 | 4.7151 | 5.03x |
| exposure.adjust_log | `{'gain': 0.7, 'inv': True}` | float64 | 31.3327 | 4.7591 | 6.58x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.3}` | float64 | 25.5153 | 4.7118 | 5.42x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.7, 'inv': True}` | float64 | 25.1872 | 3.7774 | 6.67x |
| exposure.rescale_intensity | `{'in_range': (0, 0.7), 'out_range': (-1, 2)}` | float64 | 17.0431 | 2.9690 | 5.74x |
