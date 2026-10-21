# Exposure correction with non-default parameters

1024² images; 1 C++ worker(s). Strict native dispatch. Medians of seven warmed, allocation-inclusive batches against original scikit-image. Speedup = reference/current; below one means slower. General transcendental functions may use scalar libm even in AVX2 builds.

| API | Options | Dtype | Original ms | Native ms | Speedup |
|---|---|---|---:|---:|---:|
| exposure.adjust_gamma | `{'gamma': 0.7, 'gain': 1.3}` | uint8 | 2.0834 | 0.1591 | 13.09x |
| exposure.adjust_gamma | `{'gamma': 0.5, 'gain': 1.3}` | uint8 | 2.0510 | 0.1475 | 13.91x |
| exposure.adjust_gamma | `{'gamma': 2, 'gain': 1.3}` | uint8 | 2.0531 | 0.1455 | 14.11x |
| exposure.adjust_log | `{'gain': 0.7}` | uint8 | 17.0619 | 0.1467 | 116.27x |
| exposure.adjust_log | `{'gain': 0.7, 'inv': True}` | uint8 | 21.4816 | 0.1454 | 147.71x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.3}` | uint8 | 26.5677 | 0.3884 | 68.41x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.7, 'inv': True}` | uint8 | 31.9627 | 0.3166 | 100.95x |
| exposure.rescale_intensity | `{'in_range': (0, 0.7), 'out_range': (-1, 2)}` | uint8 | 16.5969 | 2.7776 | 5.98x |
| exposure.adjust_gamma | `{'gamma': 0.7, 'gain': 1.3}` | float32 | 11.1142 | 7.5706 | 1.47x |
| exposure.adjust_gamma | `{'gamma': 0.5, 'gain': 1.3}` | float32 | 3.7367 | 1.4934 | 2.50x |
| exposure.adjust_gamma | `{'gamma': 2, 'gain': 1.3}` | float32 | 3.4059 | 1.2353 | 2.76x |
| exposure.adjust_log | `{'gain': 0.7}` | float32 | 4.8360 | 2.5677 | 1.88x |
| exposure.adjust_log | `{'gain': 0.7, 'inv': True}` | float32 | 9.3629 | 3.7764 | 2.48x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.3}` | float32 | 3.2580 | 0.5842 | 5.58x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.7, 'inv': True}` | float32 | 3.4604 | 0.6128 | 5.65x |
| exposure.rescale_intensity | `{'in_range': (0, 0.7), 'out_range': (-1, 2)}` | float32 | 5.6829 | 1.1195 | 5.08x |
| exposure.adjust_gamma | `{'gamma': 0.7, 'gain': 1.3}` | float64 | 21.6613 | 7.6943 | 2.82x |
| exposure.adjust_gamma | `{'gamma': 0.5, 'gain': 1.3}` | float64 | 13.9475 | 2.6828 | 5.20x |
| exposure.adjust_gamma | `{'gamma': 2, 'gain': 1.3}` | float64 | 11.9999 | 2.1464 | 5.59x |
| exposure.adjust_log | `{'gain': 0.7}` | float64 | 18.4154 | 3.2820 | 5.61x |
| exposure.adjust_log | `{'gain': 0.7, 'inv': True}` | float64 | 22.2051 | 3.4292 | 6.48x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.3}` | float64 | 22.3802 | 3.8290 | 5.84x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.7, 'inv': True}` | float64 | 24.2531 | 4.3415 | 5.59x |
| exposure.rescale_intensity | `{'in_range': (0, 0.7), 'out_range': (-1, 2)}` | float64 | 15.4725 | 2.5378 | 6.10x |
