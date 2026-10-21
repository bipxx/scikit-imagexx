# Exposure correction with non-default parameters

1024² images; 1 C++ worker(s). Strict native dispatch. Medians of seven warmed, allocation-inclusive batches against original scikit-image. Speedup = reference/current; below one means slower. General transcendental functions may use scalar libm even in AVX2 builds.

| API | Options | Dtype | Original ms | Native ms | Speedup |
|---|---|---|---:|---:|---:|
| exposure.adjust_gamma | `{'gamma': 0.7, 'gain': 1.3}` | uint8 | 2.1234 | 0.1519 | 13.98x |
| exposure.adjust_gamma | `{'gamma': 0.5, 'gain': 1.3}` | uint8 | 2.0736 | 0.1487 | 13.94x |
| exposure.adjust_gamma | `{'gamma': 2, 'gain': 1.3}` | uint8 | 2.1026 | 0.1474 | 14.27x |
| exposure.adjust_log | `{'gain': 0.7}` | uint8 | 18.2748 | 0.1445 | 126.49x |
| exposure.adjust_log | `{'gain': 0.7, 'inv': True}` | uint8 | 25.9844 | 0.1526 | 170.33x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.3}` | uint8 | 23.2890 | 0.2350 | 99.11x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.7, 'inv': True}` | uint8 | 43.7537 | 0.3116 | 140.44x |
| exposure.rescale_intensity | `{'in_range': (0, 0.7), 'out_range': (-1, 2)}` | uint8 | 25.1163 | 4.1298 | 6.08x |
| exposure.adjust_gamma | `{'gamma': 0.7, 'gain': 1.3}` | float32 | 12.2511 | 8.1900 | 1.50x |
| exposure.adjust_gamma | `{'gamma': 0.5, 'gain': 1.3}` | float32 | 4.1131 | 1.7813 | 2.31x |
| exposure.adjust_gamma | `{'gamma': 2, 'gain': 1.3}` | float32 | 7.3226 | 1.9834 | 3.69x |
| exposure.adjust_log | `{'gain': 0.7}` | float32 | 7.7538 | 2.8115 | 2.76x |
| exposure.adjust_log | `{'gain': 0.7, 'inv': True}` | float32 | 10.1950 | 4.0006 | 2.55x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.3}` | float32 | 5.7752 | 1.1241 | 5.14x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.7, 'inv': True}` | float32 | 8.0298 | 1.0602 | 7.57x |
| exposure.rescale_intensity | `{'in_range': (0, 0.7), 'out_range': (-1, 2)}` | float32 | 6.7267 | 0.7246 | 9.28x |
| exposure.adjust_gamma | `{'gamma': 0.7, 'gain': 1.3}` | float64 | 28.1913 | 8.5774 | 3.29x |
| exposure.adjust_gamma | `{'gamma': 0.5, 'gain': 1.3}` | float64 | 20.0934 | 4.2816 | 4.69x |
| exposure.adjust_gamma | `{'gamma': 2, 'gain': 1.3}` | float64 | 20.2252 | 3.6583 | 5.53x |
| exposure.adjust_log | `{'gain': 0.7}` | float64 | 22.3196 | 3.4695 | 6.43x |
| exposure.adjust_log | `{'gain': 0.7, 'inv': True}` | float64 | 22.9302 | 3.6586 | 6.27x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.3}` | float64 | 26.4342 | 3.5319 | 7.48x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.7, 'inv': True}` | float64 | 29.3762 | 4.5858 | 6.41x |
| exposure.rescale_intensity | `{'in_range': (0, 0.7), 'out_range': (-1, 2)}` | float64 | 16.3785 | 2.5896 | 6.32x |
