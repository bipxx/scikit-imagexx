# Exposure correction with non-default parameters

1024² images; 1 C++ worker(s). Strict native dispatch. Medians of seven warmed, allocation-inclusive batches against original scikit-image. Speedup = reference/current; below one means slower. General transcendental functions may use scalar libm even in AVX2 builds.

| API | Options | Dtype | Original ms | Native ms | Speedup |
|---|---|---|---:|---:|---:|
| exposure.adjust_gamma | `{'gamma': 0.7, 'gain': 1.3}` | uint8 | 2.2226 | 0.1810 | 12.28x |
| exposure.adjust_gamma | `{'gamma': 0.5, 'gain': 1.3}` | uint8 | 2.2168 | 0.1796 | 12.34x |
| exposure.adjust_gamma | `{'gamma': 2, 'gain': 1.3}` | uint8 | 2.6267 | 0.1877 | 14.00x |
| exposure.adjust_log | `{'gain': 0.7}` | uint8 | 20.8005 | 0.1658 | 125.44x |
| exposure.adjust_log | `{'gain': 0.7, 'inv': True}` | uint8 | 23.5375 | 0.1611 | 146.07x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.3}` | uint8 | 26.6659 | 0.2351 | 113.43x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.7, 'inv': True}` | uint8 | 31.3925 | 0.2477 | 126.71x |
| exposure.rescale_intensity | `{'in_range': (0, 0.7), 'out_range': (-1, 2)}` | uint8 | 19.9582 | 3.3236 | 6.01x |
| exposure.adjust_gamma | `{'gamma': 0.7, 'gain': 1.3}` | float32 | 14.0324 | 8.2537 | 1.70x |
| exposure.adjust_gamma | `{'gamma': 0.5, 'gain': 1.3}` | float32 | 3.9742 | 1.4839 | 2.68x |
| exposure.adjust_gamma | `{'gamma': 2, 'gain': 1.3}` | float32 | 7.0088 | 1.9653 | 3.57x |
| exposure.adjust_log | `{'gain': 0.7}` | float32 | 12.3642 | 3.4084 | 3.63x |
| exposure.adjust_log | `{'gain': 0.7, 'inv': True}` | float32 | 11.1475 | 3.9754 | 2.80x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.3}` | float32 | 5.2707 | 0.8928 | 5.90x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.7, 'inv': True}` | float32 | 6.4738 | 1.4042 | 4.61x |
| exposure.rescale_intensity | `{'in_range': (0, 0.7), 'out_range': (-1, 2)}` | float32 | 5.2871 | 1.0302 | 5.13x |
| exposure.adjust_gamma | `{'gamma': 0.7, 'gain': 1.3}` | float64 | 26.8648 | 8.7820 | 3.06x |
| exposure.adjust_gamma | `{'gamma': 0.5, 'gain': 1.3}` | float64 | 15.8782 | 3.2886 | 4.83x |
| exposure.adjust_gamma | `{'gamma': 2, 'gain': 1.3}` | float64 | 14.1565 | 2.9128 | 4.86x |
| exposure.adjust_log | `{'gain': 0.7}` | float64 | 32.2346 | 5.5962 | 5.76x |
| exposure.adjust_log | `{'gain': 0.7, 'inv': True}` | float64 | 32.7615 | 5.3096 | 6.17x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.3}` | float64 | 35.4177 | 5.7012 | 6.21x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.7, 'inv': True}` | float64 | 40.8074 | 5.9141 | 6.90x |
| exposure.rescale_intensity | `{'in_range': (0, 0.7), 'out_range': (-1, 2)}` | float64 | 16.4398 | 3.4556 | 4.76x |
