# Exposure correction with non-default parameters

1024² images; 1 C++ worker(s). Strict native dispatch. Medians of seven warmed, allocation-inclusive batches against original scikit-image. Speedup = reference/current; below one means slower. General transcendental functions may use scalar libm even in AVX2 builds.

| API | Options | Dtype | Original ms | Native ms | Speedup |
|---|---|---|---:|---:|---:|
| exposure.adjust_gamma | `{'gamma': 0.7, 'gain': 1.3}` | uint8 | 2.1108 | 0.1573 | 13.42x |
| exposure.adjust_gamma | `{'gamma': 0.5, 'gain': 1.3}` | uint8 | 2.1940 | 0.1795 | 12.22x |
| exposure.adjust_gamma | `{'gamma': 2, 'gain': 1.3}` | uint8 | 2.9766 | 0.2852 | 10.44x |
| exposure.adjust_log | `{'gain': 0.7}` | uint8 | 32.0495 | 0.2446 | 131.03x |
| exposure.adjust_log | `{'gain': 0.7, 'inv': True}` | uint8 | 41.1957 | 0.2565 | 160.63x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.3}` | uint8 | 41.4308 | 0.2663 | 155.57x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.7, 'inv': True}` | uint8 | 48.1633 | 0.2822 | 170.66x |
| exposure.rescale_intensity | `{'in_range': (0, 0.7), 'out_range': (-1, 2)}` | uint8 | 32.8349 | 4.4510 | 7.38x |
| exposure.adjust_gamma | `{'gamma': 0.7, 'gain': 1.3}` | float32 | 41.6803 | 12.3451 | 3.38x |
| exposure.adjust_gamma | `{'gamma': 0.5, 'gain': 1.3}` | float32 | 9.6905 | 3.2898 | 2.95x |
| exposure.adjust_gamma | `{'gamma': 2, 'gain': 1.3}` | float32 | 13.2431 | 3.8277 | 3.46x |
| exposure.adjust_log | `{'gain': 0.7}` | float32 | 15.2495 | 3.6706 | 4.15x |
| exposure.adjust_log | `{'gain': 0.7, 'inv': True}` | float32 | 18.7425 | 5.5154 | 3.40x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.3}` | float32 | 6.4026 | 1.0841 | 5.91x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.7, 'inv': True}` | float32 | 7.1287 | 1.1417 | 6.24x |
| exposure.rescale_intensity | `{'in_range': (0, 0.7), 'out_range': (-1, 2)}` | float32 | 7.9497 | 1.7119 | 4.64x |
| exposure.adjust_gamma | `{'gamma': 0.7, 'gain': 1.3}` | float64 | 39.7255 | 9.5316 | 4.17x |
| exposure.adjust_gamma | `{'gamma': 0.5, 'gain': 1.3}` | float64 | 17.6184 | 3.8586 | 4.57x |
| exposure.adjust_gamma | `{'gamma': 2, 'gain': 1.3}` | float64 | 14.3137 | 2.8430 | 5.03x |
| exposure.adjust_log | `{'gain': 0.7}` | float64 | 27.5465 | 4.8749 | 5.65x |
| exposure.adjust_log | `{'gain': 0.7, 'inv': True}` | float64 | 29.1054 | 4.5975 | 6.33x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.3}` | float64 | 27.4964 | 4.2783 | 6.43x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.7, 'inv': True}` | float64 | 24.8588 | 3.5478 | 7.01x |
| exposure.rescale_intensity | `{'in_range': (0, 0.7), 'out_range': (-1, 2)}` | float64 | 15.8879 | 3.1396 | 5.06x |
