# Exposure correction with non-default parameters

1024² images; 1 C++ worker(s). Strict native dispatch. Medians of seven warmed, allocation-inclusive batches against original scikit-image. Speedup = reference/current; below one means slower. General transcendental functions may use scalar libm even in AVX2 builds.

| API | Options | Dtype | Original ms | Native ms | Speedup |
|---|---|---|---:|---:|---:|
| exposure.adjust_gamma | `{'gamma': 0.7, 'gain': 1.3}` | uint8 | 2.0249 | 0.1455 | 13.92x |
| exposure.adjust_gamma | `{'gamma': 0.5, 'gain': 1.3}` | uint8 | 2.0537 | 0.1621 | 12.67x |
| exposure.adjust_gamma | `{'gamma': 2, 'gain': 1.3}` | uint8 | 2.0947 | 0.1472 | 14.23x |
| exposure.adjust_log | `{'gain': 0.7}` | uint8 | 22.2187 | 0.1562 | 142.24x |
| exposure.adjust_log | `{'gain': 0.7, 'inv': True}` | uint8 | 23.0524 | 0.1569 | 146.94x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.3}` | uint8 | 20.4833 | 0.3399 | 60.25x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.7, 'inv': True}` | uint8 | 22.1129 | 0.1490 | 148.38x |
| exposure.rescale_intensity | `{'in_range': (0, 0.7), 'out_range': (-1, 2)}` | uint8 | 13.3638 | 1.9248 | 6.94x |
| exposure.adjust_gamma | `{'gamma': 0.7, 'gain': 1.3}` | float32 | 9.4782 | 6.3709 | 1.49x |
| exposure.adjust_gamma | `{'gamma': 0.5, 'gain': 1.3}` | float32 | 2.1879 | 1.0537 | 2.08x |
| exposure.adjust_gamma | `{'gamma': 2, 'gain': 1.3}` | float32 | 2.4344 | 1.1184 | 2.18x |
| exposure.adjust_log | `{'gain': 0.7}` | float32 | 4.8769 | 2.4816 | 1.97x |
| exposure.adjust_log | `{'gain': 0.7, 'inv': True}` | float32 | 7.7451 | 3.2955 | 2.35x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.3}` | float32 | 3.3050 | 0.5402 | 6.12x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.7, 'inv': True}` | float32 | 3.3205 | 0.5565 | 5.97x |
| exposure.rescale_intensity | `{'in_range': (0, 0.7), 'out_range': (-1, 2)}` | float32 | 4.3108 | 1.0289 | 4.19x |
| exposure.adjust_gamma | `{'gamma': 0.7, 'gain': 1.3}` | float64 | 21.0300 | 6.6607 | 3.16x |
| exposure.adjust_gamma | `{'gamma': 0.5, 'gain': 1.3}` | float64 | 11.6998 | 2.5480 | 4.59x |
| exposure.adjust_gamma | `{'gamma': 2, 'gain': 1.3}` | float64 | 11.1840 | 2.2167 | 5.05x |
| exposure.adjust_log | `{'gain': 0.7}` | float64 | 18.4640 | 3.3598 | 5.50x |
| exposure.adjust_log | `{'gain': 0.7, 'inv': True}` | float64 | 21.8061 | 3.5221 | 6.19x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.3}` | float64 | 21.2530 | 3.4527 | 6.16x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.7, 'inv': True}` | float64 | 22.2620 | 3.5132 | 6.34x |
| exposure.rescale_intensity | `{'in_range': (0, 0.7), 'out_range': (-1, 2)}` | float64 | 13.1041 | 2.3814 | 5.50x |
