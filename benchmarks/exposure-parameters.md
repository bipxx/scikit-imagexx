# Exposure correction with non-default parameters

1024² images; 1 C++ worker(s). Strict native dispatch. Medians of seven warmed, allocation-inclusive batches against original scikit-image. Speedup = reference/current; below one means slower. General transcendental functions may use scalar libm even in AVX2 builds.

| API | Options | Dtype | Original ms | Native ms | Speedup |
|---|---|---|---:|---:|---:|
| exposure.adjust_gamma | `{'gamma': 0.7, 'gain': 1.3}` | uint8 | 2.0388 | 0.1689 | 12.07x |
| exposure.adjust_gamma | `{'gamma': 0.5, 'gain': 1.3}` | uint8 | 2.1393 | 0.1427 | 14.99x |
| exposure.adjust_gamma | `{'gamma': 2, 'gain': 1.3}` | uint8 | 2.0750 | 0.1507 | 13.77x |
| exposure.adjust_log | `{'gain': 0.7}` | uint8 | 16.9397 | 0.1541 | 109.91x |
| exposure.adjust_log | `{'gain': 0.7, 'inv': True}` | uint8 | 21.8592 | 0.1522 | 143.65x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.3}` | uint8 | 21.5797 | 0.1643 | 131.36x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.7, 'inv': True}` | uint8 | 23.3176 | 0.1701 | 137.08x |
| exposure.rescale_intensity | `{'in_range': (0, 0.7), 'out_range': (-1, 2)}` | uint8 | 15.7940 | 2.7987 | 5.64x |
| exposure.adjust_gamma | `{'gamma': 0.7, 'gain': 1.3}` | float32 | 9.6427 | 6.4112 | 1.50x |
| exposure.adjust_gamma | `{'gamma': 0.5, 'gain': 1.3}` | float32 | 1.8815 | 1.0286 | 1.83x |
| exposure.adjust_gamma | `{'gamma': 2, 'gain': 1.3}` | float32 | 2.1354 | 1.1393 | 1.87x |
| exposure.adjust_log | `{'gain': 0.7}` | float32 | 4.4005 | 2.6130 | 1.68x |
| exposure.adjust_log | `{'gain': 0.7, 'inv': True}` | float32 | 8.6819 | 4.1741 | 2.08x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.3}` | float32 | 3.1498 | 0.6522 | 4.83x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.7, 'inv': True}` | float32 | 3.5738 | 0.5795 | 6.17x |
| exposure.rescale_intensity | `{'in_range': (0, 0.7), 'out_range': (-1, 2)}` | float32 | 4.4344 | 1.0462 | 4.24x |
| exposure.adjust_gamma | `{'gamma': 0.7, 'gain': 1.3}` | float64 | 21.0107 | 6.6604 | 3.15x |
| exposure.adjust_gamma | `{'gamma': 0.5, 'gain': 1.3}` | float64 | 11.6164 | 2.3324 | 4.98x |
| exposure.adjust_gamma | `{'gamma': 2, 'gain': 1.3}` | float64 | 12.6796 | 2.1420 | 5.92x |
| exposure.adjust_log | `{'gain': 0.7}` | float64 | 19.0507 | 3.4978 | 5.45x |
| exposure.adjust_log | `{'gain': 0.7, 'inv': True}` | float64 | 19.7908 | 3.2849 | 6.02x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.3}` | float64 | 22.8488 | 3.5011 | 6.53x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.7, 'inv': True}` | float64 | 21.1778 | 3.7769 | 5.61x |
| exposure.rescale_intensity | `{'in_range': (0, 0.7), 'out_range': (-1, 2)}` | float64 | 13.3218 | 2.4077 | 5.53x |
