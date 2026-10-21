# Exposure correction with non-default parameters

1024² images; 1 C++ worker(s). Strict native dispatch. Medians of seven warmed, allocation-inclusive batches against original scikit-image. Speedup = reference/current; below one means slower. General transcendental functions may use scalar libm even in AVX2 builds.

| API | Options | Dtype | Original ms | Native ms | Speedup |
|---|---|---|---:|---:|---:|
| exposure.adjust_gamma | `{'gamma': 0.7, 'gain': 1.3}` | uint8 | 2.0084 | 0.1574 | 12.76x |
| exposure.adjust_gamma | `{'gamma': 0.5, 'gain': 1.3}` | uint8 | 2.5279 | 0.1812 | 13.95x |
| exposure.adjust_gamma | `{'gamma': 2, 'gain': 1.3}` | uint8 | 2.1696 | 0.1438 | 15.09x |
| exposure.adjust_log | `{'gain': 0.7}` | uint8 | 16.0542 | 0.1539 | 104.34x |
| exposure.adjust_log | `{'gain': 0.7, 'inv': True}` | uint8 | 21.6250 | 0.1595 | 135.54x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.3}` | uint8 | 19.3247 | 0.1726 | 111.99x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.7, 'inv': True}` | uint8 | 22.6008 | 0.1779 | 127.04x |
| exposure.rescale_intensity | `{'in_range': (0, 0.7), 'out_range': (-1, 2)}` | uint8 | 13.9785 | 2.1821 | 6.41x |
| exposure.adjust_gamma | `{'gamma': 0.7, 'gain': 1.3}` | float32 | 9.0089 | 7.5297 | 1.20x |
| exposure.adjust_gamma | `{'gamma': 0.5, 'gain': 1.3}` | float32 | 1.9778 | 0.9265 | 2.13x |
| exposure.adjust_gamma | `{'gamma': 2, 'gain': 1.3}` | float32 | 2.2684 | 1.0125 | 2.24x |
| exposure.adjust_log | `{'gain': 0.7}` | float32 | 4.9529 | 2.6649 | 1.86x |
| exposure.adjust_log | `{'gain': 0.7, 'inv': True}` | float32 | 8.3761 | 3.5267 | 2.38x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.3}` | float32 | 3.5896 | 0.7033 | 5.10x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.7, 'inv': True}` | float32 | 3.1470 | 0.5907 | 5.33x |
| exposure.rescale_intensity | `{'in_range': (0, 0.7), 'out_range': (-1, 2)}` | float32 | 4.2651 | 1.0889 | 3.92x |
| exposure.adjust_gamma | `{'gamma': 0.7, 'gain': 1.3}` | float64 | 20.5551 | 7.0724 | 2.91x |
| exposure.adjust_gamma | `{'gamma': 0.5, 'gain': 1.3}` | float64 | 13.0005 | 2.7295 | 4.76x |
| exposure.adjust_gamma | `{'gamma': 2, 'gain': 1.3}` | float64 | 12.6690 | 2.2426 | 5.65x |
| exposure.adjust_log | `{'gain': 0.7}` | float64 | 18.8723 | 3.0575 | 6.17x |
| exposure.adjust_log | `{'gain': 0.7, 'inv': True}` | float64 | 23.4460 | 3.7350 | 6.28x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.3}` | float64 | 25.0762 | 3.6913 | 6.79x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.7, 'inv': True}` | float64 | 22.7911 | 3.7660 | 6.05x |
| exposure.rescale_intensity | `{'in_range': (0, 0.7), 'out_range': (-1, 2)}` | float64 | 13.6371 | 2.2307 | 6.11x |
