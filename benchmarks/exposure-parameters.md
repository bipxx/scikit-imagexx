# Exposure correction with non-default parameters

1024² images; 1 C++ worker(s). Strict native dispatch. Medians of seven warmed, allocation-inclusive batches against original scikit-image. Speedup = reference/current; below one means slower. General transcendental functions may use scalar libm even in AVX2 builds.

| API | Options | Dtype | Original ms | Native ms | Speedup |
|---|---|---|---:|---:|---:|
| exposure.adjust_gamma | `{'gamma': 0.7, 'gain': 1.3}` | uint8 | 2.0210 | 0.1442 | 14.01x |
| exposure.adjust_gamma | `{'gamma': 0.5, 'gain': 1.3}` | uint8 | 2.0448 | 0.1494 | 13.69x |
| exposure.adjust_gamma | `{'gamma': 2, 'gain': 1.3}` | uint8 | 2.0378 | 0.1464 | 13.92x |
| exposure.adjust_log | `{'gain': 0.7}` | uint8 | 16.6921 | 0.1422 | 117.35x |
| exposure.adjust_log | `{'gain': 0.7, 'inv': True}` | uint8 | 22.1468 | 0.1522 | 145.53x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.3}` | uint8 | 20.6381 | 0.1737 | 118.82x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.7, 'inv': True}` | uint8 | 22.2141 | 0.1533 | 144.89x |
| exposure.rescale_intensity | `{'in_range': (0, 0.7), 'out_range': (-1, 2)}` | uint8 | 16.2292 | 2.1692 | 7.48x |
| exposure.adjust_gamma | `{'gamma': 0.7, 'gain': 1.3}` | float32 | 10.7257 | 7.7570 | 1.38x |
| exposure.adjust_gamma | `{'gamma': 0.5, 'gain': 1.3}` | float32 | 3.0785 | 1.1545 | 2.67x |
| exposure.adjust_gamma | `{'gamma': 2, 'gain': 1.3}` | float32 | 2.2353 | 1.0848 | 2.06x |
| exposure.adjust_log | `{'gain': 0.7}` | float32 | 4.6090 | 2.6419 | 1.74x |
| exposure.adjust_log | `{'gain': 0.7, 'inv': True}` | float32 | 8.2394 | 3.6008 | 2.29x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.3}` | float32 | 4.2346 | 0.6453 | 6.56x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.7, 'inv': True}` | float32 | 3.3221 | 0.5833 | 5.70x |
| exposure.rescale_intensity | `{'in_range': (0, 0.7), 'out_range': (-1, 2)}` | float32 | 4.8231 | 1.1510 | 4.19x |
| exposure.adjust_gamma | `{'gamma': 0.7, 'gain': 1.3}` | float64 | 22.2456 | 6.8750 | 3.24x |
| exposure.adjust_gamma | `{'gamma': 0.5, 'gain': 1.3}` | float64 | 12.8163 | 2.7181 | 4.72x |
| exposure.adjust_gamma | `{'gamma': 2, 'gain': 1.3}` | float64 | 13.3144 | 2.4325 | 5.47x |
| exposure.adjust_log | `{'gain': 0.7}` | float64 | 17.6013 | 3.0844 | 5.71x |
| exposure.adjust_log | `{'gain': 0.7, 'inv': True}` | float64 | 21.3013 | 3.5840 | 5.94x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.3}` | float64 | 21.4532 | 3.4833 | 6.16x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.7, 'inv': True}` | float64 | 21.1071 | 3.5111 | 6.01x |
| exposure.rescale_intensity | `{'in_range': (0, 0.7), 'out_range': (-1, 2)}` | float64 | 13.7400 | 2.3142 | 5.94x |
