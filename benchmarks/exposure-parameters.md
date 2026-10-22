# Exposure correction with non-default parameters

1024² images; 1 C++ worker(s). Strict native dispatch. Medians of seven warmed, allocation-inclusive batches against original scikit-image. Speedup = reference/current; below one means slower. General transcendental functions may use scalar libm even in AVX2 builds.

| API | Options | Dtype | Original ms | Native ms | Speedup |
|---|---|---|---:|---:|---:|
| exposure.adjust_gamma | `{'gamma': 0.7, 'gain': 1.3}` | uint8 | 2.0957 | 0.1503 | 13.94x |
| exposure.adjust_gamma | `{'gamma': 0.5, 'gain': 1.3}` | uint8 | 2.0878 | 0.1480 | 14.11x |
| exposure.adjust_gamma | `{'gamma': 2, 'gain': 1.3}` | uint8 | 2.0970 | 0.1487 | 14.10x |
| exposure.adjust_log | `{'gain': 0.7}` | uint8 | 18.2342 | 0.1498 | 121.76x |
| exposure.adjust_log | `{'gain': 0.7, 'inv': True}` | uint8 | 24.2602 | 0.1607 | 150.95x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.3}` | uint8 | 26.4444 | 0.1587 | 166.64x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.7, 'inv': True}` | uint8 | 24.5008 | 0.1612 | 151.98x |
| exposure.rescale_intensity | `{'in_range': (0, 0.7), 'out_range': (-1, 2)}` | uint8 | 14.6127 | 2.2157 | 6.60x |
| exposure.adjust_gamma | `{'gamma': 0.7, 'gain': 1.3}` | float32 | 9.7191 | 6.2800 | 1.55x |
| exposure.adjust_gamma | `{'gamma': 0.5, 'gain': 1.3}` | float32 | 3.1864 | 1.2365 | 2.58x |
| exposure.adjust_gamma | `{'gamma': 2, 'gain': 1.3}` | float32 | 3.1880 | 1.1755 | 2.71x |
| exposure.adjust_log | `{'gain': 0.7}` | float32 | 6.9175 | 2.5986 | 2.66x |
| exposure.adjust_log | `{'gain': 0.7, 'inv': True}` | float32 | 9.9424 | 3.6036 | 2.76x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.3}` | float32 | 5.5890 | 0.7555 | 7.40x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.7, 'inv': True}` | float32 | 4.2545 | 0.6482 | 6.56x |
| exposure.rescale_intensity | `{'in_range': (0, 0.7), 'out_range': (-1, 2)}` | float32 | 4.9473 | 0.3603 | 13.73x |
| exposure.adjust_gamma | `{'gamma': 0.7, 'gain': 1.3}` | float64 | 24.4488 | 7.0159 | 3.48x |
| exposure.adjust_gamma | `{'gamma': 0.5, 'gain': 1.3}` | float64 | 17.5909 | 3.9678 | 4.43x |
| exposure.adjust_gamma | `{'gamma': 2, 'gain': 1.3}` | float64 | 18.5664 | 3.9539 | 4.70x |
| exposure.adjust_log | `{'gain': 0.7}` | float64 | 25.7415 | 4.2840 | 6.01x |
| exposure.adjust_log | `{'gain': 0.7, 'inv': True}` | float64 | 34.6146 | 5.3377 | 6.48x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.3}` | float64 | 37.2620 | 5.2781 | 7.06x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.7, 'inv': True}` | float64 | 43.7095 | 5.2347 | 8.35x |
| exposure.rescale_intensity | `{'in_range': (0, 0.7), 'out_range': (-1, 2)}` | float64 | 26.5544 | 4.5420 | 5.85x |
