# Exposure correction with non-default parameters

1024² images; 1 C++ worker(s). Strict native dispatch. Medians of seven warmed, allocation-inclusive batches against original scikit-image. Speedup = reference/current; below one means slower. General transcendental functions may use scalar libm even in AVX2 builds.

| API | Options | Dtype | Original ms | Native ms | Speedup |
|---|---|---|---:|---:|---:|
| exposure.adjust_gamma | `{'gamma': 0.7, 'gain': 1.3}` | uint8 | 2.4927 | 0.2199 | 11.34x |
| exposure.adjust_gamma | `{'gamma': 0.5, 'gain': 1.3}` | uint8 | 2.5902 | 0.2679 | 9.67x |
| exposure.adjust_gamma | `{'gamma': 2, 'gain': 1.3}` | uint8 | 2.6440 | 0.2049 | 12.91x |
| exposure.adjust_log | `{'gain': 0.7}` | uint8 | 30.0990 | 0.2302 | 130.78x |
| exposure.adjust_log | `{'gain': 0.7, 'inv': True}` | uint8 | 37.3296 | 0.2962 | 126.03x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.3}` | uint8 | 56.3656 | 0.3828 | 147.23x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.7, 'inv': True}` | uint8 | 46.4317 | 0.3831 | 121.20x |
| exposure.rescale_intensity | `{'in_range': (0, 0.7), 'out_range': (-1, 2)}` | uint8 | 25.8159 | 3.8327 | 6.74x |
| exposure.adjust_gamma | `{'gamma': 0.7, 'gain': 1.3}` | float32 | 18.3074 | 9.9949 | 1.83x |
| exposure.adjust_gamma | `{'gamma': 0.5, 'gain': 1.3}` | float32 | 11.5108 | 3.0730 | 3.75x |
| exposure.adjust_gamma | `{'gamma': 2, 'gain': 1.3}` | float32 | 10.5270 | 3.0384 | 3.46x |
| exposure.adjust_log | `{'gain': 0.7}` | float32 | 17.2509 | 4.3905 | 3.93x |
| exposure.adjust_log | `{'gain': 0.7, 'inv': True}` | float32 | 19.0216 | 6.2717 | 3.03x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.3}` | float32 | 17.1436 | 2.2138 | 7.74x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.7, 'inv': True}` | float32 | 16.0385 | 1.8406 | 8.71x |
| exposure.rescale_intensity | `{'in_range': (0, 0.7), 'out_range': (-1, 2)}` | float32 | 15.7593 | 2.0472 | 7.70x |
| exposure.adjust_gamma | `{'gamma': 0.7, 'gain': 1.3}` | float64 | 48.9931 | 10.7677 | 4.55x |
| exposure.adjust_gamma | `{'gamma': 0.5, 'gain': 1.3}` | float64 | 24.1829 | 4.8972 | 4.94x |
| exposure.adjust_gamma | `{'gamma': 2, 'gain': 1.3}` | float64 | 30.3350 | 5.6400 | 5.38x |
| exposure.adjust_log | `{'gain': 0.7}` | float64 | 41.0136 | 7.4715 | 5.49x |
| exposure.adjust_log | `{'gain': 0.7, 'inv': True}` | float64 | 46.0426 | 7.2707 | 6.33x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.3}` | float64 | 54.4501 | 7.5469 | 7.21x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.7, 'inv': True}` | float64 | 47.0226 | 6.5769 | 7.15x |
| exposure.rescale_intensity | `{'in_range': (0, 0.7), 'out_range': (-1, 2)}` | float64 | 28.3833 | 5.1025 | 5.56x |
