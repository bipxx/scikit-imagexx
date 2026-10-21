# Exposure correction with non-default parameters

1024² images; 1 C++ worker(s). Strict native dispatch. Medians of seven warmed, allocation-inclusive batches against original scikit-image. Speedup = reference/current; below one means slower. General transcendental functions may use scalar libm even in AVX2 builds.

| API | Options | Dtype | Original ms | Native ms | Speedup |
|---|---|---|---:|---:|---:|
| exposure.adjust_gamma | `{'gamma': 0.7, 'gain': 1.3}` | uint8 | 2.3696 | 0.1706 | 13.89x |
| exposure.adjust_gamma | `{'gamma': 0.5, 'gain': 1.3}` | uint8 | 2.6353 | 0.2119 | 12.44x |
| exposure.adjust_gamma | `{'gamma': 2, 'gain': 1.3}` | uint8 | 2.5015 | 0.1754 | 14.26x |
| exposure.adjust_log | `{'gain': 0.7}` | uint8 | 27.5024 | 0.1813 | 151.71x |
| exposure.adjust_log | `{'gain': 0.7, 'inv': True}` | uint8 | 28.0254 | 0.1727 | 162.25x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.3}` | uint8 | 27.5301 | 0.1531 | 179.81x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.7, 'inv': True}` | uint8 | 29.1329 | 0.1823 | 159.80x |
| exposure.rescale_intensity | `{'in_range': (0, 0.7), 'out_range': (-1, 2)}` | uint8 | 21.1698 | 3.0979 | 6.83x |
| exposure.adjust_gamma | `{'gamma': 0.7, 'gain': 1.3}` | float32 | 11.3782 | 6.5199 | 1.75x |
| exposure.adjust_gamma | `{'gamma': 0.5, 'gain': 1.3}` | float32 | 5.3619 | 1.9885 | 2.70x |
| exposure.adjust_gamma | `{'gamma': 2, 'gain': 1.3}` | float32 | 5.9534 | 1.8367 | 3.24x |
| exposure.adjust_log | `{'gain': 0.7}` | float32 | 6.2007 | 2.5670 | 2.42x |
| exposure.adjust_log | `{'gain': 0.7, 'inv': True}` | float32 | 9.5732 | 3.5099 | 2.73x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.3}` | float32 | 4.2692 | 0.5859 | 7.29x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.7, 'inv': True}` | float32 | 4.4174 | 0.6525 | 6.77x |
| exposure.rescale_intensity | `{'in_range': (0, 0.7), 'out_range': (-1, 2)}` | float32 | 5.9615 | 1.4108 | 4.23x |
| exposure.adjust_gamma | `{'gamma': 0.7, 'gain': 1.3}` | float64 | 24.6447 | 7.0134 | 3.51x |
| exposure.adjust_gamma | `{'gamma': 0.5, 'gain': 1.3}` | float64 | 14.1875 | 3.1544 | 4.50x |
| exposure.adjust_gamma | `{'gamma': 2, 'gain': 1.3}` | float64 | 14.8158 | 3.0762 | 4.82x |
| exposure.adjust_log | `{'gain': 0.7}` | float64 | 19.9898 | 3.8615 | 5.18x |
| exposure.adjust_log | `{'gain': 0.7, 'inv': True}` | float64 | 25.9078 | 3.6377 | 7.12x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.3}` | float64 | 27.3233 | 3.9206 | 6.97x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.7, 'inv': True}` | float64 | 29.2614 | 4.1569 | 7.04x |
| exposure.rescale_intensity | `{'in_range': (0, 0.7), 'out_range': (-1, 2)}` | float64 | 23.8757 | 4.3386 | 5.50x |
