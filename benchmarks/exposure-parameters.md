# Exposure correction with non-default parameters

1024² images; 1 C++ worker(s). Strict native dispatch. Medians of seven warmed, allocation-inclusive batches against original scikit-image. Speedup = reference/current; below one means slower. General transcendental functions may use scalar libm even in AVX2 builds.

| API | Options | Dtype | Original ms | Native ms | Speedup |
|---|---|---|---:|---:|---:|
| exposure.adjust_gamma | `{'gamma': 0.7, 'gain': 1.3}` | uint8 | 2.0317 | 0.1469 | 13.83x |
| exposure.adjust_gamma | `{'gamma': 0.5, 'gain': 1.3}` | uint8 | 2.1290 | 0.1475 | 14.43x |
| exposure.adjust_gamma | `{'gamma': 2, 'gain': 1.3}` | uint8 | 2.1112 | 0.1512 | 13.96x |
| exposure.adjust_log | `{'gain': 0.7}` | uint8 | 18.7931 | 0.1626 | 115.54x |
| exposure.adjust_log | `{'gain': 0.7, 'inv': True}` | uint8 | 20.9172 | 0.1582 | 132.20x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.3}` | uint8 | 19.7771 | 0.1442 | 137.13x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.7, 'inv': True}` | uint8 | 23.0183 | 0.1482 | 155.34x |
| exposure.rescale_intensity | `{'in_range': (0, 0.7), 'out_range': (-1, 2)}` | uint8 | 13.6132 | 2.1113 | 6.45x |
| exposure.adjust_gamma | `{'gamma': 0.7, 'gain': 1.3}` | float32 | 8.8336 | 6.3254 | 1.40x |
| exposure.adjust_gamma | `{'gamma': 0.5, 'gain': 1.3}` | float32 | 2.9254 | 1.3613 | 2.15x |
| exposure.adjust_gamma | `{'gamma': 2, 'gain': 1.3}` | float32 | 2.9295 | 1.0572 | 2.77x |
| exposure.adjust_log | `{'gain': 0.7}` | float32 | 4.6757 | 2.4340 | 1.92x |
| exposure.adjust_log | `{'gain': 0.7, 'inv': True}` | float32 | 8.1306 | 3.3518 | 2.43x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.3}` | float32 | 3.2495 | 0.5694 | 5.71x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.7, 'inv': True}` | float32 | 3.4659 | 0.5580 | 6.21x |
| exposure.rescale_intensity | `{'in_range': (0, 0.7), 'out_range': (-1, 2)}` | float32 | 4.3237 | 0.9489 | 4.56x |
| exposure.adjust_gamma | `{'gamma': 0.7, 'gain': 1.3}` | float64 | 20.1225 | 6.5682 | 3.06x |
| exposure.adjust_gamma | `{'gamma': 0.5, 'gain': 1.3}` | float64 | 13.0711 | 2.5481 | 5.13x |
| exposure.adjust_gamma | `{'gamma': 2, 'gain': 1.3}` | float64 | 13.9311 | 3.0035 | 4.64x |
| exposure.adjust_log | `{'gain': 0.7}` | float64 | 21.0690 | 3.4538 | 6.10x |
| exposure.adjust_log | `{'gain': 0.7, 'inv': True}` | float64 | 22.1633 | 3.2369 | 6.85x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.3}` | float64 | 21.4413 | 3.3219 | 6.45x |
| exposure.adjust_sigmoid | `{'gain': 7, 'cutoff': 0.7, 'inv': True}` | float64 | 21.0662 | 3.4341 | 6.13x |
| exposure.rescale_intensity | `{'in_range': (0, 0.7), 'out_range': (-1, 2)}` | float64 | 13.8544 | 2.4507 | 5.65x |
