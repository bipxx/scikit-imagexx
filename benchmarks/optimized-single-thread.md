# Measured performance

Run: 2026-10-21T01:40:10.161531+00:00

CPU: Intel64 Family 6 Model 151 Stepping 2, GenuineIntel (20 logical CPUs). AVX2: True.

Python 3.13.14, NumPy 2.5.4, SciPy 1.18.1, scikit-image 0.26.0.

Median milliseconds for warmed, allocation-inclusive Python calls. Reference thread pools are limited to one; C++ worker counts are explicit. Small inputs use a serial path. Speedup = reference / C++; values below 1 are regressions. Input conversion is included. These synthetic workloads do not establish full-library performance.

| Algorithm | Size | Dtype | C++ workers | Reference ms | C++ ms | Speedup | Max error |
|---|---:|---|---:|---:|---:|---:|---:|
| rgb2gray | 256 | float32 | 1 | 0.0428 | 0.0115 | 3.71x | 0 |
| gaussian_sigma1 | 256 | float32 | 1 | 0.4990 | 0.1252 | 3.98x | 0 |
| sobel | 256 | float32 | 1 | 0.6075 | 0.0454 | 13.38x | 1.2e-07 |
| rgb2gray | 256 | float64 | 1 | 0.0606 | 0.0392 | 1.54x | 2.2e-16 |
| gaussian_sigma1 | 256 | float64 | 1 | 0.5767 | 0.1219 | 4.73x | 4.4e-16 |
| sobel | 256 | float64 | 1 | 0.9597 | 0.1196 | 8.03x | 2.8e-16 |
| rgb2gray | 256 | uint8 | 1 | 0.2187 | 0.0241 | 9.08x | 3.3e-16 |
| gaussian_sigma1 | 256 | uint8 | 1 | 0.6388 | 0.1742 | 3.67x | 3.3e-16 |
| sobel | 256 | uint8 | 1 | 1.0105 | 0.1540 | 6.56x | 2.2e-16 |
| threshold_otsu | 256 | uint8 | 1 | 0.0999 | 0.0166 | 6.03x | 0 |
| rgb2gray | 1024 | float32 | 1 | 1.4121 | 1.2480 | 1.13x | 0 |
| gaussian_sigma1 | 1024 | float32 | 1 | 16.7847 | 3.2440 | 5.17x | 0 |
| sobel | 1024 | float32 | 1 | 23.8960 | 2.4162 | 9.89x | 1.2e-07 |
| rgb2gray | 1024 | float64 | 1 | 4.7746 | 3.9539 | 1.21x | 2.2e-16 |
| gaussian_sigma1 | 1024 | float64 | 1 | 21.7100 | 5.5080 | 3.94x | 4.4e-16 |
| sobel | 1024 | float64 | 1 | 25.0284 | 6.3934 | 3.91x | 3.3e-16 |
| rgb2gray | 1024 | uint8 | 1 | 12.2167 | 2.4410 | 5.00x | 3.3e-16 |
| gaussian_sigma1 | 1024 | uint8 | 1 | 25.1965 | 5.8478 | 4.31x | 4.4e-16 |
| sobel | 1024 | uint8 | 1 | 36.0318 | 3.7557 | 9.59x | 2.8e-16 |
| threshold_otsu | 1024 | uint8 | 1 | 4.2339 | 0.2989 | 14.17x | 0 |
| rgb2gray | 2048 | float32 | 1 | 9.3412 | 7.2448 | 1.29x | 0 |
| gaussian_sigma1 | 2048 | float32 | 1 | 95.2817 | 18.8359 | 5.06x | 0 |
| sobel | 2048 | float32 | 1 | 85.4955 | 11.6007 | 7.37x | 1.2e-07 |
| rgb2gray | 2048 | float64 | 1 | 12.7608 | 11.1421 | 1.15x | 2.2e-16 |
| gaussian_sigma1 | 2048 | float64 | 1 | 147.8655 | 32.3140 | 4.58x | 4.4e-16 |
| sobel | 2048 | float64 | 1 | 122.1463 | 23.3438 | 5.23x | 3.3e-16 |
| rgb2gray | 2048 | uint8 | 1 | 62.0160 | 11.7024 | 5.30x | 3.3e-16 |
| gaussian_sigma1 | 2048 | uint8 | 1 | 144.7788 | 26.4932 | 5.46x | 4.4e-16 |
| sobel | 2048 | uint8 | 1 | 125.4912 | 14.2773 | 8.79x | 3.3e-16 |
| threshold_otsu | 2048 | uint8 | 1 | 18.8991 | 1.2631 | 14.96x | 0 |

## Same-run comparison with the previous binary

Previous and current implementations run in the same process, with shuffled timing order and identical inputs. Values below 1x are regressions.

| Algorithm | Size | Dtype | Workers | Previous ms | Current ms | Improvement |
|---|---:|---|---:|---:|---:|---:|
| rgb2gray | 256 | float32 | 1 | 0.0273 | 0.0115 | 2.36x |
| gaussian_sigma1 | 256 | float32 | 1 | 0.2805 | 0.1252 | 2.24x |
| sobel | 256 | float32 | 1 | 0.0689 | 0.0454 | 1.52x |
| rgb2gray | 256 | float64 | 1 | 0.0530 | 0.0392 | 1.35x |
| gaussian_sigma1 | 256 | float64 | 1 | 0.2504 | 0.1219 | 2.06x |
| sobel | 256 | float64 | 1 | 0.1563 | 0.1196 | 1.31x |
| rgb2gray | 256 | uint8 | 1 | 0.1028 | 0.0241 | 4.27x |
| gaussian_sigma1 | 256 | uint8 | 1 | 0.3719 | 0.1742 | 2.14x |
| sobel | 256 | uint8 | 1 | 0.4084 | 0.1540 | 2.65x |
| threshold_otsu | 256 | uint8 | 1 | 0.0176 | 0.0166 | 1.06x |
| rgb2gray | 1024 | float32 | 1 | 1.0166 | 1.2480 | 0.81x |
| gaussian_sigma1 | 1024 | float32 | 1 | 7.6807 | 3.2440 | 2.37x |
| sobel | 1024 | float32 | 1 | 3.2679 | 2.4162 | 1.35x |
| rgb2gray | 1024 | float64 | 1 | 5.7577 | 3.9539 | 1.46x |
| gaussian_sigma1 | 1024 | float64 | 1 | 11.5914 | 5.5080 | 2.10x |
| sobel | 1024 | float64 | 1 | 7.1328 | 6.3934 | 1.12x |
| rgb2gray | 1024 | uint8 | 1 | 3.9749 | 2.4410 | 1.63x |
| gaussian_sigma1 | 1024 | uint8 | 1 | 12.1435 | 5.8478 | 2.08x |
| sobel | 1024 | uint8 | 1 | 11.0735 | 3.7557 | 2.95x |
| threshold_otsu | 1024 | uint8 | 1 | 0.3381 | 0.2989 | 1.13x |
| rgb2gray | 2048 | float32 | 1 | 9.8018 | 7.2448 | 1.35x |
| gaussian_sigma1 | 2048 | float32 | 1 | 35.9786 | 18.8359 | 1.91x |
| sobel | 2048 | float32 | 1 | 13.1016 | 11.6007 | 1.13x |
| rgb2gray | 2048 | float64 | 1 | 14.2995 | 11.1421 | 1.28x |
| gaussian_sigma1 | 2048 | float64 | 1 | 49.9607 | 32.3140 | 1.55x |
| sobel | 2048 | float64 | 1 | 28.5653 | 23.3438 | 1.22x |
| rgb2gray | 2048 | uint8 | 1 | 17.5341 | 11.7024 | 1.50x |
| gaussian_sigma1 | 2048 | uint8 | 1 | 56.7029 | 26.4932 | 2.14x |
| sobel | 2048 | uint8 | 1 | 34.3922 | 14.2773 | 2.41x |
| threshold_otsu | 2048 | uint8 | 1 | 1.3346 | 1.2631 | 1.06x |

Raw samples, repetitions, loop counts, source hash, and environment are recorded in the companion JSON file.
