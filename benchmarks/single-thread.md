# Measured performance

Run: 2026-10-21T01:21:12.011141+00:00

CPU: Intel64 Family 6 Model 151 Stepping 2, GenuineIntel (20 logical CPUs). AVX2: True.

Python 3.13.14, NumPy 2.5.4, SciPy 1.18.1, scikit-image 0.26.0.

Median milliseconds for warmed, allocation-inclusive Python calls. Reference thread pools are limited to one; C++ worker counts are explicit. Small inputs use a serial path. Speedup = reference / C++; values below 1 are regressions. Input conversion is included. These synthetic workloads do not establish full-library performance.

| Algorithm | Size | Dtype | C++ workers | Reference ms | C++ ms | Speedup | Max error |
|---|---:|---|---:|---:|---:|---:|---:|
| rgb2gray | 256 | float32 | 1 | 0.0756 | 0.0363 | 2.08x | 0 |
| gaussian_sigma1 | 256 | float32 | 1 | 0.5158 | 0.2837 | 1.82x | 0 |
| sobel | 256 | float32 | 1 | 0.6910 | 0.0712 | 9.70x | 1.2e-07 |
| rgb2gray | 256 | float64 | 1 | 0.0664 | 0.0560 | 1.18x | 2.2e-16 |
| gaussian_sigma1 | 256 | float64 | 1 | 0.5728 | 0.2604 | 2.20x | 4.4e-16 |
| sobel | 256 | float64 | 1 | 0.8514 | 0.1600 | 5.32x | 2.8e-16 |
| rgb2gray | 256 | uint8 | 1 | 0.2934 | 0.1058 | 2.77x | 3.3e-16 |
| gaussian_sigma1 | 256 | uint8 | 1 | 0.6158 | 0.3251 | 1.89x | 3.3e-16 |
| sobel | 256 | uint8 | 1 | 1.0086 | 0.2418 | 4.17x | 2.8e-16 |
| threshold_otsu | 256 | uint8 | 1 | 0.1226 | 0.0194 | 6.34x | 0 |
| rgb2gray | 1024 | float32 | 1 | 1.5344 | 1.0658 | 1.44x | 0 |
| gaussian_sigma1 | 1024 | float32 | 1 | 16.9792 | 6.4591 | 2.63x | 0 |
| sobel | 1024 | float32 | 1 | 20.2095 | 4.0454 | 5.00x | 1.2e-07 |
| rgb2gray | 1024 | float64 | 1 | 4.9662 | 4.6828 | 1.06x | 2.2e-16 |
| gaussian_sigma1 | 1024 | float64 | 1 | 30.4375 | 16.2575 | 1.87x | 4.4e-16 |
| sobel | 1024 | float64 | 1 | 25.5407 | 7.4419 | 3.43x | 3.3e-16 |
| rgb2gray | 1024 | uint8 | 1 | 16.6128 | 5.2813 | 3.15x | 3.3e-16 |
| gaussian_sigma1 | 1024 | uint8 | 1 | 26.5868 | 13.3034 | 2.00x | 4.4e-16 |
| sobel | 1024 | uint8 | 1 | 32.7058 | 14.7050 | 2.22x | 3.3e-16 |
| threshold_otsu | 1024 | uint8 | 1 | 3.9588 | 0.2795 | 14.16x | 0 |
| rgb2gray | 2048 | float32 | 1 | 11.1129 | 12.9607 | 0.86x | 0 |
| gaussian_sigma1 | 2048 | float32 | 1 | 114.5660 | 33.7744 | 3.39x | 0 |
| sobel | 2048 | float32 | 1 | 84.5857 | 13.4976 | 6.27x | 1.2e-07 |
| rgb2gray | 2048 | float64 | 1 | 12.2365 | 13.4589 | 0.91x | 2.2e-16 |
| gaussian_sigma1 | 2048 | float64 | 1 | 121.4685 | 32.1466 | 3.78x | 4.4e-16 |
| sobel | 2048 | float64 | 1 | 86.1333 | 27.3131 | 3.15x | 3.3e-16 |
| rgb2gray | 2048 | uint8 | 1 | 39.6714 | 12.9223 | 3.07x | 3.3e-16 |
| gaussian_sigma1 | 2048 | uint8 | 1 | 127.0326 | 46.3933 | 2.74x | 4.4e-16 |
| sobel | 2048 | uint8 | 1 | 118.4898 | 39.8824 | 2.97x | 3.3e-16 |
| threshold_otsu | 2048 | uint8 | 1 | 13.0523 | 1.1545 | 11.31x | 0 |

Raw samples, repetitions, loop counts, source hash, and environment are recorded in the companion JSON file.
