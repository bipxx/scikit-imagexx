# Measured performance

Run: 2026-10-21T01:16:27.393769+00:00

CPU: Intel64 Family 6 Model 151 Stepping 2, GenuineIntel (20 logical CPUs). AVX2: True.

Python 3.13.14, NumPy 2.5.4, SciPy 1.18.1, scikit-image 0.26.0.

Median milliseconds for warmed, allocation-inclusive Python calls. Reference thread pools are limited to one; C++ worker counts are explicit. Small inputs use a serial path. Speedup = reference / C++; values below 1 are regressions. Input conversion is included. These synthetic workloads do not establish full-library performance.

| Algorithm | Size | Dtype | C++ workers | Reference ms | C++ ms | Speedup | Max error |
|---|---:|---|---:|---:|---:|---:|---:|
| rgb2gray | 256 | float32 | 1 | 0.0441 | 0.0274 | 1.61x | 0 |
| rgb2gray | 256 | float32 | 4 | 0.0472 | 0.0264 | 1.79x | 0 |
| rgb2gray | 256 | float32 | 8 | 0.0446 | 0.0263 | 1.69x | 0 |
| gaussian_sigma1 | 256 | float32 | 1 | 0.4921 | 0.2760 | 1.78x | 0 |
| gaussian_sigma1 | 256 | float32 | 4 | 0.4921 | 0.2794 | 1.76x | 0 |
| gaussian_sigma1 | 256 | float32 | 8 | 0.5040 | 0.2794 | 1.80x | 0 |
| sobel | 256 | float32 | 1 | 0.6327 | 0.0697 | 9.08x | 1.2e-07 |
| sobel | 256 | float32 | 4 | 0.6728 | 0.0728 | 9.24x | 1.2e-07 |
| sobel | 256 | float32 | 8 | 0.6406 | 0.0699 | 9.17x | 1.2e-07 |
| rgb2gray | 256 | float64 | 1 | 0.0608 | 0.0539 | 1.13x | 2.2e-16 |
| rgb2gray | 256 | float64 | 4 | 0.0607 | 0.0544 | 1.12x | 2.2e-16 |
| rgb2gray | 256 | float64 | 8 | 0.0682 | 0.0540 | 1.26x | 2.2e-16 |
| gaussian_sigma1 | 256 | float64 | 1 | 0.8471 | 0.3783 | 2.24x | 4.4e-16 |
| gaussian_sigma1 | 256 | float64 | 4 | 0.7273 | 0.2891 | 2.52x | 4.4e-16 |
| gaussian_sigma1 | 256 | float64 | 8 | 0.8071 | 0.4119 | 1.96x | 4.4e-16 |
| sobel | 256 | float64 | 1 | 1.3502 | 0.1739 | 7.76x | 2.8e-16 |
| sobel | 256 | float64 | 4 | 0.9139 | 0.1583 | 5.77x | 2.8e-16 |
| sobel | 256 | float64 | 8 | 0.9355 | 0.1694 | 5.52x | 2.8e-16 |
| rgb2gray | 256 | uint8 | 1 | 0.3001 | 0.1663 | 1.80x | 3.3e-16 |
| rgb2gray | 256 | uint8 | 4 | 0.2996 | 0.1092 | 2.74x | 3.3e-16 |
| rgb2gray | 256 | uint8 | 8 | 0.2858 | 0.1201 | 2.38x | 3.3e-16 |
| gaussian_sigma1 | 256 | uint8 | 1 | 0.7639 | 0.3815 | 2.00x | 3.3e-16 |
| gaussian_sigma1 | 256 | uint8 | 4 | 0.7657 | 0.3875 | 1.98x | 3.3e-16 |
| gaussian_sigma1 | 256 | uint8 | 8 | 0.7908 | 0.4503 | 1.76x | 3.3e-16 |
| sobel | 256 | uint8 | 1 | 1.2821 | 0.2960 | 4.33x | 2.8e-16 |
| sobel | 256 | uint8 | 4 | 0.9356 | 0.2210 | 4.23x | 2.8e-16 |
| sobel | 256 | uint8 | 8 | 0.8361 | 0.2236 | 3.74x | 2.8e-16 |
| threshold_otsu | 256 | uint8 | 1 | 0.1203 | 0.0184 | 6.54x | 0 |
| threshold_otsu | 256 | uint8 | 4 | 0.1071 | 0.0191 | 5.62x | 0 |
| threshold_otsu | 256 | uint8 | 8 | 0.1090 | 0.0181 | 6.03x | 0 |
| rgb2gray | 1024 | float32 | 1 | 2.6464 | 3.3314 | 0.79x | 0 |
| rgb2gray | 1024 | float32 | 4 | 2.3109 | 1.5726 | 1.47x | 0 |
| rgb2gray | 1024 | float32 | 8 | 2.2946 | 0.9006 | 2.55x | 0 |
| gaussian_sigma1 | 1024 | float32 | 1 | 19.2147 | 6.9187 | 2.78x | 0 |
| gaussian_sigma1 | 1024 | float32 | 4 | 19.7144 | 5.0140 | 3.93x | 0 |
| gaussian_sigma1 | 1024 | float32 | 8 | 26.5182 | 5.3644 | 4.94x | 0 |
| sobel | 1024 | float32 | 1 | 23.3042 | 1.7461 | 13.35x | 1.2e-07 |
| sobel | 1024 | float32 | 4 | 16.9864 | 1.2827 | 13.24x | 1.2e-07 |
| sobel | 1024 | float32 | 8 | 14.6074 | 0.5672 | 25.75x | 1.2e-07 |
| rgb2gray | 1024 | float64 | 1 | 4.8440 | 5.0576 | 0.96x | 2.2e-16 |
| rgb2gray | 1024 | float64 | 4 | 5.2882 | 5.0537 | 1.05x | 2.2e-16 |
| rgb2gray | 1024 | float64 | 8 | 5.2101 | 4.4741 | 1.16x | 2.2e-16 |
| gaussian_sigma1 | 1024 | float64 | 1 | 24.7802 | 11.4421 | 2.17x | 4.4e-16 |
| gaussian_sigma1 | 1024 | float64 | 4 | 24.9785 | 8.3111 | 3.01x | 4.4e-16 |
| gaussian_sigma1 | 1024 | float64 | 8 | 25.5106 | 6.9861 | 3.65x | 4.4e-16 |
| sobel | 1024 | float64 | 1 | 31.4748 | 6.4511 | 4.88x | 3.3e-16 |
| sobel | 1024 | float64 | 4 | 27.8445 | 4.5039 | 6.18x | 3.3e-16 |
| sobel | 1024 | float64 | 8 | 27.4800 | 3.8587 | 7.12x | 3.3e-16 |
| rgb2gray | 1024 | uint8 | 1 | 11.8909 | 4.6905 | 2.54x | 3.3e-16 |
| rgb2gray | 1024 | uint8 | 4 | 15.1961 | 4.2195 | 3.60x | 3.3e-16 |
| rgb2gray | 1024 | uint8 | 8 | 13.4216 | 3.6523 | 3.67x | 3.3e-16 |
| gaussian_sigma1 | 1024 | uint8 | 1 | 30.7275 | 17.0064 | 1.81x | 4.4e-16 |
| gaussian_sigma1 | 1024 | uint8 | 4 | 30.7481 | 11.5473 | 2.66x | 4.4e-16 |
| gaussian_sigma1 | 1024 | uint8 | 8 | 31.9456 | 10.3787 | 3.08x | 4.4e-16 |
| sobel | 1024 | uint8 | 1 | 28.4950 | 8.8101 | 3.23x | 3.3e-16 |
| sobel | 1024 | uint8 | 4 | 32.5403 | 7.2984 | 4.46x | 3.3e-16 |
| sobel | 1024 | uint8 | 8 | 30.8333 | 7.0069 | 4.40x | 3.3e-16 |
| threshold_otsu | 1024 | uint8 | 1 | 6.4950 | 0.2830 | 22.95x | 0 |
| threshold_otsu | 1024 | uint8 | 4 | 5.7005 | 0.1444 | 39.47x | 0 |
| threshold_otsu | 1024 | uint8 | 8 | 5.8125 | 0.1147 | 50.68x | 0 |
| rgb2gray | 2048 | float32 | 1 | 11.2585 | 12.4485 | 0.90x | 0 |
| rgb2gray | 2048 | float32 | 4 | 13.9517 | 10.1279 | 1.38x | 0 |
| rgb2gray | 2048 | float32 | 8 | 14.0189 | 9.2402 | 1.52x | 0 |
| gaussian_sigma1 | 2048 | float32 | 1 | 99.1272 | 30.1017 | 3.29x | 0 |
| gaussian_sigma1 | 2048 | float32 | 4 | 113.1730 | 17.2033 | 6.58x | 0 |
| gaussian_sigma1 | 2048 | float32 | 8 | 96.5256 | 14.9921 | 6.44x | 0 |
| sobel | 2048 | float32 | 1 | 102.2230 | 15.0847 | 6.78x | 1.2e-07 |
| sobel | 2048 | float32 | 4 | 85.6972 | 8.3144 | 10.31x | 1.2e-07 |
| sobel | 2048 | float32 | 8 | 96.0480 | 7.9495 | 12.08x | 1.2e-07 |
| rgb2gray | 2048 | float64 | 1 | 23.6577 | 31.7530 | 0.75x | 2.2e-16 |
| rgb2gray | 2048 | float64 | 4 | 23.5481 | 20.3783 | 1.16x | 2.2e-16 |
| rgb2gray | 2048 | float64 | 8 | 32.9082 | 18.2488 | 1.80x | 2.2e-16 |
| gaussian_sigma1 | 2048 | float64 | 1 | 158.5317 | 51.1307 | 3.10x | 4.4e-16 |
| gaussian_sigma1 | 2048 | float64 | 4 | 162.5353 | 31.5458 | 5.15x | 4.4e-16 |
| gaussian_sigma1 | 2048 | float64 | 8 | 151.5895 | 30.2813 | 5.01x | 4.4e-16 |
| sobel | 2048 | float64 | 1 | 121.8729 | 25.1491 | 4.85x | 3.3e-16 |
| sobel | 2048 | float64 | 4 | 139.0581 | 16.9150 | 8.22x | 3.3e-16 |
| sobel | 2048 | float64 | 8 | 146.5694 | 16.5448 | 8.86x | 3.3e-16 |
| rgb2gray | 2048 | uint8 | 1 | 64.9683 | 18.1823 | 3.57x | 3.3e-16 |
| rgb2gray | 2048 | uint8 | 4 | 59.9184 | 12.6962 | 4.72x | 3.3e-16 |
| rgb2gray | 2048 | uint8 | 8 | 48.0280 | 10.7075 | 4.49x | 3.3e-16 |
| gaussian_sigma1 | 2048 | uint8 | 1 | 136.2948 | 49.2144 | 2.77x | 4.4e-16 |
| gaussian_sigma1 | 2048 | uint8 | 4 | 116.4850 | 31.0712 | 3.75x | 4.4e-16 |
| gaussian_sigma1 | 2048 | uint8 | 8 | 112.3663 | 29.9381 | 3.75x | 4.4e-16 |
| sobel | 2048 | uint8 | 1 | 104.7019 | 35.6088 | 2.94x | 3.3e-16 |
| sobel | 2048 | uint8 | 4 | 98.6352 | 22.6123 | 4.36x | 3.3e-16 |
| sobel | 2048 | uint8 | 8 | 110.6243 | 20.3026 | 5.45x | 3.3e-16 |
| threshold_otsu | 2048 | uint8 | 1 | 12.3294 | 1.0804 | 11.41x | 0 |
| threshold_otsu | 2048 | uint8 | 4 | 12.4811 | 0.3470 | 35.97x | 0 |
| threshold_otsu | 2048 | uint8 | 8 | 15.9809 | 0.3359 | 47.58x | 0 |

Raw samples, repetitions, loop counts, source hash, and environment are recorded in the companion JSON file.
