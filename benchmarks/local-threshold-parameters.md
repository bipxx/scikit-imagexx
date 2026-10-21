# Local threshold performance

Reference: original scikit-image 0.26.0. Reference thread pools use one worker; native workers are explicit. Each case checks dtype, shape, warnings and numerical parity with fallback disabled. Timings include dispatch, layout copying, reflected padding, both integral images, local statistics and output allocation. Both scalar and NumPy int64 window metadata are covered, as are strong float64 weights and ranges. All raw samples and slower cases are retained in the JSON.

| API | Shape | Input | Workload | Workers | Original ms | C++ ms | Speedup |
|---|---|---|---|---:|---:|---:|---:|
| filters.threshold_niblack | 512×512 | uint8 | window-15 | 1 | 9.4867 | 2.6189 | 3.62x |
| filters.threshold_niblack | 512×512 | uint8 | window-15 | 4 | 9.9114 | 1.4891 | 6.66x |
| filters.threshold_sauvola | 512×512 | uint8 | window-15 | 1 | 11.0669 | 3.7919 | 2.92x |
| filters.threshold_sauvola | 512×512 | uint8 | window-15 | 4 | 10.0246 | 0.9614 | 10.43x |
| filters.threshold_niblack | 512×512 | uint8 | window-3 | 1 | 10.3022 | 2.3177 | 4.45x |
| filters.threshold_niblack | 512×512 | uint8 | window-3 | 4 | 9.6827 | 1.2560 | 7.71x |
| filters.threshold_sauvola | 512×512 | uint8 | window-3 | 1 | 13.8421 | 4.5452 | 3.05x |
| filters.threshold_sauvola | 512×512 | uint8 | window-3 | 4 | 11.6079 | 1.6633 | 6.98x |
| filters.threshold_niblack | 512×512 | uint8 | window-31 | 1 | 11.4096 | 2.8085 | 4.06x |
| filters.threshold_niblack | 512×512 | uint8 | window-31 | 4 | 11.7607 | 1.3199 | 8.91x |
| filters.threshold_sauvola | 512×512 | uint8 | window-31 | 1 | 11.1384 | 3.6985 | 3.01x |
| filters.threshold_sauvola | 512×512 | uint8 | window-31 | 4 | 11.6155 | 1.4038 | 8.27x |
| filters.threshold_niblack | 512×512 | uint8 | window-127 | 1 | 14.5010 | 3.4687 | 4.18x |
| filters.threshold_niblack | 512×512 | uint8 | window-127 | 4 | 17.7212 | 2.8882 | 6.14x |
| filters.threshold_sauvola | 512×512 | uint8 | window-127 | 1 | 18.6836 | 5.8008 | 3.22x |
| filters.threshold_sauvola | 512×512 | uint8 | window-127 | 4 | 16.4905 | 2.8695 | 5.75x |
| filters.threshold_niblack | 512×512 | uint8 | window-(3, 31) | 1 | 9.9479 | 2.4849 | 4.00x |
| filters.threshold_niblack | 512×512 | uint8 | window-(3, 31) | 4 | 9.6708 | 1.3326 | 7.26x |
| filters.threshold_sauvola | 512×512 | uint8 | window-(3, 31) | 1 | 10.9440 | 4.1920 | 2.61x |
| filters.threshold_sauvola | 512×512 | uint8 | window-(3, 31) | 4 | 9.2277 | 1.0923 | 8.45x |
| filters.threshold_niblack | 512×512 | uint8 | reversed | 1 | 10.3655 | 2.7330 | 3.79x |
| filters.threshold_niblack | 512×512 | uint8 | reversed | 4 | 8.3271 | 1.0597 | 7.86x |
| filters.threshold_sauvola | 512×512 | uint8 | reversed | 1 | 11.7029 | 3.5147 | 3.33x |
| filters.threshold_sauvola | 512×512 | uint8 | reversed | 4 | 9.1355 | 1.0638 | 8.59x |
| filters.threshold_niblack | 512×512 | uint8 | strided | 1 | 9.4229 | 2.4279 | 3.88x |
| filters.threshold_niblack | 512×512 | uint8 | strided | 4 | 11.5783 | 1.2556 | 9.22x |
| filters.threshold_sauvola | 512×512 | uint8 | strided | 1 | 15.4616 | 3.9613 | 3.90x |
| filters.threshold_sauvola | 512×512 | uint8 | strided | 4 | 11.0000 | 1.7023 | 6.46x |
| filters.threshold_niblack | 512×512 | uint8 | fortran | 1 | 20.2364 | 3.0331 | 6.67x |
| filters.threshold_niblack | 512×512 | uint8 | fortran | 4 | 14.3369 | 1.0719 | 13.37x |
| filters.threshold_sauvola | 512×512 | uint8 | fortran | 1 | 16.6773 | 4.6794 | 3.56x |
| filters.threshold_sauvola | 512×512 | uint8 | fortran | 4 | 14.0153 | 1.0174 | 13.78x |
| filters.threshold_niblack | 1×512×512 | uint8 | volume | 1 | 20.6162 | 5.2504 | 3.93x |
| filters.threshold_niblack | 1×512×512 | uint8 | volume | 4 | 18.0730 | 2.5824 | 7.00x |
| filters.threshold_sauvola | 1×512×512 | uint8 | volume | 1 | 19.6960 | 5.3515 | 3.68x |
| filters.threshold_sauvola | 1×512×512 | uint8 | volume | 4 | 17.8445 | 2.9246 | 6.10x |
| filters.threshold_niblack | 512×512 | uint16 | window-15 | 1 | 7.6387 | 2.0753 | 3.68x |
| filters.threshold_niblack | 512×512 | uint16 | window-15 | 4 | 8.0411 | 1.0284 | 7.82x |
| filters.threshold_sauvola | 512×512 | uint16 | window-15 | 1 | 8.3840 | 2.1738 | 3.86x |
| filters.threshold_sauvola | 512×512 | uint16 | window-15 | 4 | 11.4584 | 1.7412 | 6.58x |
| filters.threshold_niblack | 512×512 | uint16 | window-3 | 1 | 7.9591 | 2.1296 | 3.74x |
| filters.threshold_niblack | 512×512 | uint16 | window-3 | 4 | 15.2455 | 1.9635 | 7.76x |
| filters.threshold_sauvola | 512×512 | uint16 | window-3 | 1 | 19.7995 | 4.6664 | 4.24x |
| filters.threshold_sauvola | 512×512 | uint16 | window-3 | 4 | 25.3527 | 2.3455 | 10.81x |
| filters.threshold_niblack | 512×512 | uint16 | window-31 | 1 | 32.0275 | 7.7637 | 4.13x |
| filters.threshold_niblack | 512×512 | uint16 | window-31 | 4 | 23.5297 | 1.7731 | 13.27x |
| filters.threshold_sauvola | 512×512 | uint16 | window-31 | 1 | 20.1463 | 5.0907 | 3.96x |
| filters.threshold_sauvola | 512×512 | uint16 | window-31 | 4 | 22.9920 | 2.4214 | 9.50x |
| filters.threshold_niblack | 512×512 | uint16 | window-127 | 1 | 13.8856 | 4.6793 | 2.97x |
| filters.threshold_niblack | 512×512 | uint16 | window-127 | 4 | 10.2206 | 1.1354 | 9.00x |
| filters.threshold_sauvola | 512×512 | uint16 | window-127 | 1 | 11.4839 | 4.2169 | 2.72x |
| filters.threshold_sauvola | 512×512 | uint16 | window-127 | 4 | 10.5774 | 1.2216 | 8.66x |
| filters.threshold_niblack | 512×512 | uint16 | window-(3, 31) | 1 | 7.6860 | 1.9254 | 3.99x |
| filters.threshold_niblack | 512×512 | uint16 | window-(3, 31) | 4 | 10.5607 | 1.8011 | 5.86x |
| filters.threshold_sauvola | 512×512 | uint16 | window-(3, 31) | 1 | 11.4128 | 2.6423 | 4.32x |
| filters.threshold_sauvola | 512×512 | uint16 | window-(3, 31) | 4 | 12.9663 | 1.9163 | 6.77x |
| filters.threshold_niblack | 512×512 | uint16 | reversed | 1 | 13.3278 | 4.7859 | 2.78x |
| filters.threshold_niblack | 512×512 | uint16 | reversed | 4 | 10.3125 | 1.9839 | 5.20x |
| filters.threshold_sauvola | 512×512 | uint16 | reversed | 1 | 16.8634 | 5.2320 | 3.22x |
| filters.threshold_sauvola | 512×512 | uint16 | reversed | 4 | 11.6317 | 1.6478 | 7.06x |
| filters.threshold_niblack | 512×512 | uint16 | strided | 1 | 8.1520 | 3.2583 | 2.50x |
| filters.threshold_niblack | 512×512 | uint16 | strided | 4 | 8.3307 | 1.0411 | 8.00x |
| filters.threshold_sauvola | 512×512 | uint16 | strided | 1 | 11.0282 | 3.4791 | 3.17x |
| filters.threshold_sauvola | 512×512 | uint16 | strided | 4 | 10.4570 | 1.0628 | 9.84x |
| filters.threshold_niblack | 512×512 | uint16 | fortran | 1 | 18.9321 | 3.6450 | 5.19x |
| filters.threshold_niblack | 512×512 | uint16 | fortran | 4 | 13.0609 | 0.9050 | 14.43x |
| filters.threshold_sauvola | 512×512 | uint16 | fortran | 1 | 15.4919 | 3.4691 | 4.47x |
| filters.threshold_sauvola | 512×512 | uint16 | fortran | 4 | 13.3604 | 1.2317 | 10.85x |
| filters.threshold_niblack | 1×512×512 | uint16 | volume | 1 | 18.1452 | 5.1320 | 3.54x |
| filters.threshold_niblack | 1×512×512 | uint16 | volume | 4 | 19.8402 | 2.0948 | 9.47x |
| filters.threshold_sauvola | 1×512×512 | uint16 | volume | 1 | 18.6929 | 5.3086 | 3.52x |
| filters.threshold_sauvola | 1×512×512 | uint16 | volume | 4 | 20.4477 | 2.3843 | 8.58x |
| filters.threshold_niblack | 512×512 | int16 | window-15 | 1 | 8.0953 | 3.0110 | 2.69x |
| filters.threshold_niblack | 512×512 | int16 | window-15 | 4 | 8.3589 | 0.6392 | 13.08x |
| filters.threshold_sauvola | 512×512 | int16 | window-15 | 1 | 8.4407 | 2.3922 | 3.53x |
| filters.threshold_sauvola | 512×512 | int16 | window-15 | 4 | 8.4694 | 0.9810 | 8.63x |
| filters.threshold_niblack | 512×512 | int16 | window-3 | 1 | 8.7167 | 2.8296 | 3.08x |
| filters.threshold_niblack | 512×512 | int16 | window-3 | 4 | 7.7695 | 0.7153 | 10.86x |
| filters.threshold_sauvola | 512×512 | int16 | window-3 | 1 | 8.7980 | 2.6157 | 3.36x |
| filters.threshold_sauvola | 512×512 | int16 | window-3 | 4 | 9.1639 | 1.1285 | 8.12x |
| filters.threshold_niblack | 512×512 | int16 | window-31 | 1 | 10.9816 | 2.7296 | 4.02x |
| filters.threshold_niblack | 512×512 | int16 | window-31 | 4 | 8.8289 | 1.4668 | 6.02x |
| filters.threshold_sauvola | 512×512 | int16 | window-31 | 1 | 9.2603 | 2.4636 | 3.76x |
| filters.threshold_sauvola | 512×512 | int16 | window-31 | 4 | 9.8879 | 1.4500 | 6.82x |
| filters.threshold_niblack | 512×512 | int16 | window-127 | 1 | 10.7509 | 4.1120 | 2.61x |
| filters.threshold_niblack | 512×512 | int16 | window-127 | 4 | 11.7502 | 1.6996 | 6.91x |
| filters.threshold_sauvola | 512×512 | int16 | window-127 | 1 | 12.5671 | 4.6591 | 2.70x |
| filters.threshold_sauvola | 512×512 | int16 | window-127 | 4 | 10.2878 | 1.2769 | 8.06x |
| filters.threshold_niblack | 512×512 | int16 | window-(3, 31) | 1 | 9.0270 | 2.5284 | 3.57x |
| filters.threshold_niblack | 512×512 | int16 | window-(3, 31) | 4 | 8.6717 | 1.4021 | 6.18x |
| filters.threshold_sauvola | 512×512 | int16 | window-(3, 31) | 1 | 8.4885 | 2.1484 | 3.95x |
| filters.threshold_sauvola | 512×512 | int16 | window-(3, 31) | 4 | 8.7786 | 0.9786 | 8.97x |
| filters.threshold_niblack | 512×512 | int16 | reversed | 1 | 8.3200 | 3.3060 | 2.52x |
| filters.threshold_niblack | 512×512 | int16 | reversed | 4 | 8.0931 | 0.7374 | 10.98x |
| filters.threshold_sauvola | 512×512 | int16 | reversed | 1 | 8.5876 | 2.2413 | 3.83x |
| filters.threshold_sauvola | 512×512 | int16 | reversed | 4 | 9.7638 | 1.0009 | 9.75x |
| filters.threshold_niblack | 512×512 | int16 | strided | 1 | 8.3768 | 2.1063 | 3.98x |
| filters.threshold_niblack | 512×512 | int16 | strided | 4 | 7.7181 | 0.9677 | 7.98x |
| filters.threshold_sauvola | 512×512 | int16 | strided | 1 | 9.5363 | 2.3249 | 4.10x |
| filters.threshold_sauvola | 512×512 | int16 | strided | 4 | 15.3256 | 2.1723 | 7.06x |
| filters.threshold_niblack | 512×512 | int16 | fortran | 1 | 21.9616 | 3.5650 | 6.16x |
| filters.threshold_niblack | 512×512 | int16 | fortran | 4 | 21.3969 | 2.1490 | 9.96x |
| filters.threshold_sauvola | 512×512 | int16 | fortran | 1 | 26.9446 | 6.3115 | 4.27x |
| filters.threshold_sauvola | 512×512 | int16 | fortran | 4 | 25.9908 | 1.9206 | 13.53x |
| filters.threshold_niblack | 1×512×512 | int16 | volume | 1 | 25.7388 | 7.5024 | 3.43x |
| filters.threshold_niblack | 1×512×512 | int16 | volume | 4 | 18.1461 | 1.9514 | 9.30x |
| filters.threshold_sauvola | 1×512×512 | int16 | volume | 1 | 19.0057 | 4.9573 | 3.83x |
| filters.threshold_sauvola | 1×512×512 | int16 | volume | 4 | 17.4893 | 2.5057 | 6.98x |
| filters.threshold_niblack | 512×512 | float32 | window-15 | 1 | 7.8991 | 2.1708 | 3.64x |
| filters.threshold_niblack | 512×512 | float32 | window-15 | 4 | 7.9040 | 0.6343 | 12.46x |
| filters.threshold_sauvola | 512×512 | float32 | window-15 | 1 | 7.3572 | 2.0310 | 3.62x |
| filters.threshold_sauvola | 512×512 | float32 | window-15 | 4 | 7.0102 | 0.8659 | 8.10x |
| filters.threshold_niblack | 512×512 | float32 | window-3 | 1 | 5.8556 | 2.0360 | 2.88x |
| filters.threshold_niblack | 512×512 | float32 | window-3 | 4 | 6.1015 | 1.0019 | 6.09x |
| filters.threshold_sauvola | 512×512 | float32 | window-3 | 1 | 6.8883 | 1.8761 | 3.67x |
| filters.threshold_sauvola | 512×512 | float32 | window-3 | 4 | 7.1504 | 0.7911 | 9.04x |
| filters.threshold_niblack | 512×512 | float32 | window-31 | 1 | 7.6813 | 2.3358 | 3.29x |
| filters.threshold_niblack | 512×512 | float32 | window-31 | 4 | 8.2936 | 1.2258 | 6.77x |
| filters.threshold_sauvola | 512×512 | float32 | window-31 | 1 | 6.7473 | 2.8657 | 2.35x |
| filters.threshold_sauvola | 512×512 | float32 | window-31 | 4 | 7.3044 | 0.9358 | 7.81x |
| filters.threshold_niblack | 512×512 | float32 | window-127 | 1 | 13.0227 | 6.2256 | 2.09x |
| filters.threshold_niblack | 512×512 | float32 | window-127 | 4 | 8.8703 | 1.0622 | 8.35x |
| filters.threshold_sauvola | 512×512 | float32 | window-127 | 1 | 11.3759 | 4.5306 | 2.51x |
| filters.threshold_sauvola | 512×512 | float32 | window-127 | 4 | 9.5611 | 1.4531 | 6.58x |
| filters.threshold_niblack | 512×512 | float32 | window-(3, 31) | 1 | 6.9223 | 2.0018 | 3.46x |
| filters.threshold_niblack | 512×512 | float32 | window-(3, 31) | 4 | 7.2593 | 2.8923 | 2.51x |
| filters.threshold_sauvola | 512×512 | float32 | window-(3, 31) | 1 | 7.2213 | 1.9245 | 3.75x |
| filters.threshold_sauvola | 512×512 | float32 | window-(3, 31) | 4 | 8.1787 | 1.9506 | 4.19x |
| filters.threshold_niblack | 512×512 | float32 | reversed | 1 | 7.1487 | 2.7034 | 2.64x |
| filters.threshold_niblack | 512×512 | float32 | reversed | 4 | 7.1837 | 1.2988 | 5.53x |
| filters.threshold_sauvola | 512×512 | float32 | reversed | 1 | 6.5954 | 2.5216 | 2.62x |
| filters.threshold_sauvola | 512×512 | float32 | reversed | 4 | 6.9528 | 0.9366 | 7.42x |
| filters.threshold_niblack | 512×512 | float32 | strided | 1 | 9.7639 | 2.7703 | 3.52x |
| filters.threshold_niblack | 512×512 | float32 | strided | 4 | 7.7232 | 1.7507 | 4.41x |
| filters.threshold_sauvola | 512×512 | float32 | strided | 1 | 7.0081 | 2.4273 | 2.89x |
| filters.threshold_sauvola | 512×512 | float32 | strided | 4 | 6.6583 | 1.2160 | 5.48x |
| filters.threshold_niblack | 512×512 | float32 | fortran | 1 | 10.9423 | 3.0795 | 3.55x |
| filters.threshold_niblack | 512×512 | float32 | fortran | 4 | 10.0809 | 1.1363 | 8.87x |
| filters.threshold_sauvola | 512×512 | float32 | fortran | 1 | 10.6287 | 3.0172 | 3.52x |
| filters.threshold_sauvola | 512×512 | float32 | fortran | 4 | 11.4317 | 1.1472 | 9.96x |
| filters.threshold_niblack | 1×512×512 | float32 | volume | 1 | 16.3774 | 4.5735 | 3.58x |
| filters.threshold_niblack | 1×512×512 | float32 | volume | 4 | 18.1337 | 2.4920 | 7.28x |
| filters.threshold_sauvola | 1×512×512 | float32 | volume | 1 | 14.9911 | 5.0934 | 2.94x |
| filters.threshold_sauvola | 1×512×512 | float32 | volume | 4 | 16.5214 | 2.2600 | 7.31x |
| filters.threshold_niblack | 512×512 | float32 | int64-window | 1 | 7.1444 | 2.5127 | 2.84x |
| filters.threshold_niblack | 512×512 | float32 | int64-window | 4 | 6.3582 | 0.7918 | 8.03x |
| filters.threshold_sauvola | 512×512 | float32 | int64-window | 1 | 7.9673 | 2.4362 | 3.27x |
| filters.threshold_sauvola | 512×512 | float32 | int64-window | 4 | 6.4539 | 1.3195 | 4.89x |
| filters.threshold_niblack | 512×512 | float32 | k-0.0 | 1 | 5.8445 | 2.1982 | 2.66x |
| filters.threshold_niblack | 512×512 | float32 | k-0.0 | 4 | 6.8862 | 1.4127 | 4.87x |
| filters.threshold_sauvola | 512×512 | float32 | k-0.0 | 1 | 6.7354 | 2.8510 | 2.36x |
| filters.threshold_sauvola | 512×512 | float32 | k-0.0 | 4 | 6.1383 | 1.2802 | 4.79x |
| filters.threshold_niblack | 512×512 | float32 | k--0.2 | 1 | 6.0804 | 2.1712 | 2.80x |
| filters.threshold_niblack | 512×512 | float32 | k--0.2 | 4 | 5.9313 | 0.8393 | 7.07x |
| filters.threshold_sauvola | 512×512 | float32 | k--0.2 | 1 | 6.2440 | 2.0452 | 3.05x |
| filters.threshold_sauvola | 512×512 | float32 | k--0.2 | 4 | 6.5625 | 1.2333 | 5.32x |
| filters.threshold_niblack | 512×512 | float32 | k-np.float32(0.17) | 1 | 6.6025 | 2.2307 | 2.96x |
| filters.threshold_niblack | 512×512 | float32 | k-np.float32(0.17) | 4 | 6.8289 | 0.8271 | 8.26x |
| filters.threshold_sauvola | 512×512 | float32 | k-np.float32(0.17) | 1 | 7.3985 | 2.3164 | 3.19x |
| filters.threshold_sauvola | 512×512 | float32 | k-np.float32(0.17) | 4 | 7.2188 | 1.1876 | 6.08x |
| filters.threshold_niblack | 512×512 | float32 | k-np.float64(0.17) | 1 | 7.4750 | 2.1757 | 3.44x |
| filters.threshold_niblack | 512×512 | float32 | k-np.float64(0.17) | 4 | 7.7938 | 1.2044 | 6.47x |
| filters.threshold_sauvola | 512×512 | float32 | k-np.float64(0.17) | 1 | 7.3160 | 2.3441 | 3.12x |
| filters.threshold_sauvola | 512×512 | float32 | k-np.float64(0.17) | 4 | 7.3273 | 1.1174 | 6.56x |
| filters.threshold_sauvola | 512×512 | float32 | r-0.5 | 1 | 7.5518 | 2.3294 | 3.24x |
| filters.threshold_sauvola | 512×512 | float32 | r-0.5 | 4 | 6.1604 | 1.1519 | 5.35x |
| filters.threshold_sauvola | 512×512 | float32 | r-128.0 | 1 | 7.9733 | 4.0378 | 1.97x |
| filters.threshold_sauvola | 512×512 | float32 | r-128.0 | 4 | 6.6432 | 1.0841 | 6.13x |
| filters.threshold_sauvola | 512×512 | float32 | r-np.float64(0.5) | 1 | 11.1064 | 5.0052 | 2.22x |
| filters.threshold_sauvola | 512×512 | float32 | r-np.float64(0.5) | 4 | 9.7432 | 1.2434 | 7.84x |
| filters.threshold_niblack | 512×512 | float64 | window-15 | 1 | 8.4692 | 2.3436 | 3.61x |
| filters.threshold_niblack | 512×512 | float64 | window-15 | 4 | 8.1227 | 1.1187 | 7.26x |
| filters.threshold_sauvola | 512×512 | float64 | window-15 | 1 | 10.0598 | 2.5590 | 3.93x |
| filters.threshold_sauvola | 512×512 | float64 | window-15 | 4 | 7.9843 | 1.0238 | 7.80x |
| filters.threshold_niblack | 512×512 | float64 | window-3 | 1 | 8.4087 | 2.0676 | 4.07x |
| filters.threshold_niblack | 512×512 | float64 | window-3 | 4 | 9.0712 | 0.9748 | 9.31x |
| filters.threshold_sauvola | 512×512 | float64 | window-3 | 1 | 9.0450 | 2.6543 | 3.41x |
| filters.threshold_sauvola | 512×512 | float64 | window-3 | 4 | 7.8654 | 1.2622 | 6.23x |
| filters.threshold_niblack | 512×512 | float64 | window-31 | 1 | 10.1617 | 2.9644 | 3.43x |
| filters.threshold_niblack | 512×512 | float64 | window-31 | 4 | 9.0855 | 0.9491 | 9.57x |
| filters.threshold_sauvola | 512×512 | float64 | window-31 | 1 | 9.5915 | 2.5690 | 3.73x |
| filters.threshold_sauvola | 512×512 | float64 | window-31 | 4 | 9.6063 | 1.1408 | 8.42x |
| filters.threshold_niblack | 512×512 | float64 | window-127 | 1 | 11.9676 | 3.2958 | 3.63x |
| filters.threshold_niblack | 512×512 | float64 | window-127 | 4 | 10.1074 | 1.5835 | 6.38x |
| filters.threshold_sauvola | 512×512 | float64 | window-127 | 1 | 10.9876 | 4.0474 | 2.71x |
| filters.threshold_sauvola | 512×512 | float64 | window-127 | 4 | 11.4818 | 1.3278 | 8.65x |
| filters.threshold_niblack | 512×512 | float64 | window-(3, 31) | 1 | 7.1113 | 2.0203 | 3.52x |
| filters.threshold_niblack | 512×512 | float64 | window-(3, 31) | 4 | 7.6471 | 0.7723 | 9.90x |
| filters.threshold_sauvola | 512×512 | float64 | window-(3, 31) | 1 | 8.8704 | 2.6464 | 3.35x |
| filters.threshold_sauvola | 512×512 | float64 | window-(3, 31) | 4 | 9.2741 | 1.2133 | 7.64x |
| filters.threshold_niblack | 512×512 | float64 | reversed | 1 | 9.0611 | 3.2304 | 2.80x |
| filters.threshold_niblack | 512×512 | float64 | reversed | 4 | 12.6450 | 1.6093 | 7.86x |
| filters.threshold_sauvola | 512×512 | float64 | reversed | 1 | 10.0019 | 3.7872 | 2.64x |
| filters.threshold_sauvola | 512×512 | float64 | reversed | 4 | 9.0550 | 1.9174 | 4.72x |
| filters.threshold_niblack | 512×512 | float64 | strided | 1 | 10.0283 | 3.8405 | 2.61x |
| filters.threshold_niblack | 512×512 | float64 | strided | 4 | 9.2498 | 2.7721 | 3.34x |
| filters.threshold_sauvola | 512×512 | float64 | strided | 1 | 10.3830 | 3.8593 | 2.69x |
| filters.threshold_sauvola | 512×512 | float64 | strided | 4 | 10.0907 | 2.3118 | 4.36x |
| filters.threshold_niblack | 512×512 | float64 | fortran | 1 | 13.2635 | 4.9344 | 2.69x |
| filters.threshold_niblack | 512×512 | float64 | fortran | 4 | 12.8240 | 3.1483 | 4.07x |
| filters.threshold_sauvola | 512×512 | float64 | fortran | 1 | 12.2453 | 4.6368 | 2.64x |
| filters.threshold_sauvola | 512×512 | float64 | fortran | 4 | 13.4041 | 3.5229 | 3.80x |
| filters.threshold_niblack | 1×512×512 | float64 | volume | 1 | 17.2355 | 5.5583 | 3.10x |
| filters.threshold_niblack | 1×512×512 | float64 | volume | 4 | 17.7101 | 2.1144 | 8.38x |
| filters.threshold_sauvola | 1×512×512 | float64 | volume | 1 | 16.8614 | 5.0644 | 3.33x |
| filters.threshold_sauvola | 1×512×512 | float64 | volume | 4 | 20.9292 | 2.8660 | 7.30x |
| filters.threshold_niblack | 512×512 | float64 | int64-window | 1 | 7.8133 | 3.0254 | 2.58x |
| filters.threshold_niblack | 512×512 | float64 | int64-window | 4 | 11.0041 | 1.1370 | 9.68x |
| filters.threshold_sauvola | 512×512 | float64 | int64-window | 1 | 10.5335 | 3.6934 | 2.85x |
| filters.threshold_sauvola | 512×512 | float64 | int64-window | 4 | 8.0871 | 0.7918 | 10.21x |
| filters.threshold_niblack | 512×512 | float64 | k-0.0 | 1 | 8.6903 | 3.4672 | 2.51x |
| filters.threshold_niblack | 512×512 | float64 | k-0.0 | 4 | 8.0398 | 0.9506 | 8.46x |
| filters.threshold_sauvola | 512×512 | float64 | k-0.0 | 1 | 10.3131 | 3.8115 | 2.71x |
| filters.threshold_sauvola | 512×512 | float64 | k-0.0 | 4 | 8.4393 | 1.2619 | 6.69x |
| filters.threshold_niblack | 512×512 | float64 | k--0.2 | 1 | 9.1075 | 3.6223 | 2.51x |
| filters.threshold_niblack | 512×512 | float64 | k--0.2 | 4 | 9.1280 | 1.0718 | 8.52x |
| filters.threshold_sauvola | 512×512 | float64 | k--0.2 | 1 | 11.4146 | 4.3109 | 2.65x |
| filters.threshold_sauvola | 512×512 | float64 | k--0.2 | 4 | 8.9812 | 1.0354 | 8.67x |
| filters.threshold_niblack | 512×512 | float64 | k-np.float32(0.17) | 1 | 13.8475 | 5.3075 | 2.61x |
| filters.threshold_niblack | 512×512 | float64 | k-np.float32(0.17) | 4 | 10.7462 | 1.5815 | 6.80x |
| filters.threshold_sauvola | 512×512 | float64 | k-np.float32(0.17) | 1 | 10.0938 | 3.6396 | 2.77x |
| filters.threshold_sauvola | 512×512 | float64 | k-np.float32(0.17) | 4 | 9.3308 | 1.0202 | 9.15x |
| filters.threshold_niblack | 512×512 | float64 | k-np.float64(0.17) | 1 | 8.1137 | 3.4286 | 2.37x |
| filters.threshold_niblack | 512×512 | float64 | k-np.float64(0.17) | 4 | 8.6111 | 0.7816 | 11.02x |
| filters.threshold_sauvola | 512×512 | float64 | k-np.float64(0.17) | 1 | 9.1345 | 3.1220 | 2.93x |
| filters.threshold_sauvola | 512×512 | float64 | k-np.float64(0.17) | 4 | 8.8100 | 0.7970 | 11.05x |
| filters.threshold_sauvola | 512×512 | float64 | r-0.5 | 1 | 9.3799 | 3.2355 | 2.90x |
| filters.threshold_sauvola | 512×512 | float64 | r-0.5 | 4 | 8.7864 | 1.0806 | 8.13x |
| filters.threshold_sauvola | 512×512 | float64 | r-128.0 | 1 | 10.3137 | 3.4941 | 2.95x |
| filters.threshold_sauvola | 512×512 | float64 | r-128.0 | 4 | 9.6555 | 1.0019 | 9.64x |
| filters.threshold_sauvola | 512×512 | float64 | r-np.float64(0.5) | 1 | 10.0691 | 3.6495 | 2.76x |
| filters.threshold_sauvola | 512×512 | float64 | r-np.float64(0.5) | 4 | 8.1483 | 0.9111 | 8.94x |
| filters.threshold_niblack | 512×512 | bool | additional-width | 1 | 8.8883 | 2.2616 | 3.93x |
| filters.threshold_niblack | 512×512 | bool | additional-width | 4 | 8.6371 | 0.6439 | 13.41x |
| filters.threshold_sauvola | 512×512 | bool | additional-width | 1 | 8.4821 | 2.6789 | 3.17x |
| filters.threshold_sauvola | 512×512 | bool | additional-width | 4 | 10.5084 | 1.1873 | 8.85x |
| filters.threshold_niblack | 512×512 | int8 | additional-width | 1 | 9.4808 | 1.9734 | 4.80x |
| filters.threshold_niblack | 512×512 | int8 | additional-width | 4 | 7.8525 | 0.5600 | 14.02x |
| filters.threshold_sauvola | 512×512 | int8 | additional-width | 1 | 7.9111 | 2.1065 | 3.76x |
| filters.threshold_sauvola | 512×512 | int8 | additional-width | 4 | 8.1869 | 0.8098 | 10.11x |
| filters.threshold_niblack | 512×512 | uint32 | additional-width | 1 | 7.9680 | 2.0290 | 3.93x |
| filters.threshold_niblack | 512×512 | uint32 | additional-width | 4 | 7.5754 | 0.9573 | 7.91x |
| filters.threshold_sauvola | 512×512 | uint32 | additional-width | 1 | 8.9619 | 2.1022 | 4.26x |
| filters.threshold_sauvola | 512×512 | uint32 | additional-width | 4 | 11.0970 | 1.2984 | 8.55x |
| filters.threshold_niblack | 512×512 | int32 | additional-width | 1 | 11.3365 | 4.3757 | 2.59x |
| filters.threshold_niblack | 512×512 | int32 | additional-width | 4 | 9.9321 | 1.3537 | 7.34x |
| filters.threshold_sauvola | 512×512 | int32 | additional-width | 1 | 9.3698 | 2.1698 | 4.32x |
| filters.threshold_sauvola | 512×512 | int32 | additional-width | 4 | 9.1670 | 0.7429 | 12.34x |
| filters.threshold_niblack | 512×512 | uint64 | additional-width | 1 | 7.9674 | 3.4508 | 2.31x |
| filters.threshold_niblack | 512×512 | uint64 | additional-width | 4 | 9.9799 | 1.9315 | 5.17x |
| filters.threshold_sauvola | 512×512 | uint64 | additional-width | 1 | 9.0665 | 3.9533 | 2.29x |
| filters.threshold_sauvola | 512×512 | uint64 | additional-width | 4 | 9.8328 | 1.3163 | 7.47x |
| filters.threshold_niblack | 512×512 | int64 | additional-width | 1 | 7.5911 | 2.2639 | 3.35x |
| filters.threshold_niblack | 512×512 | int64 | additional-width | 4 | 10.1526 | 0.9452 | 10.74x |
| filters.threshold_sauvola | 512×512 | int64 | additional-width | 1 | 9.6566 | 2.6157 | 3.69x |
| filters.threshold_sauvola | 512×512 | int64 | additional-width | 4 | 7.9992 | 0.9891 | 8.09x |
