# Local threshold performance

Reference: original scikit-image 0.26.0. Reference thread pools use one worker; native workers are explicit. Each case checks dtype, shape, warnings and numerical parity with fallback disabled. Timings include dispatch, layout copying, reflected padding, both integral images, local statistics and output allocation. Both scalar and NumPy int64 window metadata are covered, as are strong float64 weights and ranges. All raw samples and slower cases are retained in the JSON.

| API | Shape | Input | Workload | Workers | Original ms | C++ ms | Speedup |
|---|---|---|---|---:|---:|---:|---:|
| filters.threshold_niblack | 512×512 | uint8 | window-15 | 1 | 8.6489 | 4.1347 | 2.09x |
| filters.threshold_sauvola | 512×512 | uint8 | window-15 | 1 | 9.9385 | 3.5598 | 2.79x |
| filters.threshold_niblack | 512×512 | uint8 | window-3 | 1 | 8.5115 | 3.8115 | 2.23x |
| filters.threshold_sauvola | 512×512 | uint8 | window-3 | 1 | 8.4667 | 2.8506 | 2.97x |
| filters.threshold_niblack | 512×512 | uint8 | window-31 | 1 | 9.2132 | 4.1902 | 2.20x |
| filters.threshold_sauvola | 512×512 | uint8 | window-31 | 1 | 11.5890 | 5.3497 | 2.17x |
| filters.threshold_niblack | 512×512 | uint8 | window-127 | 1 | 14.1299 | 5.1344 | 2.75x |
| filters.threshold_sauvola | 512×512 | uint8 | window-127 | 1 | 15.6783 | 7.2983 | 2.15x |
| filters.threshold_niblack | 512×512 | uint8 | window-(3, 31) | 1 | 8.7444 | 2.2781 | 3.84x |
| filters.threshold_sauvola | 512×512 | uint8 | window-(3, 31) | 1 | 10.7728 | 3.6057 | 2.99x |
| filters.threshold_niblack | 512×512 | uint8 | reversed | 1 | 10.1028 | 2.7201 | 3.71x |
| filters.threshold_sauvola | 512×512 | uint8 | reversed | 1 | 11.4593 | 4.1421 | 2.77x |
| filters.threshold_niblack | 512×512 | uint8 | strided | 1 | 8.3905 | 2.3205 | 3.62x |
| filters.threshold_sauvola | 512×512 | uint8 | strided | 1 | 12.9937 | 3.8466 | 3.38x |
| filters.threshold_niblack | 512×512 | uint8 | fortran | 1 | 16.6394 | 3.6574 | 4.55x |
| filters.threshold_sauvola | 512×512 | uint8 | fortran | 1 | 21.1221 | 3.8582 | 5.47x |
| filters.threshold_niblack | 1×512×512 | uint8 | volume | 1 | 26.2680 | 7.3081 | 3.59x |
| filters.threshold_sauvola | 1×512×512 | uint8 | volume | 1 | 26.7741 | 7.3638 | 3.64x |
| filters.threshold_niblack | 512×512 | uint16 | window-15 | 1 | 13.9635 | 3.3535 | 4.16x |
| filters.threshold_sauvola | 512×512 | uint16 | window-15 | 1 | 12.9223 | 3.0605 | 4.22x |
| filters.threshold_niblack | 512×512 | uint16 | window-3 | 1 | 10.1214 | 2.6926 | 3.76x |
| filters.threshold_sauvola | 512×512 | uint16 | window-3 | 1 | 16.6858 | 5.0757 | 3.29x |
| filters.threshold_niblack | 512×512 | uint16 | window-31 | 1 | 13.1351 | 3.6072 | 3.64x |
| filters.threshold_sauvola | 512×512 | uint16 | window-31 | 1 | 18.1723 | 5.2759 | 3.44x |
| filters.threshold_niblack | 512×512 | uint16 | window-127 | 1 | 20.0127 | 6.5075 | 3.08x |
| filters.threshold_sauvola | 512×512 | uint16 | window-127 | 1 | 19.6817 | 7.0968 | 2.77x |
| filters.threshold_niblack | 512×512 | uint16 | window-(3, 31) | 1 | 11.6890 | 2.9652 | 3.94x |
| filters.threshold_sauvola | 512×512 | uint16 | window-(3, 31) | 1 | 17.6516 | 3.5331 | 5.00x |
| filters.threshold_niblack | 512×512 | uint16 | reversed | 1 | 14.6629 | 4.9076 | 2.99x |
| filters.threshold_sauvola | 512×512 | uint16 | reversed | 1 | 13.1992 | 3.8938 | 3.39x |
| filters.threshold_niblack | 512×512 | uint16 | strided | 1 | 15.0919 | 5.3668 | 2.81x |
| filters.threshold_sauvola | 512×512 | uint16 | strided | 1 | 12.0499 | 3.3169 | 3.63x |
| filters.threshold_niblack | 512×512 | uint16 | fortran | 1 | 18.0874 | 4.4160 | 4.10x |
| filters.threshold_sauvola | 512×512 | uint16 | fortran | 1 | 25.3756 | 5.0835 | 4.99x |
| filters.threshold_niblack | 1×512×512 | uint16 | volume | 1 | 26.1020 | 8.1122 | 3.22x |
| filters.threshold_sauvola | 1×512×512 | uint16 | volume | 1 | 27.2073 | 8.5119 | 3.20x |
| filters.threshold_niblack | 512×512 | int16 | window-15 | 1 | 11.3835 | 4.3989 | 2.59x |
| filters.threshold_sauvola | 512×512 | int16 | window-15 | 1 | 13.3549 | 3.7850 | 3.53x |
| filters.threshold_niblack | 512×512 | int16 | window-3 | 1 | 13.1125 | 3.5119 | 3.73x |
| filters.threshold_sauvola | 512×512 | int16 | window-3 | 1 | 10.5725 | 3.1094 | 3.40x |
| filters.threshold_niblack | 512×512 | int16 | window-31 | 1 | 13.6973 | 4.9658 | 2.76x |
| filters.threshold_sauvola | 512×512 | int16 | window-31 | 1 | 12.8886 | 5.0499 | 2.55x |
| filters.threshold_niblack | 512×512 | int16 | window-127 | 1 | 14.5870 | 6.0500 | 2.41x |
| filters.threshold_sauvola | 512×512 | int16 | window-127 | 1 | 17.1875 | 5.9885 | 2.87x |
| filters.threshold_niblack | 512×512 | int16 | window-(3, 31) | 1 | 18.0466 | 4.8418 | 3.73x |
| filters.threshold_sauvola | 512×512 | int16 | window-(3, 31) | 1 | 14.5284 | 4.7575 | 3.05x |
| filters.threshold_niblack | 512×512 | int16 | reversed | 1 | 12.4127 | 4.6794 | 2.65x |
| filters.threshold_sauvola | 512×512 | int16 | reversed | 1 | 15.2190 | 3.9445 | 3.86x |
| filters.threshold_niblack | 512×512 | int16 | strided | 1 | 12.3123 | 3.1607 | 3.90x |
| filters.threshold_sauvola | 512×512 | int16 | strided | 1 | 14.1751 | 4.2859 | 3.31x |
| filters.threshold_niblack | 512×512 | int16 | fortran | 1 | 12.6659 | 3.7098 | 3.41x |
| filters.threshold_sauvola | 512×512 | int16 | fortran | 1 | 25.7196 | 4.7623 | 5.40x |
| filters.threshold_niblack | 1×512×512 | int16 | volume | 1 | 21.9693 | 6.0370 | 3.64x |
| filters.threshold_sauvola | 1×512×512 | int16 | volume | 1 | 23.2405 | 6.9512 | 3.34x |
| filters.threshold_niblack | 512×512 | float32 | window-15 | 1 | 7.2426 | 3.3309 | 2.17x |
| filters.threshold_sauvola | 512×512 | float32 | window-15 | 1 | 8.0835 | 2.8599 | 2.83x |
| filters.threshold_niblack | 512×512 | float32 | window-3 | 1 | 7.4520 | 3.2150 | 2.32x |
| filters.threshold_sauvola | 512×512 | float32 | window-3 | 1 | 7.2405 | 2.5852 | 2.80x |
| filters.threshold_niblack | 512×512 | float32 | window-31 | 1 | 8.6805 | 2.1817 | 3.98x |
| filters.threshold_sauvola | 512×512 | float32 | window-31 | 1 | 11.1362 | 3.2005 | 3.48x |
| filters.threshold_niblack | 512×512 | float32 | window-127 | 1 | 14.2766 | 4.5997 | 3.10x |
| filters.threshold_sauvola | 512×512 | float32 | window-127 | 1 | 14.7347 | 7.2044 | 2.05x |
| filters.threshold_niblack | 512×512 | float32 | window-(3, 31) | 1 | 9.1214 | 2.2584 | 4.04x |
| filters.threshold_sauvola | 512×512 | float32 | window-(3, 31) | 1 | 8.7868 | 2.3138 | 3.80x |
| filters.threshold_niblack | 512×512 | float32 | reversed | 1 | 6.9168 | 2.3657 | 2.92x |
| filters.threshold_sauvola | 512×512 | float32 | reversed | 1 | 7.8511 | 2.4412 | 3.22x |
| filters.threshold_niblack | 512×512 | float32 | strided | 1 | 7.0086 | 3.2585 | 2.15x |
| filters.threshold_sauvola | 512×512 | float32 | strided | 1 | 6.6269 | 2.3021 | 2.88x |
| filters.threshold_niblack | 512×512 | float32 | fortran | 1 | 12.8916 | 3.6723 | 3.51x |
| filters.threshold_sauvola | 512×512 | float32 | fortran | 1 | 12.2811 | 2.9821 | 4.12x |
| filters.threshold_niblack | 1×512×512 | float32 | volume | 1 | 18.4052 | 5.7954 | 3.18x |
| filters.threshold_sauvola | 1×512×512 | float32 | volume | 1 | 18.5524 | 6.5271 | 2.84x |
| filters.threshold_niblack | 512×512 | float32 | int64-window | 1 | 8.2206 | 2.5956 | 3.17x |
| filters.threshold_sauvola | 512×512 | float32 | int64-window | 1 | 7.6162 | 2.8822 | 2.64x |
| filters.threshold_niblack | 512×512 | float32 | k-0.0 | 1 | 6.3359 | 2.5936 | 2.44x |
| filters.threshold_sauvola | 512×512 | float32 | k-0.0 | 1 | 6.8931 | 1.9629 | 3.51x |
| filters.threshold_niblack | 512×512 | float32 | k--0.2 | 1 | 7.0072 | 2.4874 | 2.82x |
| filters.threshold_sauvola | 512×512 | float32 | k--0.2 | 1 | 6.9828 | 2.3058 | 3.03x |
| filters.threshold_niblack | 512×512 | float32 | k-np.float32(0.17) | 1 | 7.0591 | 2.6003 | 2.71x |
| filters.threshold_sauvola | 512×512 | float32 | k-np.float32(0.17) | 1 | 9.6466 | 2.3966 | 4.03x |
| filters.threshold_niblack | 512×512 | float32 | k-np.float64(0.17) | 1 | 7.5339 | 2.1395 | 3.52x |
| filters.threshold_sauvola | 512×512 | float32 | k-np.float64(0.17) | 1 | 8.7108 | 2.8872 | 3.02x |
| filters.threshold_sauvola | 512×512 | float32 | r-0.5 | 1 | 9.7115 | 2.9020 | 3.35x |
| filters.threshold_sauvola | 512×512 | float32 | r-128.0 | 1 | 11.2915 | 4.6756 | 2.42x |
| filters.threshold_sauvola | 512×512 | float32 | r-np.float64(0.5) | 1 | 10.2270 | 3.0576 | 3.34x |
| filters.threshold_niblack | 512×512 | float64 | window-15 | 1 | 9.5351 | 2.3396 | 4.08x |
| filters.threshold_sauvola | 512×512 | float64 | window-15 | 1 | 9.1820 | 2.2458 | 4.09x |
| filters.threshold_niblack | 512×512 | float64 | window-3 | 1 | 8.5553 | 1.9589 | 4.37x |
| filters.threshold_sauvola | 512×512 | float64 | window-3 | 1 | 8.3548 | 2.2897 | 3.65x |
| filters.threshold_niblack | 512×512 | float64 | window-31 | 1 | 8.2253 | 3.3005 | 2.49x |
| filters.threshold_sauvola | 512×512 | float64 | window-31 | 1 | 9.8116 | 2.3447 | 4.18x |
| filters.threshold_niblack | 512×512 | float64 | window-127 | 1 | 12.8759 | 5.9957 | 2.15x |
| filters.threshold_sauvola | 512×512 | float64 | window-127 | 1 | 11.0890 | 4.3092 | 2.57x |
| filters.threshold_niblack | 512×512 | float64 | window-(3, 31) | 1 | 7.7767 | 2.0100 | 3.87x |
| filters.threshold_sauvola | 512×512 | float64 | window-(3, 31) | 1 | 9.0342 | 2.5816 | 3.50x |
| filters.threshold_niblack | 512×512 | float64 | reversed | 1 | 9.5218 | 3.4290 | 2.78x |
| filters.threshold_sauvola | 512×512 | float64 | reversed | 1 | 9.6218 | 3.9889 | 2.41x |
| filters.threshold_niblack | 512×512 | float64 | strided | 1 | 9.6242 | 3.5770 | 2.69x |
| filters.threshold_sauvola | 512×512 | float64 | strided | 1 | 10.7642 | 4.2227 | 2.55x |
| filters.threshold_niblack | 512×512 | float64 | fortran | 1 | 14.0986 | 4.8281 | 2.92x |
| filters.threshold_sauvola | 512×512 | float64 | fortran | 1 | 14.9020 | 5.0026 | 2.98x |
| filters.threshold_niblack | 1×512×512 | float64 | volume | 1 | 18.9592 | 4.8598 | 3.90x |
| filters.threshold_sauvola | 1×512×512 | float64 | volume | 1 | 18.1866 | 5.4534 | 3.33x |
| filters.threshold_niblack | 512×512 | float64 | int64-window | 1 | 9.2164 | 3.4823 | 2.65x |
| filters.threshold_sauvola | 512×512 | float64 | int64-window | 1 | 10.4762 | 3.4553 | 3.03x |
| filters.threshold_niblack | 512×512 | float64 | k-0.0 | 1 | 8.2153 | 3.2008 | 2.57x |
| filters.threshold_sauvola | 512×512 | float64 | k-0.0 | 1 | 10.1641 | 3.4839 | 2.92x |
| filters.threshold_niblack | 512×512 | float64 | k--0.2 | 1 | 9.4727 | 3.5227 | 2.69x |
| filters.threshold_sauvola | 512×512 | float64 | k--0.2 | 1 | 9.1086 | 3.5343 | 2.58x |
| filters.threshold_niblack | 512×512 | float64 | k-np.float32(0.17) | 1 | 8.9053 | 3.6447 | 2.44x |
| filters.threshold_sauvola | 512×512 | float64 | k-np.float32(0.17) | 1 | 10.1304 | 3.7980 | 2.67x |
| filters.threshold_niblack | 512×512 | float64 | k-np.float64(0.17) | 1 | 10.2152 | 3.7691 | 2.71x |
| filters.threshold_sauvola | 512×512 | float64 | k-np.float64(0.17) | 1 | 8.6082 | 2.1948 | 3.92x |
| filters.threshold_sauvola | 512×512 | float64 | r-0.5 | 1 | 9.7793 | 2.6174 | 3.74x |
| filters.threshold_sauvola | 512×512 | float64 | r-128.0 | 1 | 8.6789 | 2.2389 | 3.88x |
| filters.threshold_sauvola | 512×512 | float64 | r-np.float64(0.5) | 1 | 9.3245 | 3.0417 | 3.07x |
| filters.threshold_niblack | 512×512 | bool | additional-width | 1 | 9.0370 | 1.8609 | 4.86x |
| filters.threshold_sauvola | 512×512 | bool | additional-width | 1 | 9.8731 | 3.5756 | 2.76x |
| filters.threshold_niblack | 512×512 | int8 | additional-width | 1 | 11.4089 | 2.5823 | 4.42x |
| filters.threshold_sauvola | 512×512 | int8 | additional-width | 1 | 12.3142 | 3.6580 | 3.37x |
| filters.threshold_niblack | 512×512 | uint32 | additional-width | 1 | 20.8359 | 5.1875 | 4.02x |
| filters.threshold_sauvola | 512×512 | uint32 | additional-width | 1 | 12.9752 | 3.1674 | 4.10x |
| filters.threshold_niblack | 512×512 | int32 | additional-width | 1 | 16.5922 | 4.8407 | 3.43x |
| filters.threshold_sauvola | 512×512 | int32 | additional-width | 1 | 12.5388 | 2.8275 | 4.43x |
| filters.threshold_niblack | 512×512 | uint64 | additional-width | 1 | 12.9163 | 3.9642 | 3.26x |
| filters.threshold_sauvola | 512×512 | uint64 | additional-width | 1 | 12.4322 | 4.8129 | 2.58x |
| filters.threshold_niblack | 512×512 | int64 | additional-width | 1 | 11.9566 | 2.9234 | 4.09x |
| filters.threshold_sauvola | 512×512 | int64 | additional-width | 1 | 12.5625 | 3.3578 | 3.74x |
