# Local threshold performance

Reference: original scikit-image 0.26.0. Reference thread pools use one worker; native workers are explicit. Each case checks dtype, shape, warnings and numerical parity with fallback disabled. Timings include dispatch, layout copying, reflected padding, both integral images, local statistics and output allocation. Both scalar and NumPy int64 window metadata are covered, as are strong float64 weights and ranges. All raw samples and slower cases are retained in the JSON.

| API | Shape | Input | Workload | Workers | Original ms | C++ ms | Speedup |
|---|---|---|---|---:|---:|---:|---:|
| filters.threshold_niblack | 512×512 | uint8 | window-15 | 1 | 8.8125 | 4.0314 | 2.19x |
| filters.threshold_sauvola | 512×512 | uint8 | window-15 | 1 | 8.7522 | 2.3129 | 3.78x |
| filters.threshold_niblack | 512×512 | uint8 | window-3 | 1 | 8.5749 | 3.6938 | 2.32x |
| filters.threshold_sauvola | 512×512 | uint8 | window-3 | 1 | 9.4221 | 3.0361 | 3.10x |
| filters.threshold_niblack | 512×512 | uint8 | window-31 | 1 | 9.3856 | 4.4674 | 2.10x |
| filters.threshold_sauvola | 512×512 | uint8 | window-31 | 1 | 9.7300 | 2.4025 | 4.05x |
| filters.threshold_niblack | 512×512 | uint8 | window-127 | 1 | 11.5059 | 3.0870 | 3.73x |
| filters.threshold_sauvola | 512×512 | uint8 | window-127 | 1 | 13.5795 | 7.7091 | 1.76x |
| filters.threshold_niblack | 512×512 | uint8 | window-(3, 31) | 1 | 11.1941 | 4.5983 | 2.43x |
| filters.threshold_sauvola | 512×512 | uint8 | window-(3, 31) | 1 | 9.7606 | 4.9843 | 1.96x |
| filters.threshold_niblack | 512×512 | uint8 | reversed | 1 | 8.9531 | 3.8697 | 2.31x |
| filters.threshold_sauvola | 512×512 | uint8 | reversed | 1 | 11.4091 | 4.9233 | 2.32x |
| filters.threshold_niblack | 512×512 | uint8 | strided | 1 | 8.7422 | 3.4677 | 2.52x |
| filters.threshold_sauvola | 512×512 | uint8 | strided | 1 | 11.8903 | 5.0430 | 2.36x |
| filters.threshold_niblack | 512×512 | uint8 | fortran | 1 | 12.9434 | 3.9339 | 3.29x |
| filters.threshold_sauvola | 512×512 | uint8 | fortran | 1 | 16.3937 | 5.2871 | 3.10x |
| filters.threshold_niblack | 1×512×512 | uint8 | volume | 1 | 22.2187 | 9.0080 | 2.47x |
| filters.threshold_sauvola | 1×512×512 | uint8 | volume | 1 | 22.3744 | 7.9455 | 2.82x |
| filters.threshold_niblack | 512×512 | uint16 | window-15 | 1 | 9.1818 | 3.9676 | 2.31x |
| filters.threshold_sauvola | 512×512 | uint16 | window-15 | 1 | 9.1544 | 3.1206 | 2.93x |
| filters.threshold_niblack | 512×512 | uint16 | window-3 | 1 | 9.0456 | 3.4869 | 2.59x |
| filters.threshold_sauvola | 512×512 | uint16 | window-3 | 1 | 9.6397 | 3.9379 | 2.45x |
| filters.threshold_niblack | 512×512 | uint16 | window-31 | 1 | 9.4117 | 4.4508 | 2.11x |
| filters.threshold_sauvola | 512×512 | uint16 | window-31 | 1 | 10.1245 | 4.8827 | 2.07x |
| filters.threshold_niblack | 512×512 | uint16 | window-127 | 1 | 12.2995 | 6.9208 | 1.78x |
| filters.threshold_sauvola | 512×512 | uint16 | window-127 | 1 | 13.5025 | 7.1395 | 1.89x |
| filters.threshold_niblack | 512×512 | uint16 | window-(3, 31) | 1 | 12.0793 | 3.1413 | 3.85x |
| filters.threshold_sauvola | 512×512 | uint16 | window-(3, 31) | 1 | 13.4898 | 4.9596 | 2.72x |
| filters.threshold_niblack | 512×512 | uint16 | reversed | 1 | 9.6737 | 3.7931 | 2.55x |
| filters.threshold_sauvola | 512×512 | uint16 | reversed | 1 | 12.5643 | 5.1165 | 2.46x |
| filters.threshold_niblack | 512×512 | uint16 | strided | 1 | 13.0307 | 5.0198 | 2.60x |
| filters.threshold_sauvola | 512×512 | uint16 | strided | 1 | 16.4410 | 6.0496 | 2.72x |
| filters.threshold_niblack | 512×512 | uint16 | fortran | 1 | 17.6736 | 3.5934 | 4.92x |
| filters.threshold_sauvola | 512×512 | uint16 | fortran | 1 | 18.1472 | 3.9529 | 4.59x |
| filters.threshold_niblack | 1×512×512 | uint16 | volume | 1 | 24.0952 | 6.5284 | 3.69x |
| filters.threshold_sauvola | 1×512×512 | uint16 | volume | 1 | 22.6595 | 6.2121 | 3.65x |
| filters.threshold_niblack | 512×512 | int16 | window-15 | 1 | 9.5544 | 3.9989 | 2.39x |
| filters.threshold_sauvola | 512×512 | int16 | window-15 | 1 | 12.4833 | 3.5570 | 3.51x |
| filters.threshold_niblack | 512×512 | int16 | window-3 | 1 | 9.1184 | 2.8385 | 3.21x |
| filters.threshold_sauvola | 512×512 | int16 | window-3 | 1 | 11.2692 | 2.7567 | 4.09x |
| filters.threshold_niblack | 512×512 | int16 | window-31 | 1 | 14.1095 | 3.1434 | 4.49x |
| filters.threshold_sauvola | 512×512 | int16 | window-31 | 1 | 11.7110 | 2.4940 | 4.70x |
| filters.threshold_niblack | 512×512 | int16 | window-127 | 1 | 13.9188 | 5.2618 | 2.65x |
| filters.threshold_sauvola | 512×512 | int16 | window-127 | 1 | 15.0654 | 5.3495 | 2.82x |
| filters.threshold_niblack | 512×512 | int16 | window-(3, 31) | 1 | 10.9465 | 3.1803 | 3.44x |
| filters.threshold_sauvola | 512×512 | int16 | window-(3, 31) | 1 | 10.3258 | 2.3969 | 4.31x |
| filters.threshold_niblack | 512×512 | int16 | reversed | 1 | 12.1986 | 4.2387 | 2.88x |
| filters.threshold_sauvola | 512×512 | int16 | reversed | 1 | 8.3112 | 2.2793 | 3.65x |
| filters.threshold_niblack | 512×512 | int16 | strided | 1 | 8.6367 | 3.5602 | 2.43x |
| filters.threshold_sauvola | 512×512 | int16 | strided | 1 | 8.2730 | 2.2900 | 3.61x |
| filters.threshold_niblack | 512×512 | int16 | fortran | 1 | 13.8238 | 3.4294 | 4.03x |
| filters.threshold_sauvola | 512×512 | int16 | fortran | 1 | 16.4930 | 3.8272 | 4.31x |
| filters.threshold_niblack | 1×512×512 | int16 | volume | 1 | 19.6618 | 5.9850 | 3.29x |
| filters.threshold_sauvola | 1×512×512 | int16 | volume | 1 | 20.2263 | 5.2080 | 3.88x |
| filters.threshold_niblack | 512×512 | float32 | window-15 | 1 | 7.5663 | 2.4626 | 3.07x |
| filters.threshold_sauvola | 512×512 | float32 | window-15 | 1 | 7.5584 | 1.9084 | 3.96x |
| filters.threshold_niblack | 512×512 | float32 | window-3 | 1 | 6.4466 | 2.2170 | 2.91x |
| filters.threshold_sauvola | 512×512 | float32 | window-3 | 1 | 7.5759 | 1.8472 | 4.10x |
| filters.threshold_niblack | 512×512 | float32 | window-31 | 1 | 7.1322 | 1.9897 | 3.58x |
| filters.threshold_sauvola | 512×512 | float32 | window-31 | 1 | 7.0720 | 2.0811 | 3.40x |
| filters.threshold_niblack | 512×512 | float32 | window-127 | 1 | 12.2496 | 4.6894 | 2.61x |
| filters.threshold_sauvola | 512×512 | float32 | window-127 | 1 | 11.3923 | 4.8338 | 2.36x |
| filters.threshold_niblack | 512×512 | float32 | window-(3, 31) | 1 | 6.2994 | 2.0465 | 3.08x |
| filters.threshold_sauvola | 512×512 | float32 | window-(3, 31) | 1 | 7.6196 | 1.7785 | 4.28x |
| filters.threshold_niblack | 512×512 | float32 | reversed | 1 | 6.8922 | 2.5693 | 2.68x |
| filters.threshold_sauvola | 512×512 | float32 | reversed | 1 | 9.3590 | 2.8866 | 3.24x |
| filters.threshold_niblack | 512×512 | float32 | strided | 1 | 8.0851 | 2.7866 | 2.90x |
| filters.threshold_sauvola | 512×512 | float32 | strided | 1 | 7.6567 | 2.6565 | 2.88x |
| filters.threshold_niblack | 512×512 | float32 | fortran | 1 | 10.3841 | 3.0019 | 3.46x |
| filters.threshold_sauvola | 512×512 | float32 | fortran | 1 | 12.1174 | 3.0752 | 3.94x |
| filters.threshold_niblack | 1×512×512 | float32 | volume | 1 | 18.4234 | 5.0891 | 3.62x |
| filters.threshold_sauvola | 1×512×512 | float32 | volume | 1 | 22.2519 | 5.0654 | 4.39x |
| filters.threshold_niblack | 512×512 | float32 | int64-window | 1 | 9.4556 | 3.3499 | 2.82x |
| filters.threshold_sauvola | 512×512 | float32 | int64-window | 1 | 7.3961 | 2.9039 | 2.55x |
| filters.threshold_niblack | 512×512 | float32 | k-0.0 | 1 | 7.8656 | 2.8083 | 2.80x |
| filters.threshold_sauvola | 512×512 | float32 | k-0.0 | 1 | 9.4965 | 3.2731 | 2.90x |
| filters.threshold_niblack | 512×512 | float32 | k--0.2 | 1 | 7.6733 | 2.8011 | 2.74x |
| filters.threshold_sauvola | 512×512 | float32 | k--0.2 | 1 | 7.5932 | 2.5263 | 3.01x |
| filters.threshold_niblack | 512×512 | float32 | k-np.float32(0.17) | 1 | 6.1545 | 2.5527 | 2.41x |
| filters.threshold_sauvola | 512×512 | float32 | k-np.float32(0.17) | 1 | 7.4092 | 2.7337 | 2.71x |
| filters.threshold_niblack | 512×512 | float32 | k-np.float64(0.17) | 1 | 10.4464 | 3.0058 | 3.48x |
| filters.threshold_sauvola | 512×512 | float32 | k-np.float64(0.17) | 1 | 8.8241 | 2.5422 | 3.47x |
| filters.threshold_sauvola | 512×512 | float32 | r-0.5 | 1 | 7.7446 | 2.0711 | 3.74x |
| filters.threshold_sauvola | 512×512 | float32 | r-128.0 | 1 | 7.3714 | 2.5475 | 2.89x |
| filters.threshold_sauvola | 512×512 | float32 | r-np.float64(0.5) | 1 | 8.9464 | 2.6218 | 3.41x |
| filters.threshold_niblack | 512×512 | float64 | window-15 | 1 | 12.7961 | 2.7449 | 4.66x |
| filters.threshold_sauvola | 512×512 | float64 | window-15 | 1 | 11.3964 | 4.3104 | 2.64x |
| filters.threshold_niblack | 512×512 | float64 | window-3 | 1 | 10.7815 | 2.5242 | 4.27x |
| filters.threshold_sauvola | 512×512 | float64 | window-3 | 1 | 9.9974 | 2.3319 | 4.29x |
| filters.threshold_niblack | 512×512 | float64 | window-31 | 1 | 9.3713 | 2.1969 | 4.27x |
| filters.threshold_sauvola | 512×512 | float64 | window-31 | 1 | 9.2520 | 2.4563 | 3.77x |
| filters.threshold_niblack | 512×512 | float64 | window-127 | 1 | 10.6152 | 3.3908 | 3.13x |
| filters.threshold_sauvola | 512×512 | float64 | window-127 | 1 | 16.3352 | 7.1717 | 2.28x |
| filters.threshold_niblack | 512×512 | float64 | window-(3, 31) | 1 | 9.1739 | 2.1712 | 4.23x |
| filters.threshold_sauvola | 512×512 | float64 | window-(3, 31) | 1 | 9.1921 | 2.2968 | 4.00x |
| filters.threshold_niblack | 512×512 | float64 | reversed | 1 | 8.7626 | 4.0092 | 2.19x |
| filters.threshold_sauvola | 512×512 | float64 | reversed | 1 | 13.5093 | 5.5784 | 2.42x |
| filters.threshold_niblack | 512×512 | float64 | strided | 1 | 10.0333 | 4.6971 | 2.14x |
| filters.threshold_sauvola | 512×512 | float64 | strided | 1 | 9.8590 | 4.5346 | 2.17x |
| filters.threshold_niblack | 512×512 | float64 | fortran | 1 | 13.1975 | 5.3857 | 2.45x |
| filters.threshold_sauvola | 512×512 | float64 | fortran | 1 | 16.4800 | 6.8693 | 2.40x |
| filters.threshold_niblack | 1×512×512 | float64 | volume | 1 | 20.7797 | 5.5198 | 3.76x |
| filters.threshold_sauvola | 1×512×512 | float64 | volume | 1 | 19.0804 | 5.4244 | 3.52x |
| filters.threshold_niblack | 512×512 | float64 | int64-window | 1 | 9.2711 | 4.2134 | 2.20x |
| filters.threshold_sauvola | 512×512 | float64 | int64-window | 1 | 10.6967 | 2.4904 | 4.30x |
| filters.threshold_niblack | 512×512 | float64 | k-0.0 | 1 | 10.2694 | 3.7662 | 2.73x |
| filters.threshold_sauvola | 512×512 | float64 | k-0.0 | 1 | 8.3774 | 2.3412 | 3.58x |
| filters.threshold_niblack | 512×512 | float64 | k--0.2 | 1 | 8.8455 | 3.1928 | 2.77x |
| filters.threshold_sauvola | 512×512 | float64 | k--0.2 | 1 | 9.6042 | 2.8166 | 3.41x |
| filters.threshold_niblack | 512×512 | float64 | k-np.float32(0.17) | 1 | 12.3838 | 4.0945 | 3.02x |
| filters.threshold_sauvola | 512×512 | float64 | k-np.float32(0.17) | 1 | 9.9715 | 2.8323 | 3.52x |
| filters.threshold_niblack | 512×512 | float64 | k-np.float64(0.17) | 1 | 9.5213 | 3.9150 | 2.43x |
| filters.threshold_sauvola | 512×512 | float64 | k-np.float64(0.17) | 1 | 9.5937 | 2.3095 | 4.15x |
| filters.threshold_sauvola | 512×512 | float64 | r-0.5 | 1 | 8.5183 | 2.2331 | 3.81x |
| filters.threshold_sauvola | 512×512 | float64 | r-128.0 | 1 | 8.6599 | 2.7822 | 3.11x |
| filters.threshold_sauvola | 512×512 | float64 | r-np.float64(0.5) | 1 | 8.1820 | 2.1825 | 3.75x |
| filters.threshold_niblack | 512×512 | bool | additional-width | 1 | 10.2058 | 3.3375 | 3.06x |
| filters.threshold_sauvola | 512×512 | bool | additional-width | 1 | 10.3988 | 3.2824 | 3.17x |
| filters.threshold_niblack | 512×512 | int8 | additional-width | 1 | 9.1403 | 3.4879 | 2.62x |
| filters.threshold_sauvola | 512×512 | int8 | additional-width | 1 | 10.5378 | 3.2811 | 3.21x |
| filters.threshold_niblack | 512×512 | uint32 | additional-width | 1 | 8.0370 | 3.0397 | 2.64x |
| filters.threshold_sauvola | 512×512 | uint32 | additional-width | 1 | 9.1916 | 2.0321 | 4.52x |
| filters.threshold_niblack | 512×512 | int32 | additional-width | 1 | 11.1767 | 4.0679 | 2.75x |
| filters.threshold_sauvola | 512×512 | int32 | additional-width | 1 | 9.7087 | 1.9847 | 4.89x |
| filters.threshold_niblack | 512×512 | uint64 | additional-width | 1 | 12.4511 | 5.1138 | 2.43x |
| filters.threshold_sauvola | 512×512 | uint64 | additional-width | 1 | 9.7913 | 3.9674 | 2.47x |
| filters.threshold_niblack | 512×512 | int64 | additional-width | 1 | 8.0900 | 2.0536 | 3.94x |
| filters.threshold_sauvola | 512×512 | int64 | additional-width | 1 | 9.4975 | 2.4137 | 3.93x |
