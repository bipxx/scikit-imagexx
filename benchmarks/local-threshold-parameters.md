# Local threshold performance

Reference: original scikit-image 0.26.0. Reference thread pools use one worker; native workers are explicit. Each case checks dtype, shape, warnings and numerical parity with fallback disabled. Timings include dispatch, layout copying, reflected padding, both integral images, local statistics and output allocation. Both scalar and NumPy int64 window metadata are covered, as are strong float64 weights and ranges. All raw samples and slower cases are retained in the JSON.

| API | Shape | Input | Workload | Workers | Original ms | C++ ms | Speedup |
|---|---|---|---|---:|---:|---:|---:|
| filters.threshold_niblack | 512×512 | uint8 | window-15 | 1 | 9.2211 | 2.3897 | 3.86x |
| filters.threshold_sauvola | 512×512 | uint8 | window-15 | 1 | 13.4391 | 2.4172 | 5.56x |
| filters.threshold_niblack | 512×512 | uint8 | window-3 | 1 | 8.6124 | 2.4272 | 3.55x |
| filters.threshold_sauvola | 512×512 | uint8 | window-3 | 1 | 8.8119 | 1.8709 | 4.71x |
| filters.threshold_niblack | 512×512 | uint8 | window-31 | 1 | 10.0541 | 2.5805 | 3.90x |
| filters.threshold_sauvola | 512×512 | uint8 | window-31 | 1 | 10.9415 | 2.8834 | 3.79x |
| filters.threshold_niblack | 512×512 | uint8 | window-127 | 1 | 14.3696 | 3.7875 | 3.79x |
| filters.threshold_sauvola | 512×512 | uint8 | window-127 | 1 | 14.8347 | 5.2950 | 2.80x |
| filters.threshold_niblack | 512×512 | uint8 | window-(3, 31) | 1 | 10.2528 | 2.8841 | 3.55x |
| filters.threshold_sauvola | 512×512 | uint8 | window-(3, 31) | 1 | 15.3228 | 5.9855 | 2.56x |
| filters.threshold_niblack | 512×512 | uint8 | reversed | 1 | 15.1636 | 4.6059 | 3.29x |
| filters.threshold_sauvola | 512×512 | uint8 | reversed | 1 | 16.8047 | 5.7252 | 2.94x |
| filters.threshold_niblack | 512×512 | uint8 | strided | 1 | 19.1435 | 5.3861 | 3.55x |
| filters.threshold_sauvola | 512×512 | uint8 | strided | 1 | 22.5694 | 5.9505 | 3.79x |
| filters.threshold_niblack | 512×512 | uint8 | fortran | 1 | 19.1890 | 4.9903 | 3.85x |
| filters.threshold_sauvola | 512×512 | uint8 | fortran | 1 | 20.0516 | 6.3828 | 3.14x |
| filters.threshold_niblack | 1×512×512 | uint8 | volume | 1 | 27.7423 | 11.0748 | 2.50x |
| filters.threshold_sauvola | 1×512×512 | uint8 | volume | 1 | 29.9026 | 10.2634 | 2.91x |
| filters.threshold_niblack | 512×512 | uint16 | window-15 | 1 | 10.2676 | 2.9602 | 3.47x |
| filters.threshold_sauvola | 512×512 | uint16 | window-15 | 1 | 16.9691 | 4.8346 | 3.51x |
| filters.threshold_niblack | 512×512 | uint16 | window-3 | 1 | 8.6566 | 2.4687 | 3.51x |
| filters.threshold_sauvola | 512×512 | uint16 | window-3 | 1 | 11.2766 | 3.4091 | 3.31x |
| filters.threshold_niblack | 512×512 | uint16 | window-31 | 1 | 10.4997 | 2.7946 | 3.76x |
| filters.threshold_sauvola | 512×512 | uint16 | window-31 | 1 | 12.6536 | 3.0292 | 4.18x |
| filters.threshold_niblack | 512×512 | uint16 | window-127 | 1 | 10.4667 | 3.6514 | 2.87x |
| filters.threshold_sauvola | 512×512 | uint16 | window-127 | 1 | 15.1810 | 5.0181 | 3.03x |
| filters.threshold_niblack | 512×512 | uint16 | window-(3, 31) | 1 | 9.1208 | 2.5061 | 3.64x |
| filters.threshold_sauvola | 512×512 | uint16 | window-(3, 31) | 1 | 9.5448 | 2.2283 | 4.28x |
| filters.threshold_niblack | 512×512 | uint16 | reversed | 1 | 11.0794 | 4.3694 | 2.54x |
| filters.threshold_sauvola | 512×512 | uint16 | reversed | 1 | 9.2890 | 2.7639 | 3.36x |
| filters.threshold_niblack | 512×512 | uint16 | strided | 1 | 9.7957 | 3.5970 | 2.72x |
| filters.threshold_sauvola | 512×512 | uint16 | strided | 1 | 9.0593 | 2.3174 | 3.91x |
| filters.threshold_niblack | 512×512 | uint16 | fortran | 1 | 13.4499 | 3.1834 | 4.23x |
| filters.threshold_sauvola | 512×512 | uint16 | fortran | 1 | 12.2323 | 2.2078 | 5.54x |
| filters.threshold_niblack | 1×512×512 | uint16 | volume | 1 | 19.1623 | 4.8937 | 3.92x |
| filters.threshold_sauvola | 1×512×512 | uint16 | volume | 1 | 19.3437 | 5.2496 | 3.68x |
| filters.threshold_niblack | 512×512 | int16 | window-15 | 1 | 8.4864 | 3.3152 | 2.56x |
| filters.threshold_sauvola | 512×512 | int16 | window-15 | 1 | 10.1773 | 2.9009 | 3.51x |
| filters.threshold_niblack | 512×512 | int16 | window-3 | 1 | 12.9409 | 3.9505 | 3.28x |
| filters.threshold_sauvola | 512×512 | int16 | window-3 | 1 | 9.7334 | 2.8691 | 3.39x |
| filters.threshold_niblack | 512×512 | int16 | window-31 | 1 | 7.9436 | 2.0031 | 3.97x |
| filters.threshold_sauvola | 512×512 | int16 | window-31 | 1 | 9.6080 | 3.4104 | 2.82x |
| filters.threshold_niblack | 512×512 | int16 | window-127 | 1 | 13.2767 | 5.4249 | 2.45x |
| filters.threshold_sauvola | 512×512 | int16 | window-127 | 1 | 12.5524 | 4.5734 | 2.74x |
| filters.threshold_niblack | 512×512 | int16 | window-(3, 31) | 1 | 11.7068 | 3.1407 | 3.73x |
| filters.threshold_sauvola | 512×512 | int16 | window-(3, 31) | 1 | 9.7955 | 3.3053 | 2.96x |
| filters.threshold_niblack | 512×512 | int16 | reversed | 1 | 11.2172 | 3.8725 | 2.90x |
| filters.threshold_sauvola | 512×512 | int16 | reversed | 1 | 10.0264 | 2.8193 | 3.56x |
| filters.threshold_niblack | 512×512 | int16 | strided | 1 | 9.9588 | 3.4318 | 2.90x |
| filters.threshold_sauvola | 512×512 | int16 | strided | 1 | 10.2152 | 2.6227 | 3.89x |
| filters.threshold_niblack | 512×512 | int16 | fortran | 1 | 13.4286 | 4.3148 | 3.11x |
| filters.threshold_sauvola | 512×512 | int16 | fortran | 1 | 14.2447 | 2.6819 | 5.31x |
| filters.threshold_niblack | 1×512×512 | int16 | volume | 1 | 20.8628 | 5.9923 | 3.48x |
| filters.threshold_sauvola | 1×512×512 | int16 | volume | 1 | 21.8969 | 5.4545 | 4.01x |
| filters.threshold_niblack | 512×512 | float32 | window-15 | 1 | 6.5341 | 2.5115 | 2.60x |
| filters.threshold_sauvola | 512×512 | float32 | window-15 | 1 | 7.3513 | 2.0299 | 3.62x |
| filters.threshold_niblack | 512×512 | float32 | window-3 | 1 | 7.1529 | 2.1061 | 3.40x |
| filters.threshold_sauvola | 512×512 | float32 | window-3 | 1 | 7.7104 | 2.2021 | 3.50x |
| filters.threshold_niblack | 512×512 | float32 | window-31 | 1 | 8.3393 | 3.9944 | 2.09x |
| filters.threshold_sauvola | 512×512 | float32 | window-31 | 1 | 7.1218 | 2.9658 | 2.40x |
| filters.threshold_niblack | 512×512 | float32 | window-127 | 1 | 11.6377 | 4.0992 | 2.84x |
| filters.threshold_sauvola | 512×512 | float32 | window-127 | 1 | 10.2323 | 4.8197 | 2.12x |
| filters.threshold_niblack | 512×512 | float32 | window-(3, 31) | 1 | 7.6532 | 1.9800 | 3.87x |
| filters.threshold_sauvola | 512×512 | float32 | window-(3, 31) | 1 | 11.5368 | 3.9794 | 2.90x |
| filters.threshold_niblack | 512×512 | float32 | reversed | 1 | 8.7399 | 3.2811 | 2.66x |
| filters.threshold_sauvola | 512×512 | float32 | reversed | 1 | 9.9004 | 3.7674 | 2.63x |
| filters.threshold_niblack | 512×512 | float32 | strided | 1 | 7.8282 | 3.0700 | 2.55x |
| filters.threshold_sauvola | 512×512 | float32 | strided | 1 | 7.4978 | 2.9931 | 2.51x |
| filters.threshold_niblack | 512×512 | float32 | fortran | 1 | 11.1879 | 3.0656 | 3.65x |
| filters.threshold_sauvola | 512×512 | float32 | fortran | 1 | 10.2404 | 2.9323 | 3.49x |
| filters.threshold_niblack | 1×512×512 | float32 | volume | 1 | 16.7073 | 4.8576 | 3.44x |
| filters.threshold_sauvola | 1×512×512 | float32 | volume | 1 | 18.6812 | 5.5976 | 3.34x |
| filters.threshold_niblack | 512×512 | float32 | int64-window | 1 | 7.4225 | 2.6113 | 2.84x |
| filters.threshold_sauvola | 512×512 | float32 | int64-window | 1 | 7.7802 | 2.4165 | 3.22x |
| filters.threshold_niblack | 512×512 | float32 | k-0.0 | 1 | 7.8982 | 2.4729 | 3.19x |
| filters.threshold_sauvola | 512×512 | float32 | k-0.0 | 1 | 7.4653 | 2.3535 | 3.17x |
| filters.threshold_niblack | 512×512 | float32 | k--0.2 | 1 | 6.4253 | 2.1494 | 2.99x |
| filters.threshold_sauvola | 512×512 | float32 | k--0.2 | 1 | 7.5248 | 2.2193 | 3.39x |
| filters.threshold_niblack | 512×512 | float32 | k-np.float32(0.17) | 1 | 7.9408 | 2.8324 | 2.80x |
| filters.threshold_sauvola | 512×512 | float32 | k-np.float32(0.17) | 1 | 7.6105 | 2.4926 | 3.05x |
| filters.threshold_niblack | 512×512 | float32 | k-np.float64(0.17) | 1 | 7.5674 | 2.3376 | 3.24x |
| filters.threshold_sauvola | 512×512 | float32 | k-np.float64(0.17) | 1 | 10.8850 | 2.6596 | 4.09x |
| filters.threshold_sauvola | 512×512 | float32 | r-0.5 | 1 | 7.2932 | 2.5536 | 2.86x |
| filters.threshold_sauvola | 512×512 | float32 | r-128.0 | 1 | 7.1570 | 2.5485 | 2.81x |
| filters.threshold_sauvola | 512×512 | float32 | r-np.float64(0.5) | 1 | 7.6573 | 2.3164 | 3.31x |
| filters.threshold_niblack | 512×512 | float64 | window-15 | 1 | 9.2351 | 2.2232 | 4.15x |
| filters.threshold_sauvola | 512×512 | float64 | window-15 | 1 | 9.4814 | 2.8077 | 3.38x |
| filters.threshold_niblack | 512×512 | float64 | window-3 | 1 | 8.2186 | 2.1803 | 3.77x |
| filters.threshold_sauvola | 512×512 | float64 | window-3 | 1 | 9.2105 | 2.3863 | 3.86x |
| filters.threshold_niblack | 512×512 | float64 | window-31 | 1 | 8.9672 | 2.6315 | 3.41x |
| filters.threshold_sauvola | 512×512 | float64 | window-31 | 1 | 11.3407 | 2.9524 | 3.84x |
| filters.threshold_niblack | 512×512 | float64 | window-127 | 1 | 14.9033 | 5.8730 | 2.54x |
| filters.threshold_sauvola | 512×512 | float64 | window-127 | 1 | 12.5866 | 5.5913 | 2.25x |
| filters.threshold_niblack | 512×512 | float64 | window-(3, 31) | 1 | 8.7469 | 2.4334 | 3.59x |
| filters.threshold_sauvola | 512×512 | float64 | window-(3, 31) | 1 | 11.2268 | 2.5582 | 4.39x |
| filters.threshold_niblack | 512×512 | float64 | reversed | 1 | 8.8564 | 3.8250 | 2.32x |
| filters.threshold_sauvola | 512×512 | float64 | reversed | 1 | 10.0416 | 3.8680 | 2.60x |
| filters.threshold_niblack | 512×512 | float64 | strided | 1 | 10.8458 | 5.2234 | 2.08x |
| filters.threshold_sauvola | 512×512 | float64 | strided | 1 | 13.2781 | 4.7862 | 2.77x |
| filters.threshold_niblack | 512×512 | float64 | fortran | 1 | 13.1678 | 4.7288 | 2.78x |
| filters.threshold_sauvola | 512×512 | float64 | fortran | 1 | 17.0893 | 8.1100 | 2.11x |
| filters.threshold_niblack | 1×512×512 | float64 | volume | 1 | 26.4108 | 7.7588 | 3.40x |
| filters.threshold_sauvola | 1×512×512 | float64 | volume | 1 | 19.3352 | 6.0916 | 3.17x |
| filters.threshold_niblack | 512×512 | float64 | int64-window | 1 | 8.3846 | 3.2597 | 2.57x |
| filters.threshold_sauvola | 512×512 | float64 | int64-window | 1 | 10.4800 | 4.0652 | 2.58x |
| filters.threshold_niblack | 512×512 | float64 | k-0.0 | 1 | 8.4233 | 3.0880 | 2.73x |
| filters.threshold_sauvola | 512×512 | float64 | k-0.0 | 1 | 10.5058 | 3.7147 | 2.83x |
| filters.threshold_niblack | 512×512 | float64 | k--0.2 | 1 | 9.0768 | 3.3442 | 2.71x |
| filters.threshold_sauvola | 512×512 | float64 | k--0.2 | 1 | 10.6315 | 3.9340 | 2.70x |
| filters.threshold_niblack | 512×512 | float64 | k-np.float32(0.17) | 1 | 8.5781 | 3.3478 | 2.56x |
| filters.threshold_sauvola | 512×512 | float64 | k-np.float32(0.17) | 1 | 9.4166 | 3.8514 | 2.44x |
| filters.threshold_niblack | 512×512 | float64 | k-np.float64(0.17) | 1 | 8.2868 | 3.1558 | 2.63x |
| filters.threshold_sauvola | 512×512 | float64 | k-np.float64(0.17) | 1 | 11.0447 | 4.6448 | 2.38x |
| filters.threshold_sauvola | 512×512 | float64 | r-0.5 | 1 | 9.4655 | 3.8316 | 2.47x |
| filters.threshold_sauvola | 512×512 | float64 | r-128.0 | 1 | 23.7082 | 6.1059 | 3.88x |
| filters.threshold_sauvola | 512×512 | float64 | r-np.float64(0.5) | 1 | 24.8189 | 8.8572 | 2.80x |
| filters.threshold_niblack | 512×512 | bool | additional-width | 1 | 21.1481 | 6.0941 | 3.47x |
| filters.threshold_sauvola | 512×512 | bool | additional-width | 1 | 30.9016 | 7.1343 | 4.33x |
| filters.threshold_niblack | 512×512 | int8 | additional-width | 1 | 28.1625 | 7.2762 | 3.87x |
| filters.threshold_sauvola | 512×512 | int8 | additional-width | 1 | 23.0204 | 7.1514 | 3.22x |
| filters.threshold_niblack | 512×512 | uint32 | additional-width | 1 | 17.9180 | 7.7808 | 2.30x |
| filters.threshold_sauvola | 512×512 | uint32 | additional-width | 1 | 21.6187 | 5.6989 | 3.79x |
| filters.threshold_niblack | 512×512 | int32 | additional-width | 1 | 23.2300 | 6.3236 | 3.67x |
| filters.threshold_sauvola | 512×512 | int32 | additional-width | 1 | 14.9936 | 4.3728 | 3.43x |
| filters.threshold_niblack | 512×512 | uint64 | additional-width | 1 | 15.0886 | 6.4361 | 2.34x |
| filters.threshold_sauvola | 512×512 | uint64 | additional-width | 1 | 15.5708 | 6.2746 | 2.48x |
| filters.threshold_niblack | 512×512 | int64 | additional-width | 1 | 12.6763 | 4.5763 | 2.77x |
| filters.threshold_sauvola | 512×512 | int64 | additional-width | 1 | 12.3110 | 3.1504 | 3.91x |
