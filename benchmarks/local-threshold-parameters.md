# Local threshold performance

Reference: original scikit-image 0.26.0. Reference thread pools use one worker; native workers are explicit. Each case checks dtype, shape, warnings and numerical parity with fallback disabled. Timings include dispatch, layout copying, reflected padding, both integral images, local statistics and output allocation. Both scalar and NumPy int64 window metadata are covered, as are strong float64 weights and ranges. All raw samples and slower cases are retained in the JSON.

| API | Shape | Input | Workload | Workers | Original ms | C++ ms | Speedup |
|---|---|---|---|---:|---:|---:|---:|
| filters.threshold_niblack | 512×512 | uint8 | window-15 | 1 | 10.0828 | 2.8106 | 3.59x |
| filters.threshold_sauvola | 512×512 | uint8 | window-15 | 1 | 13.4226 | 4.1042 | 3.27x |
| filters.threshold_niblack | 512×512 | uint8 | window-3 | 1 | 9.6748 | 2.8204 | 3.43x |
| filters.threshold_sauvola | 512×512 | uint8 | window-3 | 1 | 13.2834 | 4.5130 | 2.94x |
| filters.threshold_niblack | 512×512 | uint8 | window-31 | 1 | 12.7409 | 3.2727 | 3.89x |
| filters.threshold_sauvola | 512×512 | uint8 | window-31 | 1 | 14.1577 | 5.2716 | 2.69x |
| filters.threshold_niblack | 512×512 | uint8 | window-127 | 1 | 15.4928 | 4.7348 | 3.27x |
| filters.threshold_sauvola | 512×512 | uint8 | window-127 | 1 | 20.0473 | 7.5646 | 2.65x |
| filters.threshold_niblack | 512×512 | uint8 | window-(3, 31) | 1 | 13.7091 | 3.3872 | 4.05x |
| filters.threshold_sauvola | 512×512 | uint8 | window-(3, 31) | 1 | 12.1134 | 3.5108 | 3.45x |
| filters.threshold_niblack | 512×512 | uint8 | reversed | 1 | 9.4451 | 3.5917 | 2.63x |
| filters.threshold_sauvola | 512×512 | uint8 | reversed | 1 | 13.8543 | 2.8179 | 4.92x |
| filters.threshold_niblack | 512×512 | uint8 | strided | 1 | 12.0183 | 3.0338 | 3.96x |
| filters.threshold_sauvola | 512×512 | uint8 | strided | 1 | 10.2083 | 2.7086 | 3.77x |
| filters.threshold_niblack | 512×512 | uint8 | fortran | 1 | 18.1025 | 3.1474 | 5.75x |
| filters.threshold_sauvola | 512×512 | uint8 | fortran | 1 | 20.8272 | 4.3834 | 4.75x |
| filters.threshold_niblack | 1×512×512 | uint8 | volume | 1 | 25.8394 | 6.8079 | 3.80x |
| filters.threshold_sauvola | 1×512×512 | uint8 | volume | 1 | 35.7440 | 11.5586 | 3.09x |
| filters.threshold_niblack | 512×512 | uint16 | window-15 | 1 | 12.0224 | 3.5649 | 3.37x |
| filters.threshold_sauvola | 512×512 | uint16 | window-15 | 1 | 12.0296 | 3.4090 | 3.53x |
| filters.threshold_niblack | 512×512 | uint16 | window-3 | 1 | 11.3926 | 2.6922 | 4.23x |
| filters.threshold_sauvola | 512×512 | uint16 | window-3 | 1 | 25.1650 | 4.3752 | 5.75x |
| filters.threshold_niblack | 512×512 | uint16 | window-31 | 1 | 25.7552 | 6.2933 | 4.09x |
| filters.threshold_sauvola | 512×512 | uint16 | window-31 | 1 | 27.0410 | 6.9088 | 3.91x |
| filters.threshold_niblack | 512×512 | uint16 | window-127 | 1 | 21.2477 | 7.0233 | 3.03x |
| filters.threshold_sauvola | 512×512 | uint16 | window-127 | 1 | 20.0514 | 6.3947 | 3.14x |
| filters.threshold_niblack | 512×512 | uint16 | window-(3, 31) | 1 | 13.5400 | 4.2905 | 3.16x |
| filters.threshold_sauvola | 512×512 | uint16 | window-(3, 31) | 1 | 16.4590 | 4.0871 | 4.03x |
| filters.threshold_niblack | 512×512 | uint16 | reversed | 1 | 14.0563 | 4.7568 | 2.95x |
| filters.threshold_sauvola | 512×512 | uint16 | reversed | 1 | 13.4809 | 4.0631 | 3.32x |
| filters.threshold_niblack | 512×512 | uint16 | strided | 1 | 13.7278 | 4.6994 | 2.92x |
| filters.threshold_sauvola | 512×512 | uint16 | strided | 1 | 16.9681 | 4.9184 | 3.45x |
| filters.threshold_niblack | 512×512 | uint16 | fortran | 1 | 22.8383 | 4.1995 | 5.44x |
| filters.threshold_sauvola | 512×512 | uint16 | fortran | 1 | 23.9066 | 5.1559 | 4.64x |
| filters.threshold_niblack | 1×512×512 | uint16 | volume | 1 | 28.2932 | 9.3263 | 3.03x |
| filters.threshold_sauvola | 1×512×512 | uint16 | volume | 1 | 37.1494 | 12.4722 | 2.98x |
| filters.threshold_niblack | 512×512 | int16 | window-15 | 1 | 18.4331 | 5.7178 | 3.22x |
| filters.threshold_sauvola | 512×512 | int16 | window-15 | 1 | 21.3186 | 5.0318 | 4.24x |
| filters.threshold_niblack | 512×512 | int16 | window-3 | 1 | 10.4552 | 3.9388 | 2.65x |
| filters.threshold_sauvola | 512×512 | int16 | window-3 | 1 | 10.5455 | 3.5907 | 2.94x |
| filters.threshold_niblack | 512×512 | int16 | window-31 | 1 | 12.1906 | 3.9139 | 3.11x |
| filters.threshold_sauvola | 512×512 | int16 | window-31 | 1 | 18.7850 | 6.4727 | 2.90x |
| filters.threshold_niblack | 512×512 | int16 | window-127 | 1 | 17.3110 | 6.4642 | 2.68x |
| filters.threshold_sauvola | 512×512 | int16 | window-127 | 1 | 16.2675 | 6.4316 | 2.53x |
| filters.threshold_niblack | 512×512 | int16 | window-(3, 31) | 1 | 15.9417 | 4.2393 | 3.76x |
| filters.threshold_sauvola | 512×512 | int16 | window-(3, 31) | 1 | 9.9164 | 2.8406 | 3.49x |
| filters.threshold_niblack | 512×512 | int16 | reversed | 1 | 14.3237 | 4.9089 | 2.92x |
| filters.threshold_sauvola | 512×512 | int16 | reversed | 1 | 15.0401 | 5.1124 | 2.94x |
| filters.threshold_niblack | 512×512 | int16 | strided | 1 | 11.9925 | 5.2073 | 2.30x |
| filters.threshold_sauvola | 512×512 | int16 | strided | 1 | 14.8061 | 4.7355 | 3.13x |
| filters.threshold_niblack | 512×512 | int16 | fortran | 1 | 18.2881 | 3.2771 | 5.58x |
| filters.threshold_sauvola | 512×512 | int16 | fortran | 1 | 26.4067 | 4.8495 | 5.45x |
| filters.threshold_niblack | 1×512×512 | int16 | volume | 1 | 30.3545 | 9.1913 | 3.30x |
| filters.threshold_sauvola | 1×512×512 | int16 | volume | 1 | 29.0802 | 7.9251 | 3.67x |
| filters.threshold_niblack | 512×512 | float32 | window-15 | 1 | 11.2714 | 4.0410 | 2.79x |
| filters.threshold_sauvola | 512×512 | float32 | window-15 | 1 | 10.9517 | 3.3044 | 3.31x |
| filters.threshold_niblack | 512×512 | float32 | window-3 | 1 | 13.3556 | 5.4110 | 2.47x |
| filters.threshold_sauvola | 512×512 | float32 | window-3 | 1 | 9.0804 | 3.0579 | 2.97x |
| filters.threshold_niblack | 512×512 | float32 | window-31 | 1 | 15.8195 | 4.5752 | 3.46x |
| filters.threshold_sauvola | 512×512 | float32 | window-31 | 1 | 11.3927 | 3.0882 | 3.69x |
| filters.threshold_niblack | 512×512 | float32 | window-127 | 1 | 14.3935 | 5.8910 | 2.44x |
| filters.threshold_sauvola | 512×512 | float32 | window-127 | 1 | 12.7285 | 4.9321 | 2.58x |
| filters.threshold_niblack | 512×512 | float32 | window-(3, 31) | 1 | 9.4619 | 3.1969 | 2.96x |
| filters.threshold_sauvola | 512×512 | float32 | window-(3, 31) | 1 | 7.3276 | 2.6106 | 2.81x |
| filters.threshold_niblack | 512×512 | float32 | reversed | 1 | 8.9584 | 3.3850 | 2.65x |
| filters.threshold_sauvola | 512×512 | float32 | reversed | 1 | 8.9174 | 2.4961 | 3.57x |
| filters.threshold_niblack | 512×512 | float32 | strided | 1 | 8.9637 | 2.9100 | 3.08x |
| filters.threshold_sauvola | 512×512 | float32 | strided | 1 | 7.4492 | 3.0356 | 2.45x |
| filters.threshold_niblack | 512×512 | float32 | fortran | 1 | 12.8402 | 3.8415 | 3.34x |
| filters.threshold_sauvola | 512×512 | float32 | fortran | 1 | 15.3459 | 3.7610 | 4.08x |
| filters.threshold_niblack | 1×512×512 | float32 | volume | 1 | 26.7180 | 8.5561 | 3.12x |
| filters.threshold_sauvola | 1×512×512 | float32 | volume | 1 | 30.8883 | 8.2596 | 3.74x |
| filters.threshold_niblack | 512×512 | float32 | int64-window | 1 | 10.0369 | 3.1924 | 3.14x |
| filters.threshold_sauvola | 512×512 | float32 | int64-window | 1 | 9.7850 | 3.1940 | 3.06x |
| filters.threshold_niblack | 512×512 | float32 | k-0.0 | 1 | 12.8457 | 3.8930 | 3.30x |
| filters.threshold_sauvola | 512×512 | float32 | k-0.0 | 1 | 8.3783 | 2.9695 | 2.82x |
| filters.threshold_niblack | 512×512 | float32 | k--0.2 | 1 | 8.5097 | 3.1753 | 2.68x |
| filters.threshold_sauvola | 512×512 | float32 | k--0.2 | 1 | 9.1046 | 2.5657 | 3.55x |
| filters.threshold_niblack | 512×512 | float32 | k-np.float32(0.17) | 1 | 8.9640 | 3.0924 | 2.90x |
| filters.threshold_sauvola | 512×512 | float32 | k-np.float32(0.17) | 1 | 8.7733 | 2.6310 | 3.33x |
| filters.threshold_niblack | 512×512 | float32 | k-np.float64(0.17) | 1 | 8.9332 | 2.5913 | 3.45x |
| filters.threshold_sauvola | 512×512 | float32 | k-np.float64(0.17) | 1 | 9.7739 | 2.4982 | 3.91x |
| filters.threshold_sauvola | 512×512 | float32 | r-0.5 | 1 | 8.9891 | 2.5451 | 3.53x |
| filters.threshold_sauvola | 512×512 | float32 | r-128.0 | 1 | 7.6634 | 2.5294 | 3.03x |
| filters.threshold_sauvola | 512×512 | float32 | r-np.float64(0.5) | 1 | 9.2169 | 2.8001 | 3.29x |
| filters.threshold_niblack | 512×512 | float64 | window-15 | 1 | 9.7202 | 2.4351 | 3.99x |
| filters.threshold_sauvola | 512×512 | float64 | window-15 | 1 | 13.3315 | 2.9876 | 4.46x |
| filters.threshold_niblack | 512×512 | float64 | window-3 | 1 | 10.5280 | 2.2746 | 4.63x |
| filters.threshold_sauvola | 512×512 | float64 | window-3 | 1 | 11.1756 | 3.2454 | 3.44x |
| filters.threshold_niblack | 512×512 | float64 | window-31 | 1 | 13.2646 | 4.2519 | 3.12x |
| filters.threshold_sauvola | 512×512 | float64 | window-31 | 1 | 10.4700 | 2.9716 | 3.52x |
| filters.threshold_niblack | 512×512 | float64 | window-127 | 1 | 17.8144 | 5.5823 | 3.19x |
| filters.threshold_sauvola | 512×512 | float64 | window-127 | 1 | 13.0719 | 5.8094 | 2.25x |
| filters.threshold_niblack | 512×512 | float64 | window-(3, 31) | 1 | 9.8073 | 3.3284 | 2.95x |
| filters.threshold_sauvola | 512×512 | float64 | window-(3, 31) | 1 | 12.8198 | 2.9228 | 4.39x |
| filters.threshold_niblack | 512×512 | float64 | reversed | 1 | 12.2529 | 5.1093 | 2.40x |
| filters.threshold_sauvola | 512×512 | float64 | reversed | 1 | 10.1437 | 4.3688 | 2.32x |
| filters.threshold_niblack | 512×512 | float64 | strided | 1 | 10.3905 | 4.2163 | 2.46x |
| filters.threshold_sauvola | 512×512 | float64 | strided | 1 | 11.4531 | 4.7696 | 2.40x |
| filters.threshold_niblack | 512×512 | float64 | fortran | 1 | 14.5866 | 5.1234 | 2.85x |
| filters.threshold_sauvola | 512×512 | float64 | fortran | 1 | 15.4186 | 5.1791 | 2.98x |
| filters.threshold_niblack | 1×512×512 | float64 | volume | 1 | 19.5823 | 5.3015 | 3.69x |
| filters.threshold_sauvola | 1×512×512 | float64 | volume | 1 | 23.5996 | 7.0902 | 3.33x |
| filters.threshold_niblack | 512×512 | float64 | int64-window | 1 | 11.8023 | 5.0914 | 2.32x |
| filters.threshold_sauvola | 512×512 | float64 | int64-window | 1 | 12.3563 | 4.6011 | 2.69x |
| filters.threshold_niblack | 512×512 | float64 | k-0.0 | 1 | 13.6793 | 4.4032 | 3.11x |
| filters.threshold_sauvola | 512×512 | float64 | k-0.0 | 1 | 10.2163 | 3.8194 | 2.67x |
| filters.threshold_niblack | 512×512 | float64 | k--0.2 | 1 | 8.9194 | 3.3623 | 2.65x |
| filters.threshold_sauvola | 512×512 | float64 | k--0.2 | 1 | 10.2214 | 3.7436 | 2.73x |
| filters.threshold_niblack | 512×512 | float64 | k-np.float32(0.17) | 1 | 9.7551 | 3.7803 | 2.58x |
| filters.threshold_sauvola | 512×512 | float64 | k-np.float32(0.17) | 1 | 16.6875 | 5.5953 | 2.98x |
| filters.threshold_niblack | 512×512 | float64 | k-np.float64(0.17) | 1 | 8.4280 | 3.4190 | 2.47x |
| filters.threshold_sauvola | 512×512 | float64 | k-np.float64(0.17) | 1 | 10.2289 | 3.8178 | 2.68x |
| filters.threshold_sauvola | 512×512 | float64 | r-0.5 | 1 | 10.0869 | 2.4825 | 4.06x |
| filters.threshold_sauvola | 512×512 | float64 | r-128.0 | 1 | 9.9692 | 3.5138 | 2.84x |
| filters.threshold_sauvola | 512×512 | float64 | r-np.float64(0.5) | 1 | 9.0954 | 3.4024 | 2.67x |
| filters.threshold_niblack | 512×512 | bool | additional-width | 1 | 9.6759 | 2.0505 | 4.72x |
| filters.threshold_sauvola | 512×512 | bool | additional-width | 1 | 8.6398 | 2.0733 | 4.17x |
| filters.threshold_niblack | 512×512 | int8 | additional-width | 1 | 10.4093 | 2.2751 | 4.58x |
| filters.threshold_sauvola | 512×512 | int8 | additional-width | 1 | 8.7603 | 2.2063 | 3.97x |
| filters.threshold_niblack | 512×512 | uint32 | additional-width | 1 | 8.8387 | 2.1400 | 4.13x |
| filters.threshold_sauvola | 512×512 | uint32 | additional-width | 1 | 9.1373 | 2.1484 | 4.25x |
| filters.threshold_niblack | 512×512 | int32 | additional-width | 1 | 10.3781 | 3.6341 | 2.86x |
| filters.threshold_sauvola | 512×512 | int32 | additional-width | 1 | 10.3872 | 2.4795 | 4.19x |
| filters.threshold_niblack | 512×512 | uint64 | additional-width | 1 | 8.0017 | 3.6363 | 2.20x |
| filters.threshold_sauvola | 512×512 | uint64 | additional-width | 1 | 9.9441 | 3.5135 | 2.83x |
| filters.threshold_niblack | 512×512 | int64 | additional-width | 1 | 9.6463 | 2.0898 | 4.62x |
| filters.threshold_sauvola | 512×512 | int64 | additional-width | 1 | 8.7654 | 2.2265 | 3.94x |
