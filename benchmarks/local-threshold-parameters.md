# Local threshold performance

Reference: original scikit-image 0.26.0. Reference thread pools use one worker; native workers are explicit. Each case checks dtype, shape, warnings and numerical parity with fallback disabled. Timings include dispatch, layout copying, reflected padding, both integral images, local statistics and output allocation. Both scalar and NumPy int64 window metadata are covered, as are strong float64 weights and ranges. All raw samples and slower cases are retained in the JSON.

| API | Shape | Input | Workload | Workers | Original ms | C++ ms | Speedup |
|---|---|---|---|---:|---:|---:|---:|
| filters.threshold_niblack | 512×512 | uint8 | window-15 | 1 | 13.9472 | 3.4116 | 4.09x |
| filters.threshold_sauvola | 512×512 | uint8 | window-15 | 1 | 13.4731 | 3.8485 | 3.50x |
| filters.threshold_niblack | 512×512 | uint8 | window-3 | 1 | 10.2651 | 2.5485 | 4.03x |
| filters.threshold_sauvola | 512×512 | uint8 | window-3 | 1 | 11.4498 | 3.7927 | 3.02x |
| filters.threshold_niblack | 512×512 | uint8 | window-31 | 1 | 11.3006 | 3.0750 | 3.68x |
| filters.threshold_sauvola | 512×512 | uint8 | window-31 | 1 | 15.0114 | 5.2155 | 2.88x |
| filters.threshold_niblack | 512×512 | uint8 | window-127 | 1 | 18.8613 | 5.4687 | 3.45x |
| filters.threshold_sauvola | 512×512 | uint8 | window-127 | 1 | 25.0739 | 6.8419 | 3.66x |
| filters.threshold_niblack | 512×512 | uint8 | window-(3, 31) | 1 | 21.7728 | 4.5077 | 4.83x |
| filters.threshold_sauvola | 512×512 | uint8 | window-(3, 31) | 1 | 22.3627 | 5.8374 | 3.83x |
| filters.threshold_niblack | 512×512 | uint8 | reversed | 1 | 34.4102 | 5.9159 | 5.82x |
| filters.threshold_sauvola | 512×512 | uint8 | reversed | 1 | 29.9389 | 7.3486 | 4.07x |
| filters.threshold_niblack | 512×512 | uint8 | strided | 1 | 33.0850 | 10.1953 | 3.25x |
| filters.threshold_sauvola | 512×512 | uint8 | strided | 1 | 42.6845 | 10.9150 | 3.91x |
| filters.threshold_niblack | 512×512 | uint8 | fortran | 1 | 56.6259 | 7.1730 | 7.89x |
| filters.threshold_sauvola | 512×512 | uint8 | fortran | 1 | 56.9028 | 6.1434 | 9.26x |
| filters.threshold_niblack | 1×512×512 | uint8 | volume | 1 | 56.0028 | 18.2504 | 3.07x |
| filters.threshold_sauvola | 1×512×512 | uint8 | volume | 1 | 58.9976 | 21.0702 | 2.80x |
| filters.threshold_niblack | 512×512 | uint16 | window-15 | 1 | 34.7116 | 8.3787 | 4.14x |
| filters.threshold_sauvola | 512×512 | uint16 | window-15 | 1 | 36.0710 | 8.8075 | 4.10x |
| filters.threshold_niblack | 512×512 | uint16 | window-3 | 1 | 45.3244 | 7.0481 | 6.43x |
| filters.threshold_sauvola | 512×512 | uint16 | window-3 | 1 | 32.9787 | 9.1402 | 3.61x |
| filters.threshold_niblack | 512×512 | uint16 | window-31 | 1 | 29.8987 | 7.4392 | 4.02x |
| filters.threshold_sauvola | 512×512 | uint16 | window-31 | 1 | 20.9281 | 5.4177 | 3.86x |
| filters.threshold_niblack | 512×512 | uint16 | window-127 | 1 | 19.5963 | 7.2308 | 2.71x |
| filters.threshold_sauvola | 512×512 | uint16 | window-127 | 1 | 25.2413 | 8.3159 | 3.04x |
| filters.threshold_niblack | 512×512 | uint16 | window-(3, 31) | 1 | 15.2884 | 3.9663 | 3.85x |
| filters.threshold_sauvola | 512×512 | uint16 | window-(3, 31) | 1 | 22.1813 | 5.6661 | 3.91x |
| filters.threshold_niblack | 512×512 | uint16 | reversed | 1 | 14.7094 | 3.5212 | 4.18x |
| filters.threshold_sauvola | 512×512 | uint16 | reversed | 1 | 19.2029 | 4.9863 | 3.85x |
| filters.threshold_niblack | 512×512 | uint16 | strided | 1 | 15.6329 | 3.7605 | 4.16x |
| filters.threshold_sauvola | 512×512 | uint16 | strided | 1 | 15.3156 | 5.1600 | 2.97x |
| filters.threshold_niblack | 512×512 | uint16 | fortran | 1 | 15.0504 | 3.6270 | 4.15x |
| filters.threshold_sauvola | 512×512 | uint16 | fortran | 1 | 21.5271 | 2.7786 | 7.75x |
| filters.threshold_niblack | 1×512×512 | uint16 | volume | 1 | 25.0367 | 6.4441 | 3.89x |
| filters.threshold_sauvola | 1×512×512 | uint16 | volume | 1 | 21.3941 | 6.4527 | 3.32x |
| filters.threshold_niblack | 512×512 | int16 | window-15 | 1 | 11.2628 | 3.6166 | 3.11x |
| filters.threshold_sauvola | 512×512 | int16 | window-15 | 1 | 12.0537 | 3.2362 | 3.72x |
| filters.threshold_niblack | 512×512 | int16 | window-3 | 1 | 9.6104 | 3.3932 | 2.83x |
| filters.threshold_sauvola | 512×512 | int16 | window-3 | 1 | 9.4666 | 2.1332 | 4.44x |
| filters.threshold_niblack | 512×512 | int16 | window-31 | 1 | 13.7025 | 3.3651 | 4.07x |
| filters.threshold_sauvola | 512×512 | int16 | window-31 | 1 | 23.8680 | 5.7585 | 4.14x |
| filters.threshold_niblack | 512×512 | int16 | window-127 | 1 | 16.4788 | 5.8068 | 2.84x |
| filters.threshold_sauvola | 512×512 | int16 | window-127 | 1 | 14.6234 | 5.3910 | 2.71x |
| filters.threshold_niblack | 512×512 | int16 | window-(3, 31) | 1 | 12.9680 | 3.6911 | 3.51x |
| filters.threshold_sauvola | 512×512 | int16 | window-(3, 31) | 1 | 11.6360 | 3.3952 | 3.43x |
| filters.threshold_niblack | 512×512 | int16 | reversed | 1 | 11.7727 | 3.6808 | 3.20x |
| filters.threshold_sauvola | 512×512 | int16 | reversed | 1 | 10.3995 | 2.3643 | 4.40x |
| filters.threshold_niblack | 512×512 | int16 | strided | 1 | 11.9512 | 3.8159 | 3.13x |
| filters.threshold_sauvola | 512×512 | int16 | strided | 1 | 13.6039 | 4.3196 | 3.15x |
| filters.threshold_niblack | 512×512 | int16 | fortran | 1 | 15.9062 | 4.3411 | 3.66x |
| filters.threshold_sauvola | 512×512 | int16 | fortran | 1 | 41.5566 | 6.6735 | 6.23x |
| filters.threshold_niblack | 1×512×512 | int16 | volume | 1 | 54.5715 | 14.7534 | 3.70x |
| filters.threshold_sauvola | 1×512×512 | int16 | volume | 1 | 49.2752 | 13.9883 | 3.52x |
| filters.threshold_niblack | 512×512 | float32 | window-15 | 1 | 21.6827 | 6.3388 | 3.42x |
| filters.threshold_sauvola | 512×512 | float32 | window-15 | 1 | 36.4512 | 8.2899 | 4.40x |
| filters.threshold_niblack | 512×512 | float32 | window-3 | 1 | 23.8301 | 6.8231 | 3.49x |
| filters.threshold_sauvola | 512×512 | float32 | window-3 | 1 | 24.7378 | 6.2649 | 3.95x |
| filters.threshold_niblack | 512×512 | float32 | window-31 | 1 | 25.9731 | 7.3251 | 3.55x |
| filters.threshold_sauvola | 512×512 | float32 | window-31 | 1 | 23.0586 | 7.3494 | 3.14x |
| filters.threshold_niblack | 512×512 | float32 | window-127 | 1 | 32.5417 | 11.8411 | 2.75x |
| filters.threshold_sauvola | 512×512 | float32 | window-127 | 1 | 32.2216 | 10.9335 | 2.95x |
| filters.threshold_niblack | 512×512 | float32 | window-(3, 31) | 1 | 21.6356 | 6.5024 | 3.33x |
| filters.threshold_sauvola | 512×512 | float32 | window-(3, 31) | 1 | 7.6295 | 2.4010 | 3.18x |
| filters.threshold_niblack | 512×512 | float32 | reversed | 1 | 8.2271 | 3.0573 | 2.69x |
| filters.threshold_sauvola | 512×512 | float32 | reversed | 1 | 8.7592 | 3.3187 | 2.64x |
| filters.threshold_niblack | 512×512 | float32 | strided | 1 | 9.1228 | 3.2596 | 2.80x |
| filters.threshold_sauvola | 512×512 | float32 | strided | 1 | 11.8815 | 4.6236 | 2.57x |
| filters.threshold_niblack | 512×512 | float32 | fortran | 1 | 17.2350 | 4.9172 | 3.51x |
| filters.threshold_sauvola | 512×512 | float32 | fortran | 1 | 15.4643 | 3.7262 | 4.15x |
| filters.threshold_niblack | 1×512×512 | float32 | volume | 1 | 23.5090 | 7.4435 | 3.16x |
| filters.threshold_sauvola | 1×512×512 | float32 | volume | 1 | 19.8449 | 6.8671 | 2.89x |
| filters.threshold_niblack | 512×512 | float32 | int64-window | 1 | 11.7317 | 3.5499 | 3.30x |
| filters.threshold_sauvola | 512×512 | float32 | int64-window | 1 | 8.8589 | 3.2493 | 2.73x |
| filters.threshold_niblack | 512×512 | float32 | k-0.0 | 1 | 7.9353 | 2.8656 | 2.77x |
| filters.threshold_sauvola | 512×512 | float32 | k-0.0 | 1 | 9.4764 | 3.1255 | 3.03x |
| filters.threshold_niblack | 512×512 | float32 | k--0.2 | 1 | 13.7004 | 4.0445 | 3.39x |
| filters.threshold_sauvola | 512×512 | float32 | k--0.2 | 1 | 15.3454 | 4.6936 | 3.27x |
| filters.threshold_niblack | 512×512 | float32 | k-np.float32(0.17) | 1 | 11.3080 | 3.7115 | 3.05x |
| filters.threshold_sauvola | 512×512 | float32 | k-np.float32(0.17) | 1 | 11.0347 | 3.8937 | 2.83x |
| filters.threshold_niblack | 512×512 | float32 | k-np.float64(0.17) | 1 | 12.5237 | 3.5597 | 3.52x |
| filters.threshold_sauvola | 512×512 | float32 | k-np.float64(0.17) | 1 | 13.2461 | 3.3956 | 3.90x |
| filters.threshold_sauvola | 512×512 | float32 | r-0.5 | 1 | 12.3214 | 3.5251 | 3.50x |
| filters.threshold_sauvola | 512×512 | float32 | r-128.0 | 1 | 12.6543 | 3.7569 | 3.37x |
| filters.threshold_sauvola | 512×512 | float32 | r-np.float64(0.5) | 1 | 12.4596 | 4.0966 | 3.04x |
| filters.threshold_niblack | 512×512 | float64 | window-15 | 1 | 15.0127 | 3.5655 | 4.21x |
| filters.threshold_sauvola | 512×512 | float64 | window-15 | 1 | 15.6150 | 4.0100 | 3.89x |
| filters.threshold_niblack | 512×512 | float64 | window-3 | 1 | 14.4896 | 4.3658 | 3.32x |
| filters.threshold_sauvola | 512×512 | float64 | window-3 | 1 | 12.3230 | 3.6285 | 3.40x |
| filters.threshold_niblack | 512×512 | float64 | window-31 | 1 | 14.4103 | 3.6437 | 3.95x |
| filters.threshold_sauvola | 512×512 | float64 | window-31 | 1 | 14.9962 | 4.0011 | 3.75x |
| filters.threshold_niblack | 512×512 | float64 | window-127 | 1 | 17.5964 | 5.2044 | 3.38x |
| filters.threshold_sauvola | 512×512 | float64 | window-127 | 1 | 20.4332 | 8.3276 | 2.45x |
| filters.threshold_niblack | 512×512 | float64 | window-(3, 31) | 1 | 15.2886 | 4.3140 | 3.54x |
| filters.threshold_sauvola | 512×512 | float64 | window-(3, 31) | 1 | 15.6725 | 4.4817 | 3.50x |
| filters.threshold_niblack | 512×512 | float64 | reversed | 1 | 17.1204 | 6.1787 | 2.77x |
| filters.threshold_sauvola | 512×512 | float64 | reversed | 1 | 21.5339 | 7.1486 | 3.01x |
| filters.threshold_niblack | 512×512 | float64 | strided | 1 | 14.2752 | 6.0464 | 2.36x |
| filters.threshold_sauvola | 512×512 | float64 | strided | 1 | 16.0707 | 5.5768 | 2.88x |
| filters.threshold_niblack | 512×512 | float64 | fortran | 1 | 18.9382 | 6.7001 | 2.83x |
| filters.threshold_sauvola | 512×512 | float64 | fortran | 1 | 19.0020 | 7.2416 | 2.62x |
| filters.threshold_niblack | 1×512×512 | float64 | volume | 1 | 26.7848 | 7.7592 | 3.45x |
| filters.threshold_sauvola | 1×512×512 | float64 | volume | 1 | 30.2674 | 10.2875 | 2.94x |
| filters.threshold_niblack | 512×512 | float64 | int64-window | 1 | 13.9931 | 4.0925 | 3.42x |
| filters.threshold_sauvola | 512×512 | float64 | int64-window | 1 | 14.4230 | 4.2748 | 3.37x |
| filters.threshold_niblack | 512×512 | float64 | k-0.0 | 1 | 14.2858 | 4.2012 | 3.40x |
| filters.threshold_sauvola | 512×512 | float64 | k-0.0 | 1 | 13.2021 | 4.4686 | 2.95x |
| filters.threshold_niblack | 512×512 | float64 | k--0.2 | 1 | 12.2788 | 4.2650 | 2.88x |
| filters.threshold_sauvola | 512×512 | float64 | k--0.2 | 1 | 13.5472 | 5.9870 | 2.26x |
| filters.threshold_niblack | 512×512 | float64 | k-np.float32(0.17) | 1 | 13.0274 | 4.7853 | 2.72x |
| filters.threshold_sauvola | 512×512 | float64 | k-np.float32(0.17) | 1 | 9.9182 | 3.3004 | 3.01x |
| filters.threshold_niblack | 512×512 | float64 | k-np.float64(0.17) | 1 | 13.4647 | 4.6990 | 2.87x |
| filters.threshold_sauvola | 512×512 | float64 | k-np.float64(0.17) | 1 | 12.7095 | 3.6758 | 3.46x |
| filters.threshold_sauvola | 512×512 | float64 | r-0.5 | 1 | 13.5922 | 3.7940 | 3.58x |
| filters.threshold_sauvola | 512×512 | float64 | r-128.0 | 1 | 16.4845 | 4.4428 | 3.71x |
| filters.threshold_sauvola | 512×512 | float64 | r-np.float64(0.5) | 1 | 15.9044 | 4.5295 | 3.51x |
| filters.threshold_niblack | 512×512 | bool | additional-width | 1 | 11.4960 | 3.5045 | 3.28x |
| filters.threshold_sauvola | 512×512 | bool | additional-width | 1 | 15.7069 | 4.4015 | 3.57x |
| filters.threshold_niblack | 512×512 | int8 | additional-width | 1 | 17.1637 | 3.6159 | 4.75x |
| filters.threshold_sauvola | 512×512 | int8 | additional-width | 1 | 17.6772 | 4.7874 | 3.69x |
| filters.threshold_niblack | 512×512 | uint32 | additional-width | 1 | 12.0776 | 3.8686 | 3.12x |
| filters.threshold_sauvola | 512×512 | uint32 | additional-width | 1 | 13.2788 | 2.7498 | 4.83x |
| filters.threshold_niblack | 512×512 | int32 | additional-width | 1 | 10.3405 | 3.5523 | 2.91x |
| filters.threshold_sauvola | 512×512 | int32 | additional-width | 1 | 10.4326 | 2.2912 | 4.55x |
| filters.threshold_niblack | 512×512 | uint64 | additional-width | 1 | 11.2303 | 5.0915 | 2.21x |
| filters.threshold_sauvola | 512×512 | uint64 | additional-width | 1 | 12.5568 | 4.1270 | 3.04x |
| filters.threshold_niblack | 512×512 | int64 | additional-width | 1 | 11.2772 | 2.4713 | 4.56x |
| filters.threshold_sauvola | 512×512 | int64 | additional-width | 1 | 11.5346 | 2.8648 | 4.03x |
