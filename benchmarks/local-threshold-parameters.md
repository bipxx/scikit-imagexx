# Local threshold performance

Reference: original scikit-image 0.26.0. Reference thread pools use one worker; native workers are explicit. Each case checks dtype, shape, warnings and numerical parity with fallback disabled. Timings include dispatch, layout copying, reflected padding, both integral images, local statistics and output allocation. Both scalar and NumPy int64 window metadata are covered, as are strong float64 weights and ranges. All raw samples and slower cases are retained in the JSON.

| API | Shape | Input | Workload | Workers | Original ms | C++ ms | Speedup |
|---|---|---|---|---:|---:|---:|---:|
| filters.threshold_niblack | 512×512 | uint8 | window-15 | 1 | 8.5841 | 2.0366 | 4.21x |
| filters.threshold_sauvola | 512×512 | uint8 | window-15 | 1 | 9.3049 | 1.8539 | 5.02x |
| filters.threshold_niblack | 512×512 | uint8 | window-3 | 1 | 7.6338 | 2.0931 | 3.65x |
| filters.threshold_sauvola | 512×512 | uint8 | window-3 | 1 | 8.5941 | 1.8564 | 4.63x |
| filters.threshold_niblack | 512×512 | uint8 | window-31 | 1 | 8.1212 | 2.3971 | 3.39x |
| filters.threshold_sauvola | 512×512 | uint8 | window-31 | 1 | 9.4169 | 2.1134 | 4.46x |
| filters.threshold_niblack | 512×512 | uint8 | window-127 | 1 | 11.5058 | 3.3811 | 3.40x |
| filters.threshold_sauvola | 512×512 | uint8 | window-127 | 1 | 13.9590 | 4.8693 | 2.87x |
| filters.threshold_niblack | 512×512 | uint8 | window-(3, 31) | 1 | 8.1579 | 2.3064 | 3.54x |
| filters.threshold_sauvola | 512×512 | uint8 | window-(3, 31) | 1 | 8.6578 | 2.0666 | 4.19x |
| filters.threshold_niblack | 512×512 | uint8 | reversed | 1 | 10.4868 | 3.3217 | 3.16x |
| filters.threshold_sauvola | 512×512 | uint8 | reversed | 1 | 10.0975 | 2.7038 | 3.73x |
| filters.threshold_niblack | 512×512 | uint8 | strided | 1 | 9.3867 | 2.4204 | 3.88x |
| filters.threshold_sauvola | 512×512 | uint8 | strided | 1 | 13.9966 | 3.8346 | 3.65x |
| filters.threshold_niblack | 512×512 | uint8 | fortran | 1 | 13.8139 | 2.4844 | 5.56x |
| filters.threshold_sauvola | 512×512 | uint8 | fortran | 1 | 37.1455 | 8.2862 | 4.48x |
| filters.threshold_niblack | 1×512×512 | uint8 | volume | 1 | 23.2738 | 7.6316 | 3.05x |
| filters.threshold_sauvola | 1×512×512 | uint8 | volume | 1 | 26.0355 | 7.8784 | 3.30x |
| filters.threshold_niblack | 512×512 | uint16 | window-15 | 1 | 11.0667 | 2.9445 | 3.76x |
| filters.threshold_sauvola | 512×512 | uint16 | window-15 | 1 | 11.8516 | 3.9172 | 3.03x |
| filters.threshold_niblack | 512×512 | uint16 | window-3 | 1 | 15.7962 | 3.5266 | 4.48x |
| filters.threshold_sauvola | 512×512 | uint16 | window-3 | 1 | 12.3098 | 3.6714 | 3.35x |
| filters.threshold_niblack | 512×512 | uint16 | window-31 | 1 | 11.8884 | 2.7830 | 4.27x |
| filters.threshold_sauvola | 512×512 | uint16 | window-31 | 1 | 20.7265 | 5.3818 | 3.85x |
| filters.threshold_niblack | 512×512 | uint16 | window-127 | 1 | 14.8100 | 4.3940 | 3.37x |
| filters.threshold_sauvola | 512×512 | uint16 | window-127 | 1 | 21.4234 | 6.7446 | 3.18x |
| filters.threshold_niblack | 512×512 | uint16 | window-(3, 31) | 1 | 24.9820 | 4.7312 | 5.28x |
| filters.threshold_sauvola | 512×512 | uint16 | window-(3, 31) | 1 | 13.8822 | 3.6949 | 3.76x |
| filters.threshold_niblack | 512×512 | uint16 | reversed | 1 | 15.2532 | 4.6954 | 3.25x |
| filters.threshold_sauvola | 512×512 | uint16 | reversed | 1 | 9.3678 | 2.4432 | 3.83x |
| filters.threshold_niblack | 512×512 | uint16 | strided | 1 | 11.0963 | 3.9040 | 2.84x |
| filters.threshold_sauvola | 512×512 | uint16 | strided | 1 | 13.0380 | 3.6841 | 3.54x |
| filters.threshold_niblack | 512×512 | uint16 | fortran | 1 | 17.2670 | 3.8681 | 4.46x |
| filters.threshold_sauvola | 512×512 | uint16 | fortran | 1 | 21.4991 | 3.5526 | 6.05x |
| filters.threshold_niblack | 1×512×512 | uint16 | volume | 1 | 23.4024 | 7.3379 | 3.19x |
| filters.threshold_sauvola | 1×512×512 | uint16 | volume | 1 | 33.2143 | 8.2747 | 4.01x |
| filters.threshold_niblack | 512×512 | int16 | window-15 | 1 | 11.6363 | 3.7370 | 3.11x |
| filters.threshold_sauvola | 512×512 | int16 | window-15 | 1 | 13.5111 | 3.9444 | 3.43x |
| filters.threshold_niblack | 512×512 | int16 | window-3 | 1 | 11.5531 | 2.2551 | 5.12x |
| filters.threshold_sauvola | 512×512 | int16 | window-3 | 1 | 13.5040 | 2.5358 | 5.33x |
| filters.threshold_niblack | 512×512 | int16 | window-31 | 1 | 12.8680 | 3.4902 | 3.69x |
| filters.threshold_sauvola | 512×512 | int16 | window-31 | 1 | 12.3559 | 2.9072 | 4.25x |
| filters.threshold_niblack | 512×512 | int16 | window-127 | 1 | 15.7427 | 5.1517 | 3.06x |
| filters.threshold_sauvola | 512×512 | int16 | window-127 | 1 | 14.1527 | 5.3399 | 2.65x |
| filters.threshold_niblack | 512×512 | int16 | window-(3, 31) | 1 | 15.7887 | 4.1447 | 3.81x |
| filters.threshold_sauvola | 512×512 | int16 | window-(3, 31) | 1 | 12.3315 | 3.9315 | 3.14x |
| filters.threshold_niblack | 512×512 | int16 | reversed | 1 | 15.3671 | 4.7117 | 3.26x |
| filters.threshold_sauvola | 512×512 | int16 | reversed | 1 | 19.4712 | 4.4410 | 4.38x |
| filters.threshold_niblack | 512×512 | int16 | strided | 1 | 13.4633 | 2.5819 | 5.21x |
| filters.threshold_sauvola | 512×512 | int16 | strided | 1 | 13.8944 | 4.1722 | 3.33x |
| filters.threshold_niblack | 512×512 | int16 | fortran | 1 | 14.8686 | 3.9091 | 3.80x |
| filters.threshold_sauvola | 512×512 | int16 | fortran | 1 | 18.6879 | 3.6220 | 5.16x |
| filters.threshold_niblack | 1×512×512 | int16 | volume | 1 | 24.0344 | 7.2934 | 3.30x |
| filters.threshold_sauvola | 1×512×512 | int16 | volume | 1 | 21.5925 | 5.4897 | 3.93x |
| filters.threshold_niblack | 512×512 | float32 | window-15 | 1 | 6.9203 | 2.3871 | 2.90x |
| filters.threshold_sauvola | 512×512 | float32 | window-15 | 1 | 8.8512 | 3.0470 | 2.90x |
| filters.threshold_niblack | 512×512 | float32 | window-3 | 1 | 7.9921 | 2.5035 | 3.19x |
| filters.threshold_sauvola | 512×512 | float32 | window-3 | 1 | 9.1109 | 3.0124 | 3.02x |
| filters.threshold_niblack | 512×512 | float32 | window-31 | 1 | 7.5746 | 2.4158 | 3.14x |
| filters.threshold_sauvola | 512×512 | float32 | window-31 | 1 | 7.4824 | 2.3273 | 3.22x |
| filters.threshold_niblack | 512×512 | float32 | window-127 | 1 | 10.9968 | 4.9048 | 2.24x |
| filters.threshold_sauvola | 512×512 | float32 | window-127 | 1 | 12.1286 | 4.9006 | 2.47x |
| filters.threshold_niblack | 512×512 | float32 | window-(3, 31) | 1 | 7.5750 | 1.9727 | 3.84x |
| filters.threshold_sauvola | 512×512 | float32 | window-(3, 31) | 1 | 10.1844 | 2.5671 | 3.97x |
| filters.threshold_niblack | 512×512 | float32 | reversed | 1 | 7.9083 | 2.8166 | 2.81x |
| filters.threshold_sauvola | 512×512 | float32 | reversed | 1 | 7.3283 | 2.6562 | 2.76x |
| filters.threshold_niblack | 512×512 | float32 | strided | 1 | 8.6255 | 3.4608 | 2.49x |
| filters.threshold_sauvola | 512×512 | float32 | strided | 1 | 8.4042 | 2.9341 | 2.86x |
| filters.threshold_niblack | 512×512 | float32 | fortran | 1 | 15.6039 | 3.3255 | 4.69x |
| filters.threshold_sauvola | 512×512 | float32 | fortran | 1 | 13.5640 | 2.9431 | 4.61x |
| filters.threshold_niblack | 1×512×512 | float32 | volume | 1 | 16.3543 | 4.8302 | 3.39x |
| filters.threshold_sauvola | 1×512×512 | float32 | volume | 1 | 20.5462 | 5.9800 | 3.44x |
| filters.threshold_niblack | 512×512 | float32 | int64-window | 1 | 11.3717 | 4.2962 | 2.65x |
| filters.threshold_sauvola | 512×512 | float32 | int64-window | 1 | 16.1975 | 4.2804 | 3.78x |
| filters.threshold_niblack | 512×512 | float32 | k-0.0 | 1 | 19.8449 | 6.5426 | 3.03x |
| filters.threshold_sauvola | 512×512 | float32 | k-0.0 | 1 | 16.2226 | 5.9077 | 2.75x |
| filters.threshold_niblack | 512×512 | float32 | k--0.2 | 1 | 11.1478 | 3.8108 | 2.93x |
| filters.threshold_sauvola | 512×512 | float32 | k--0.2 | 1 | 7.8800 | 2.7645 | 2.85x |
| filters.threshold_niblack | 512×512 | float32 | k-np.float32(0.17) | 1 | 9.8752 | 3.1455 | 3.14x |
| filters.threshold_sauvola | 512×512 | float32 | k-np.float32(0.17) | 1 | 12.5888 | 3.3790 | 3.73x |
| filters.threshold_niblack | 512×512 | float32 | k-np.float64(0.17) | 1 | 16.3796 | 3.9359 | 4.16x |
| filters.threshold_sauvola | 512×512 | float32 | k-np.float64(0.17) | 1 | 13.2972 | 4.2412 | 3.14x |
| filters.threshold_sauvola | 512×512 | float32 | r-0.5 | 1 | 11.4698 | 4.0498 | 2.83x |
| filters.threshold_sauvola | 512×512 | float32 | r-128.0 | 1 | 10.5723 | 3.1264 | 3.38x |
| filters.threshold_sauvola | 512×512 | float32 | r-np.float64(0.5) | 1 | 13.5475 | 3.6505 | 3.71x |
| filters.threshold_niblack | 512×512 | float64 | window-15 | 1 | 13.2068 | 2.6303 | 5.02x |
| filters.threshold_sauvola | 512×512 | float64 | window-15 | 1 | 13.9324 | 3.8819 | 3.59x |
| filters.threshold_niblack | 512×512 | float64 | window-3 | 1 | 9.8428 | 2.3957 | 4.11x |
| filters.threshold_sauvola | 512×512 | float64 | window-3 | 1 | 12.0395 | 2.5839 | 4.66x |
| filters.threshold_niblack | 512×512 | float64 | window-31 | 1 | 12.2275 | 2.5979 | 4.71x |
| filters.threshold_sauvola | 512×512 | float64 | window-31 | 1 | 11.9963 | 2.4369 | 4.92x |
| filters.threshold_niblack | 512×512 | float64 | window-127 | 1 | 14.7484 | 4.9312 | 2.99x |
| filters.threshold_sauvola | 512×512 | float64 | window-127 | 1 | 15.2737 | 5.5392 | 2.76x |
| filters.threshold_niblack | 512×512 | float64 | window-(3, 31) | 1 | 8.6776 | 2.0254 | 4.28x |
| filters.threshold_sauvola | 512×512 | float64 | window-(3, 31) | 1 | 9.5356 | 2.4571 | 3.88x |
| filters.threshold_niblack | 512×512 | float64 | reversed | 1 | 10.1895 | 3.6646 | 2.78x |
| filters.threshold_sauvola | 512×512 | float64 | reversed | 1 | 9.3706 | 3.7906 | 2.47x |
| filters.threshold_niblack | 512×512 | float64 | strided | 1 | 9.1356 | 3.6766 | 2.48x |
| filters.threshold_sauvola | 512×512 | float64 | strided | 1 | 10.1537 | 4.7743 | 2.13x |
| filters.threshold_niblack | 512×512 | float64 | fortran | 1 | 21.2855 | 6.3510 | 3.35x |
| filters.threshold_sauvola | 512×512 | float64 | fortran | 1 | 22.0807 | 6.6218 | 3.33x |
| filters.threshold_niblack | 1×512×512 | float64 | volume | 1 | 24.5464 | 6.9345 | 3.54x |
| filters.threshold_sauvola | 1×512×512 | float64 | volume | 1 | 18.1945 | 5.0064 | 3.63x |
| filters.threshold_niblack | 512×512 | float64 | int64-window | 1 | 10.6421 | 3.7587 | 2.83x |
| filters.threshold_sauvola | 512×512 | float64 | int64-window | 1 | 9.1793 | 3.3796 | 2.72x |
| filters.threshold_niblack | 512×512 | float64 | k-0.0 | 1 | 8.2121 | 3.2014 | 2.57x |
| filters.threshold_sauvola | 512×512 | float64 | k-0.0 | 1 | 8.9856 | 3.4091 | 2.64x |
| filters.threshold_niblack | 512×512 | float64 | k--0.2 | 1 | 8.3428 | 3.1136 | 2.68x |
| filters.threshold_sauvola | 512×512 | float64 | k--0.2 | 1 | 10.7553 | 4.1727 | 2.58x |
| filters.threshold_niblack | 512×512 | float64 | k-np.float32(0.17) | 1 | 9.2271 | 3.4311 | 2.69x |
| filters.threshold_sauvola | 512×512 | float64 | k-np.float32(0.17) | 1 | 9.7624 | 3.4964 | 2.79x |
| filters.threshold_niblack | 512×512 | float64 | k-np.float64(0.17) | 1 | 8.2737 | 3.3525 | 2.47x |
| filters.threshold_sauvola | 512×512 | float64 | k-np.float64(0.17) | 1 | 8.3839 | 2.2347 | 3.75x |
| filters.threshold_sauvola | 512×512 | float64 | r-0.5 | 1 | 10.2979 | 2.4280 | 4.24x |
| filters.threshold_sauvola | 512×512 | float64 | r-128.0 | 1 | 31.3308 | 7.8507 | 3.99x |
| filters.threshold_sauvola | 512×512 | float64 | r-np.float64(0.5) | 1 | 9.6487 | 3.4024 | 2.84x |
| filters.threshold_niblack | 512×512 | bool | additional-width | 1 | 9.9647 | 2.0810 | 4.79x |
| filters.threshold_sauvola | 512×512 | bool | additional-width | 1 | 13.4622 | 4.2335 | 3.18x |
| filters.threshold_niblack | 512×512 | int8 | additional-width | 1 | 11.7508 | 2.6846 | 4.38x |
| filters.threshold_sauvola | 512×512 | int8 | additional-width | 1 | 9.8936 | 2.4180 | 4.09x |
| filters.threshold_niblack | 512×512 | uint32 | additional-width | 1 | 9.9492 | 2.5179 | 3.95x |
| filters.threshold_sauvola | 512×512 | uint32 | additional-width | 1 | 9.7633 | 2.3299 | 4.19x |
| filters.threshold_niblack | 512×512 | int32 | additional-width | 1 | 11.3385 | 3.6314 | 3.12x |
| filters.threshold_sauvola | 512×512 | int32 | additional-width | 1 | 12.0415 | 2.9667 | 4.06x |
| filters.threshold_niblack | 512×512 | uint64 | additional-width | 1 | 10.2947 | 3.8663 | 2.66x |
| filters.threshold_sauvola | 512×512 | uint64 | additional-width | 1 | 10.6434 | 3.7408 | 2.85x |
| filters.threshold_niblack | 512×512 | int64 | additional-width | 1 | 7.9543 | 2.5394 | 3.13x |
| filters.threshold_sauvola | 512×512 | int64 | additional-width | 1 | 10.6904 | 2.1783 | 4.91x |
