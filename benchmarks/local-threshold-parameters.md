# Local threshold performance

Reference: original scikit-image 0.26.0. Reference thread pools use one worker; native workers are explicit. Each case checks dtype, shape, warnings and numerical parity with fallback disabled. Timings include dispatch, layout copying, reflected padding, both integral images, local statistics and output allocation. Both scalar and NumPy int64 window metadata are covered, as are strong float64 weights and ranges. All raw samples and slower cases are retained in the JSON.

| API | Shape | Input | Workload | Workers | Original ms | C++ ms | Speedup |
|---|---|---|---|---:|---:|---:|---:|
| filters.threshold_niblack | 512×512 | uint8 | window-15 | 1 | 8.3854 | 2.4198 | 3.47x |
| filters.threshold_sauvola | 512×512 | uint8 | window-15 | 1 | 10.2894 | 2.0396 | 5.04x |
| filters.threshold_niblack | 512×512 | uint8 | window-3 | 1 | 7.6790 | 2.2431 | 3.42x |
| filters.threshold_sauvola | 512×512 | uint8 | window-3 | 1 | 9.6770 | 3.7548 | 2.58x |
| filters.threshold_niblack | 512×512 | uint8 | window-31 | 1 | 11.7468 | 2.5997 | 4.52x |
| filters.threshold_sauvola | 512×512 | uint8 | window-31 | 1 | 13.0835 | 4.2142 | 3.10x |
| filters.threshold_niblack | 512×512 | uint8 | window-127 | 1 | 15.2822 | 4.2213 | 3.62x |
| filters.threshold_sauvola | 512×512 | uint8 | window-127 | 1 | 18.3098 | 5.9703 | 3.07x |
| filters.threshold_niblack | 512×512 | uint8 | window-(3, 31) | 1 | 14.2222 | 3.9802 | 3.57x |
| filters.threshold_sauvola | 512×512 | uint8 | window-(3, 31) | 1 | 13.3959 | 4.3510 | 3.08x |
| filters.threshold_niblack | 512×512 | uint8 | reversed | 1 | 14.4789 | 3.9167 | 3.70x |
| filters.threshold_sauvola | 512×512 | uint8 | reversed | 1 | 18.3238 | 4.2743 | 4.29x |
| filters.threshold_niblack | 512×512 | uint8 | strided | 1 | 10.8910 | 3.1982 | 3.41x |
| filters.threshold_sauvola | 512×512 | uint8 | strided | 1 | 13.5839 | 3.2178 | 4.22x |
| filters.threshold_niblack | 512×512 | uint8 | fortran | 1 | 19.9791 | 3.5412 | 5.64x |
| filters.threshold_sauvola | 512×512 | uint8 | fortran | 1 | 17.5297 | 3.1335 | 5.59x |
| filters.threshold_niblack | 1×512×512 | uint8 | volume | 1 | 25.0758 | 7.2999 | 3.44x |
| filters.threshold_sauvola | 1×512×512 | uint8 | volume | 1 | 20.7551 | 7.8477 | 2.64x |
| filters.threshold_niblack | 512×512 | uint16 | window-15 | 1 | 10.3599 | 2.8384 | 3.65x |
| filters.threshold_sauvola | 512×512 | uint16 | window-15 | 1 | 13.8206 | 3.8920 | 3.55x |
| filters.threshold_niblack | 512×512 | uint16 | window-3 | 1 | 14.5991 | 3.5111 | 4.16x |
| filters.threshold_sauvola | 512×512 | uint16 | window-3 | 1 | 14.9149 | 4.4017 | 3.39x |
| filters.threshold_niblack | 512×512 | uint16 | window-31 | 1 | 9.6932 | 4.1478 | 2.34x |
| filters.threshold_sauvola | 512×512 | uint16 | window-31 | 1 | 18.4800 | 4.5084 | 4.10x |
| filters.threshold_niblack | 512×512 | uint16 | window-127 | 1 | 12.8304 | 4.5258 | 2.83x |
| filters.threshold_sauvola | 512×512 | uint16 | window-127 | 1 | 17.7292 | 8.1105 | 2.19x |
| filters.threshold_niblack | 512×512 | uint16 | window-(3, 31) | 1 | 10.0455 | 2.8050 | 3.58x |
| filters.threshold_sauvola | 512×512 | uint16 | window-(3, 31) | 1 | 12.8745 | 3.8390 | 3.35x |
| filters.threshold_niblack | 512×512 | uint16 | reversed | 1 | 12.2398 | 5.0005 | 2.45x |
| filters.threshold_sauvola | 512×512 | uint16 | reversed | 1 | 16.8272 | 5.2812 | 3.19x |
| filters.threshold_niblack | 512×512 | uint16 | strided | 1 | 13.8704 | 3.1201 | 4.45x |
| filters.threshold_sauvola | 512×512 | uint16 | strided | 1 | 15.6063 | 4.4499 | 3.51x |
| filters.threshold_niblack | 512×512 | uint16 | fortran | 1 | 19.6117 | 2.3452 | 8.36x |
| filters.threshold_sauvola | 512×512 | uint16 | fortran | 1 | 20.7223 | 3.8278 | 5.41x |
| filters.threshold_niblack | 1×512×512 | uint16 | volume | 1 | 22.7172 | 7.2666 | 3.13x |
| filters.threshold_sauvola | 1×512×512 | uint16 | volume | 1 | 22.5652 | 6.8466 | 3.30x |
| filters.threshold_niblack | 512×512 | int16 | window-15 | 1 | 9.6812 | 3.6860 | 2.63x |
| filters.threshold_sauvola | 512×512 | int16 | window-15 | 1 | 14.1358 | 2.4797 | 5.70x |
| filters.threshold_niblack | 512×512 | int16 | window-3 | 1 | 9.3823 | 2.1813 | 4.30x |
| filters.threshold_sauvola | 512×512 | int16 | window-3 | 1 | 8.0982 | 2.3301 | 3.48x |
| filters.threshold_niblack | 512×512 | int16 | window-31 | 1 | 10.6167 | 3.3686 | 3.15x |
| filters.threshold_sauvola | 512×512 | int16 | window-31 | 1 | 13.6357 | 4.8813 | 2.79x |
| filters.threshold_niblack | 512×512 | int16 | window-127 | 1 | 15.5226 | 5.2230 | 2.97x |
| filters.threshold_sauvola | 512×512 | int16 | window-127 | 1 | 14.0630 | 4.7881 | 2.94x |
| filters.threshold_niblack | 512×512 | int16 | window-(3, 31) | 1 | 10.4566 | 3.2902 | 3.18x |
| filters.threshold_sauvola | 512×512 | int16 | window-(3, 31) | 1 | 12.9814 | 3.8722 | 3.35x |
| filters.threshold_niblack | 512×512 | int16 | reversed | 1 | 11.4390 | 3.5944 | 3.18x |
| filters.threshold_sauvola | 512×512 | int16 | reversed | 1 | 17.3426 | 4.5281 | 3.83x |
| filters.threshold_niblack | 512×512 | int16 | strided | 1 | 16.4426 | 4.4647 | 3.68x |
| filters.threshold_sauvola | 512×512 | int16 | strided | 1 | 18.0949 | 4.3750 | 4.14x |
| filters.threshold_niblack | 512×512 | int16 | fortran | 1 | 12.6611 | 4.0440 | 3.13x |
| filters.threshold_sauvola | 512×512 | int16 | fortran | 1 | 15.0540 | 3.5771 | 4.21x |
| filters.threshold_niblack | 1×512×512 | int16 | volume | 1 | 20.7283 | 6.0329 | 3.44x |
| filters.threshold_sauvola | 1×512×512 | int16 | volume | 1 | 23.4064 | 7.2790 | 3.22x |
| filters.threshold_niblack | 512×512 | float32 | window-15 | 1 | 11.3109 | 2.9811 | 3.79x |
| filters.threshold_sauvola | 512×512 | float32 | window-15 | 1 | 8.4614 | 2.8592 | 2.96x |
| filters.threshold_niblack | 512×512 | float32 | window-3 | 1 | 7.1295 | 3.2598 | 2.19x |
| filters.threshold_sauvola | 512×512 | float32 | window-3 | 1 | 8.3131 | 2.3184 | 3.59x |
| filters.threshold_niblack | 512×512 | float32 | window-31 | 1 | 8.3331 | 3.3009 | 2.52x |
| filters.threshold_sauvola | 512×512 | float32 | window-31 | 1 | 9.0343 | 2.8401 | 3.18x |
| filters.threshold_niblack | 512×512 | float32 | window-127 | 1 | 15.9004 | 5.7843 | 2.75x |
| filters.threshold_sauvola | 512×512 | float32 | window-127 | 1 | 12.3332 | 4.7359 | 2.60x |
| filters.threshold_niblack | 512×512 | float32 | window-(3, 31) | 1 | 8.7798 | 2.1254 | 4.13x |
| filters.threshold_sauvola | 512×512 | float32 | window-(3, 31) | 1 | 7.7854 | 2.4019 | 3.24x |
| filters.threshold_niblack | 512×512 | float32 | reversed | 1 | 7.5007 | 2.7444 | 2.73x |
| filters.threshold_sauvola | 512×512 | float32 | reversed | 1 | 7.7737 | 2.8018 | 2.77x |
| filters.threshold_niblack | 512×512 | float32 | strided | 1 | 8.4936 | 2.9441 | 2.88x |
| filters.threshold_sauvola | 512×512 | float32 | strided | 1 | 10.6390 | 3.7504 | 2.84x |
| filters.threshold_niblack | 512×512 | float32 | fortran | 1 | 10.2822 | 2.8681 | 3.59x |
| filters.threshold_sauvola | 512×512 | float32 | fortran | 1 | 11.0645 | 2.9607 | 3.74x |
| filters.threshold_niblack | 1×512×512 | float32 | volume | 1 | 20.1880 | 5.1659 | 3.91x |
| filters.threshold_sauvola | 1×512×512 | float32 | volume | 1 | 20.9686 | 6.4091 | 3.27x |
| filters.threshold_niblack | 512×512 | float32 | int64-window | 1 | 10.2681 | 3.5604 | 2.88x |
| filters.threshold_sauvola | 512×512 | float32 | int64-window | 1 | 10.0165 | 3.4043 | 2.94x |
| filters.threshold_niblack | 512×512 | float32 | k-0.0 | 1 | 10.0521 | 3.2306 | 3.11x |
| filters.threshold_sauvola | 512×512 | float32 | k-0.0 | 1 | 9.6101 | 2.8716 | 3.35x |
| filters.threshold_niblack | 512×512 | float32 | k--0.2 | 1 | 9.4736 | 2.9057 | 3.26x |
| filters.threshold_sauvola | 512×512 | float32 | k--0.2 | 1 | 9.3624 | 3.1277 | 2.99x |
| filters.threshold_niblack | 512×512 | float32 | k-np.float32(0.17) | 1 | 8.6847 | 3.4007 | 2.55x |
| filters.threshold_sauvola | 512×512 | float32 | k-np.float32(0.17) | 1 | 8.6238 | 3.1724 | 2.72x |
| filters.threshold_niblack | 512×512 | float32 | k-np.float64(0.17) | 1 | 12.0325 | 3.9920 | 3.01x |
| filters.threshold_sauvola | 512×512 | float32 | k-np.float64(0.17) | 1 | 9.3334 | 3.4336 | 2.72x |
| filters.threshold_sauvola | 512×512 | float32 | r-0.5 | 1 | 7.8330 | 2.5407 | 3.08x |
| filters.threshold_sauvola | 512×512 | float32 | r-128.0 | 1 | 7.6288 | 2.2595 | 3.38x |
| filters.threshold_sauvola | 512×512 | float32 | r-np.float64(0.5) | 1 | 9.3030 | 3.1646 | 2.94x |
| filters.threshold_niblack | 512×512 | float64 | window-15 | 1 | 11.2092 | 2.7141 | 4.13x |
| filters.threshold_sauvola | 512×512 | float64 | window-15 | 1 | 11.7208 | 3.6076 | 3.25x |
| filters.threshold_niblack | 512×512 | float64 | window-3 | 1 | 12.5973 | 3.7550 | 3.35x |
| filters.threshold_sauvola | 512×512 | float64 | window-3 | 1 | 9.9153 | 2.8296 | 3.50x |
| filters.threshold_niblack | 512×512 | float64 | window-31 | 1 | 11.4746 | 3.1720 | 3.62x |
| filters.threshold_sauvola | 512×512 | float64 | window-31 | 1 | 12.3286 | 2.9174 | 4.23x |
| filters.threshold_niblack | 512×512 | float64 | window-127 | 1 | 12.8285 | 3.7344 | 3.44x |
| filters.threshold_sauvola | 512×512 | float64 | window-127 | 1 | 16.0390 | 5.0328 | 3.19x |
| filters.threshold_niblack | 512×512 | float64 | window-(3, 31) | 1 | 9.5053 | 2.7054 | 3.51x |
| filters.threshold_sauvola | 512×512 | float64 | window-(3, 31) | 1 | 12.3410 | 3.8117 | 3.24x |
| filters.threshold_niblack | 512×512 | float64 | reversed | 1 | 10.2344 | 3.7969 | 2.70x |
| filters.threshold_sauvola | 512×512 | float64 | reversed | 1 | 10.0399 | 4.0438 | 2.48x |
| filters.threshold_niblack | 512×512 | float64 | strided | 1 | 12.5361 | 4.2257 | 2.97x |
| filters.threshold_sauvola | 512×512 | float64 | strided | 1 | 11.3013 | 4.2910 | 2.63x |
| filters.threshold_niblack | 512×512 | float64 | fortran | 1 | 17.2300 | 5.4360 | 3.17x |
| filters.threshold_sauvola | 512×512 | float64 | fortran | 1 | 19.5707 | 7.0951 | 2.76x |
| filters.threshold_niblack | 1×512×512 | float64 | volume | 1 | 34.1836 | 9.8403 | 3.47x |
| filters.threshold_sauvola | 1×512×512 | float64 | volume | 1 | 28.0763 | 8.8616 | 3.17x |
| filters.threshold_niblack | 512×512 | float64 | int64-window | 1 | 10.5504 | 4.6179 | 2.28x |
| filters.threshold_sauvola | 512×512 | float64 | int64-window | 1 | 11.5812 | 3.7069 | 3.12x |
| filters.threshold_niblack | 512×512 | float64 | k-0.0 | 1 | 13.2163 | 4.7544 | 2.78x |
| filters.threshold_sauvola | 512×512 | float64 | k-0.0 | 1 | 9.9825 | 2.8873 | 3.46x |
| filters.threshold_niblack | 512×512 | float64 | k--0.2 | 1 | 11.6320 | 4.2602 | 2.73x |
| filters.threshold_sauvola | 512×512 | float64 | k--0.2 | 1 | 9.0961 | 2.3434 | 3.88x |
| filters.threshold_niblack | 512×512 | float64 | k-np.float32(0.17) | 1 | 9.7431 | 4.4153 | 2.21x |
| filters.threshold_sauvola | 512×512 | float64 | k-np.float32(0.17) | 1 | 10.6752 | 2.4845 | 4.30x |
| filters.threshold_niblack | 512×512 | float64 | k-np.float64(0.17) | 1 | 9.9661 | 3.3992 | 2.93x |
| filters.threshold_sauvola | 512×512 | float64 | k-np.float64(0.17) | 1 | 10.7256 | 3.0823 | 3.48x |
| filters.threshold_sauvola | 512×512 | float64 | r-0.5 | 1 | 10.5487 | 3.3039 | 3.19x |
| filters.threshold_sauvola | 512×512 | float64 | r-128.0 | 1 | 9.7435 | 3.5271 | 2.76x |
| filters.threshold_sauvola | 512×512 | float64 | r-np.float64(0.5) | 1 | 12.4506 | 4.2545 | 2.93x |
| filters.threshold_niblack | 512×512 | bool | additional-width | 1 | 11.5496 | 2.0716 | 5.58x |
| filters.threshold_sauvola | 512×512 | bool | additional-width | 1 | 12.0557 | 3.2216 | 3.74x |
| filters.threshold_niblack | 512×512 | int8 | additional-width | 1 | 13.9297 | 4.0866 | 3.41x |
| filters.threshold_sauvola | 512×512 | int8 | additional-width | 1 | 11.7522 | 2.5184 | 4.67x |
| filters.threshold_niblack | 512×512 | uint32 | additional-width | 1 | 7.5029 | 2.8479 | 2.63x |
| filters.threshold_sauvola | 512×512 | uint32 | additional-width | 1 | 9.2415 | 3.3739 | 2.74x |
| filters.threshold_niblack | 512×512 | int32 | additional-width | 1 | 13.2568 | 4.6581 | 2.85x |
| filters.threshold_sauvola | 512×512 | int32 | additional-width | 1 | 13.0037 | 3.6978 | 3.52x |
| filters.threshold_niblack | 512×512 | uint64 | additional-width | 1 | 14.2935 | 5.1651 | 2.77x |
| filters.threshold_sauvola | 512×512 | uint64 | additional-width | 1 | 13.1732 | 4.2917 | 3.07x |
| filters.threshold_niblack | 512×512 | int64 | additional-width | 1 | 9.4289 | 3.9622 | 2.38x |
| filters.threshold_sauvola | 512×512 | int64 | additional-width | 1 | 14.2554 | 4.4534 | 3.20x |
