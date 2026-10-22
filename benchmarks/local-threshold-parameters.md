# Local threshold performance

Reference: original scikit-image 0.26.0. Reference thread pools use one worker; native workers are explicit. Each case checks dtype, shape, warnings and numerical parity with fallback disabled. Timings include dispatch, layout copying, reflected padding, both integral images, local statistics and output allocation. Both scalar and NumPy int64 window metadata are covered, as are strong float64 weights and ranges. All raw samples and slower cases are retained in the JSON.

| API | Shape | Input | Workload | Workers | Original ms | C++ ms | Speedup |
|---|---|---|---|---:|---:|---:|---:|
| filters.threshold_niblack | 512×512 | uint8 | window-15 | 1 | 7.9535 | 2.1129 | 3.76x |
| filters.threshold_sauvola | 512×512 | uint8 | window-15 | 1 | 9.5875 | 1.9741 | 4.86x |
| filters.threshold_niblack | 512×512 | uint8 | window-3 | 1 | 8.9525 | 2.3374 | 3.83x |
| filters.threshold_sauvola | 512×512 | uint8 | window-3 | 1 | 9.3590 | 1.9379 | 4.83x |
| filters.threshold_niblack | 512×512 | uint8 | window-31 | 1 | 8.4615 | 2.1329 | 3.97x |
| filters.threshold_sauvola | 512×512 | uint8 | window-31 | 1 | 7.8761 | 1.9513 | 4.04x |
| filters.threshold_niblack | 512×512 | uint8 | window-127 | 1 | 11.7536 | 2.8381 | 4.14x |
| filters.threshold_sauvola | 512×512 | uint8 | window-127 | 1 | 12.2682 | 4.0235 | 3.05x |
| filters.threshold_niblack | 512×512 | uint8 | window-(3, 31) | 1 | 7.7514 | 2.1657 | 3.58x |
| filters.threshold_sauvola | 512×512 | uint8 | window-(3, 31) | 1 | 9.5007 | 1.9741 | 4.81x |
| filters.threshold_niblack | 512×512 | uint8 | reversed | 1 | 10.3371 | 2.4024 | 4.30x |
| filters.threshold_sauvola | 512×512 | uint8 | reversed | 1 | 8.1899 | 2.0239 | 4.05x |
| filters.threshold_niblack | 512×512 | uint8 | strided | 1 | 8.2362 | 2.1981 | 3.75x |
| filters.threshold_sauvola | 512×512 | uint8 | strided | 1 | 8.5727 | 2.2181 | 3.86x |
| filters.threshold_niblack | 512×512 | uint8 | fortran | 1 | 15.8476 | 2.5102 | 6.31x |
| filters.threshold_sauvola | 512×512 | uint8 | fortran | 1 | 16.4201 | 3.2408 | 5.07x |
| filters.threshold_niblack | 1×512×512 | uint8 | volume | 1 | 19.8951 | 5.0541 | 3.94x |
| filters.threshold_sauvola | 1×512×512 | uint8 | volume | 1 | 18.6637 | 5.4245 | 3.44x |
| filters.threshold_niblack | 512×512 | uint16 | window-15 | 1 | 7.8794 | 1.8407 | 4.28x |
| filters.threshold_sauvola | 512×512 | uint16 | window-15 | 1 | 10.4361 | 3.3015 | 3.16x |
| filters.threshold_niblack | 512×512 | uint16 | window-3 | 1 | 7.9030 | 1.7720 | 4.46x |
| filters.threshold_sauvola | 512×512 | uint16 | window-3 | 1 | 10.8150 | 3.1084 | 3.48x |
| filters.threshold_niblack | 512×512 | uint16 | window-31 | 1 | 8.1955 | 2.0103 | 4.08x |
| filters.threshold_sauvola | 512×512 | uint16 | window-31 | 1 | 11.6105 | 3.4189 | 3.40x |
| filters.threshold_niblack | 512×512 | uint16 | window-127 | 1 | 11.2220 | 4.1185 | 2.72x |
| filters.threshold_sauvola | 512×512 | uint16 | window-127 | 1 | 12.6550 | 4.4636 | 2.84x |
| filters.threshold_niblack | 512×512 | uint16 | window-(3, 31) | 1 | 8.0772 | 1.8542 | 4.36x |
| filters.threshold_sauvola | 512×512 | uint16 | window-(3, 31) | 1 | 11.5359 | 3.0931 | 3.73x |
| filters.threshold_niblack | 512×512 | uint16 | reversed | 1 | 8.5781 | 3.2277 | 2.66x |
| filters.threshold_sauvola | 512×512 | uint16 | reversed | 1 | 10.4986 | 3.1231 | 3.36x |
| filters.threshold_niblack | 512×512 | uint16 | strided | 1 | 8.3423 | 3.3529 | 2.49x |
| filters.threshold_sauvola | 512×512 | uint16 | strided | 1 | 10.1923 | 3.2988 | 3.09x |
| filters.threshold_niblack | 512×512 | uint16 | fortran | 1 | 12.7025 | 3.2583 | 3.90x |
| filters.threshold_sauvola | 512×512 | uint16 | fortran | 1 | 15.9376 | 3.3686 | 4.73x |
| filters.threshold_niblack | 1×512×512 | uint16 | volume | 1 | 18.4167 | 5.4375 | 3.39x |
| filters.threshold_sauvola | 1×512×512 | uint16 | volume | 1 | 16.8432 | 5.0042 | 3.37x |
| filters.threshold_niblack | 512×512 | int16 | window-15 | 1 | 7.9707 | 3.1349 | 2.54x |
| filters.threshold_sauvola | 512×512 | int16 | window-15 | 1 | 7.9637 | 2.0295 | 3.92x |
| filters.threshold_niblack | 512×512 | int16 | window-3 | 1 | 8.8612 | 1.8153 | 4.88x |
| filters.threshold_sauvola | 512×512 | int16 | window-3 | 1 | 7.8024 | 1.8485 | 4.22x |
| filters.threshold_niblack | 512×512 | int16 | window-31 | 1 | 9.7837 | 2.9317 | 3.34x |
| filters.threshold_sauvola | 512×512 | int16 | window-31 | 1 | 7.7199 | 2.0248 | 3.81x |
| filters.threshold_niblack | 512×512 | int16 | window-127 | 1 | 10.6180 | 3.9179 | 2.71x |
| filters.threshold_sauvola | 512×512 | int16 | window-127 | 1 | 11.1589 | 4.1970 | 2.66x |
| filters.threshold_niblack | 512×512 | int16 | window-(3, 31) | 1 | 9.0810 | 2.7789 | 3.27x |
| filters.threshold_sauvola | 512×512 | int16 | window-(3, 31) | 1 | 9.0116 | 3.2410 | 2.78x |
| filters.threshold_niblack | 512×512 | int16 | reversed | 1 | 8.4707 | 3.2931 | 2.57x |
| filters.threshold_sauvola | 512×512 | int16 | reversed | 1 | 11.4808 | 3.4545 | 3.32x |
| filters.threshold_niblack | 512×512 | int16 | strided | 1 | 9.7611 | 3.1560 | 3.09x |
| filters.threshold_sauvola | 512×512 | int16 | strided | 1 | 10.1176 | 3.2460 | 3.12x |
| filters.threshold_niblack | 512×512 | int16 | fortran | 1 | 12.5456 | 2.9726 | 4.22x |
| filters.threshold_sauvola | 512×512 | int16 | fortran | 1 | 14.1798 | 3.3815 | 4.19x |
| filters.threshold_niblack | 1×512×512 | int16 | volume | 1 | 17.9326 | 4.4345 | 4.04x |
| filters.threshold_sauvola | 1×512×512 | int16 | volume | 1 | 21.9714 | 5.4420 | 4.04x |
| filters.threshold_niblack | 512×512 | float32 | window-15 | 1 | 7.6692 | 1.7891 | 4.29x |
| filters.threshold_sauvola | 512×512 | float32 | window-15 | 1 | 6.9634 | 1.7526 | 3.97x |
| filters.threshold_niblack | 512×512 | float32 | window-3 | 1 | 7.3671 | 2.0238 | 3.64x |
| filters.threshold_sauvola | 512×512 | float32 | window-3 | 1 | 7.2298 | 1.9482 | 3.71x |
| filters.threshold_niblack | 512×512 | float32 | window-31 | 1 | 8.1530 | 3.0335 | 2.69x |
| filters.threshold_sauvola | 512×512 | float32 | window-31 | 1 | 7.9217 | 2.0135 | 3.93x |
| filters.threshold_niblack | 512×512 | float32 | window-127 | 1 | 9.7315 | 4.1114 | 2.37x |
| filters.threshold_sauvola | 512×512 | float32 | window-127 | 1 | 10.2634 | 4.3378 | 2.37x |
| filters.threshold_niblack | 512×512 | float32 | window-(3, 31) | 1 | 7.1142 | 1.7401 | 4.09x |
| filters.threshold_sauvola | 512×512 | float32 | window-(3, 31) | 1 | 7.3408 | 1.8623 | 3.94x |
| filters.threshold_niblack | 512×512 | float32 | reversed | 1 | 6.3057 | 2.1603 | 2.92x |
| filters.threshold_sauvola | 512×512 | float32 | reversed | 1 | 6.5598 | 2.1286 | 3.08x |
| filters.threshold_niblack | 512×512 | float32 | strided | 1 | 6.7128 | 2.0208 | 3.32x |
| filters.threshold_sauvola | 512×512 | float32 | strided | 1 | 6.3156 | 2.3262 | 2.72x |
| filters.threshold_niblack | 512×512 | float32 | fortran | 1 | 9.4411 | 2.1247 | 4.44x |
| filters.threshold_sauvola | 512×512 | float32 | fortran | 1 | 10.0159 | 2.4661 | 4.06x |
| filters.threshold_niblack | 1×512×512 | float32 | volume | 1 | 16.1281 | 4.8122 | 3.35x |
| filters.threshold_sauvola | 1×512×512 | float32 | volume | 1 | 15.6842 | 4.7298 | 3.32x |
| filters.threshold_niblack | 512×512 | float32 | int64-window | 1 | 6.7406 | 2.2655 | 2.98x |
| filters.threshold_sauvola | 512×512 | float32 | int64-window | 1 | 8.2428 | 2.5213 | 3.27x |
| filters.threshold_niblack | 512×512 | float32 | k-0.0 | 1 | 8.7918 | 2.9579 | 2.97x |
| filters.threshold_sauvola | 512×512 | float32 | k-0.0 | 1 | 6.4451 | 2.3424 | 2.75x |
| filters.threshold_niblack | 512×512 | float32 | k--0.2 | 1 | 7.0063 | 2.6709 | 2.62x |
| filters.threshold_sauvola | 512×512 | float32 | k--0.2 | 1 | 6.4184 | 2.1672 | 2.96x |
| filters.threshold_niblack | 512×512 | float32 | k-np.float32(0.17) | 1 | 6.5996 | 2.1068 | 3.13x |
| filters.threshold_sauvola | 512×512 | float32 | k-np.float32(0.17) | 1 | 6.8989 | 2.3731 | 2.91x |
| filters.threshold_niblack | 512×512 | float32 | k-np.float64(0.17) | 1 | 6.9499 | 2.0823 | 3.34x |
| filters.threshold_sauvola | 512×512 | float32 | k-np.float64(0.17) | 1 | 8.1244 | 1.9197 | 4.23x |
| filters.threshold_sauvola | 512×512 | float32 | r-0.5 | 1 | 6.5823 | 1.8561 | 3.55x |
| filters.threshold_sauvola | 512×512 | float32 | r-128.0 | 1 | 7.5979 | 2.6002 | 2.92x |
| filters.threshold_sauvola | 512×512 | float32 | r-np.float64(0.5) | 1 | 14.0849 | 3.1568 | 4.46x |
| filters.threshold_niblack | 512×512 | float64 | window-15 | 1 | 10.4661 | 2.2155 | 4.72x |
| filters.threshold_sauvola | 512×512 | float64 | window-15 | 1 | 11.4453 | 3.0029 | 3.81x |
| filters.threshold_niblack | 512×512 | float64 | window-3 | 1 | 8.2891 | 2.0547 | 4.03x |
| filters.threshold_sauvola | 512×512 | float64 | window-3 | 1 | 8.7748 | 2.2410 | 3.92x |
| filters.threshold_niblack | 512×512 | float64 | window-31 | 1 | 8.3535 | 2.0347 | 4.11x |
| filters.threshold_sauvola | 512×512 | float64 | window-31 | 1 | 8.9309 | 2.5137 | 3.55x |
| filters.threshold_niblack | 512×512 | float64 | window-127 | 1 | 10.2140 | 3.2922 | 3.10x |
| filters.threshold_sauvola | 512×512 | float64 | window-127 | 1 | 12.1458 | 4.7692 | 2.55x |
| filters.threshold_niblack | 512×512 | float64 | window-(3, 31) | 1 | 7.9986 | 2.0813 | 3.84x |
| filters.threshold_sauvola | 512×512 | float64 | window-(3, 31) | 1 | 10.6111 | 2.5440 | 4.17x |
| filters.threshold_niblack | 512×512 | float64 | reversed | 1 | 10.4714 | 3.8107 | 2.75x |
| filters.threshold_sauvola | 512×512 | float64 | reversed | 1 | 8.7719 | 3.7794 | 2.32x |
| filters.threshold_niblack | 512×512 | float64 | strided | 1 | 9.0181 | 3.1478 | 2.86x |
| filters.threshold_sauvola | 512×512 | float64 | strided | 1 | 8.7728 | 3.9403 | 2.23x |
| filters.threshold_niblack | 512×512 | float64 | fortran | 1 | 12.2361 | 4.9329 | 2.48x |
| filters.threshold_sauvola | 512×512 | float64 | fortran | 1 | 14.0951 | 5.0625 | 2.78x |
| filters.threshold_niblack | 1×512×512 | float64 | volume | 1 | 19.0518 | 5.0507 | 3.77x |
| filters.threshold_sauvola | 1×512×512 | float64 | volume | 1 | 18.0028 | 5.2161 | 3.45x |
| filters.threshold_niblack | 512×512 | float64 | int64-window | 1 | 8.3459 | 3.4021 | 2.45x |
| filters.threshold_sauvola | 512×512 | float64 | int64-window | 1 | 10.4655 | 3.4512 | 3.03x |
| filters.threshold_niblack | 512×512 | float64 | k-0.0 | 1 | 8.8165 | 3.2851 | 2.68x |
| filters.threshold_sauvola | 512×512 | float64 | k-0.0 | 1 | 9.7046 | 3.6154 | 2.68x |
| filters.threshold_niblack | 512×512 | float64 | k--0.2 | 1 | 8.5904 | 3.2619 | 2.63x |
| filters.threshold_sauvola | 512×512 | float64 | k--0.2 | 1 | 9.0603 | 3.4298 | 2.64x |
| filters.threshold_niblack | 512×512 | float64 | k-np.float32(0.17) | 1 | 8.9091 | 3.2836 | 2.71x |
| filters.threshold_sauvola | 512×512 | float64 | k-np.float32(0.17) | 1 | 8.6464 | 3.4196 | 2.53x |
| filters.threshold_niblack | 512×512 | float64 | k-np.float64(0.17) | 1 | 8.1981 | 3.4369 | 2.39x |
| filters.threshold_sauvola | 512×512 | float64 | k-np.float64(0.17) | 1 | 8.3863 | 2.1291 | 3.94x |
| filters.threshold_sauvola | 512×512 | float64 | r-0.5 | 1 | 9.9788 | 2.6362 | 3.79x |
| filters.threshold_sauvola | 512×512 | float64 | r-128.0 | 1 | 10.7082 | 2.7643 | 3.87x |
| filters.threshold_sauvola | 512×512 | float64 | r-np.float64(0.5) | 1 | 12.4024 | 3.6437 | 3.40x |
| filters.threshold_niblack | 512×512 | bool | additional-width | 1 | 9.7774 | 1.8486 | 5.29x |
| filters.threshold_sauvola | 512×512 | bool | additional-width | 1 | 9.6304 | 2.1555 | 4.47x |
| filters.threshold_niblack | 512×512 | int8 | additional-width | 1 | 8.9404 | 1.8787 | 4.76x |
| filters.threshold_sauvola | 512×512 | int8 | additional-width | 1 | 8.5385 | 2.0050 | 4.26x |
| filters.threshold_niblack | 512×512 | uint32 | additional-width | 1 | 8.6213 | 2.0683 | 4.17x |
| filters.threshold_sauvola | 512×512 | uint32 | additional-width | 1 | 9.2141 | 1.9650 | 4.69x |
| filters.threshold_niblack | 512×512 | int32 | additional-width | 1 | 9.6040 | 3.3462 | 2.87x |
| filters.threshold_sauvola | 512×512 | int32 | additional-width | 1 | 8.8615 | 2.2574 | 3.93x |
| filters.threshold_niblack | 512×512 | uint64 | additional-width | 1 | 8.0679 | 3.2586 | 2.48x |
| filters.threshold_sauvola | 512×512 | uint64 | additional-width | 1 | 9.5602 | 3.4040 | 2.81x |
| filters.threshold_niblack | 512×512 | int64 | additional-width | 1 | 8.8332 | 1.8768 | 4.71x |
| filters.threshold_sauvola | 512×512 | int64 | additional-width | 1 | 9.2815 | 2.0453 | 4.54x |
