# Local threshold performance

Reference: original scikit-image 0.26.0. Reference thread pools use one worker; native workers are explicit. Each case checks dtype, shape, warnings and numerical parity with fallback disabled. Timings include dispatch, layout copying, reflected padding, both integral images, local statistics and output allocation. Both scalar and NumPy int64 window metadata are covered, as are strong float64 weights and ranges. All raw samples and slower cases are retained in the JSON.

| API | Shape | Input | Workload | Workers | Original ms | C++ ms | Speedup |
|---|---|---|---|---:|---:|---:|---:|
| filters.threshold_niblack | 512×512 | uint8 | window-15 | 1 | 10.1210 | 2.8364 | 3.57x |
| filters.threshold_sauvola | 512×512 | uint8 | window-15 | 1 | 11.1527 | 2.6857 | 4.15x |
| filters.threshold_niblack | 512×512 | uint8 | window-3 | 1 | 10.2086 | 2.6527 | 3.85x |
| filters.threshold_sauvola | 512×512 | uint8 | window-3 | 1 | 9.1041 | 3.6252 | 2.51x |
| filters.threshold_niblack | 512×512 | uint8 | window-31 | 1 | 9.1553 | 2.4786 | 3.69x |
| filters.threshold_sauvola | 512×512 | uint8 | window-31 | 1 | 11.1387 | 4.4271 | 2.52x |
| filters.threshold_niblack | 512×512 | uint8 | window-127 | 1 | 13.7613 | 3.6642 | 3.76x |
| filters.threshold_sauvola | 512×512 | uint8 | window-127 | 1 | 16.9933 | 6.0135 | 2.83x |
| filters.threshold_niblack | 512×512 | uint8 | window-(3, 31) | 1 | 9.8329 | 3.1908 | 3.08x |
| filters.threshold_sauvola | 512×512 | uint8 | window-(3, 31) | 1 | 12.7645 | 3.6498 | 3.50x |
| filters.threshold_niblack | 512×512 | uint8 | reversed | 1 | 10.6806 | 2.6283 | 4.06x |
| filters.threshold_sauvola | 512×512 | uint8 | reversed | 1 | 16.0841 | 4.3915 | 3.66x |
| filters.threshold_niblack | 512×512 | uint8 | strided | 1 | 14.6264 | 3.6963 | 3.96x |
| filters.threshold_sauvola | 512×512 | uint8 | strided | 1 | 17.1696 | 4.7671 | 3.60x |
| filters.threshold_niblack | 512×512 | uint8 | fortran | 1 | 19.3080 | 3.9676 | 4.87x |
| filters.threshold_sauvola | 512×512 | uint8 | fortran | 1 | 22.2330 | 4.2151 | 5.27x |
| filters.threshold_niblack | 1×512×512 | uint8 | volume | 1 | 32.0878 | 8.1163 | 3.95x |
| filters.threshold_sauvola | 1×512×512 | uint8 | volume | 1 | 37.1373 | 11.3292 | 3.28x |
| filters.threshold_niblack | 512×512 | uint16 | window-15 | 1 | 21.4944 | 5.5636 | 3.86x |
| filters.threshold_sauvola | 512×512 | uint16 | window-15 | 1 | 11.3453 | 4.1598 | 2.73x |
| filters.threshold_niblack | 512×512 | uint16 | window-3 | 1 | 13.9632 | 3.4947 | 4.00x |
| filters.threshold_sauvola | 512×512 | uint16 | window-3 | 1 | 11.5070 | 3.5427 | 3.25x |
| filters.threshold_niblack | 512×512 | uint16 | window-31 | 1 | 11.2169 | 3.4893 | 3.21x |
| filters.threshold_sauvola | 512×512 | uint16 | window-31 | 1 | 15.4511 | 3.7968 | 4.07x |
| filters.threshold_niblack | 512×512 | uint16 | window-127 | 1 | 16.4849 | 6.1917 | 2.66x |
| filters.threshold_sauvola | 512×512 | uint16 | window-127 | 1 | 17.0116 | 5.0499 | 3.37x |
| filters.threshold_niblack | 512×512 | uint16 | window-(3, 31) | 1 | 13.7379 | 3.0065 | 4.57x |
| filters.threshold_sauvola | 512×512 | uint16 | window-(3, 31) | 1 | 15.7230 | 4.7474 | 3.31x |
| filters.threshold_niblack | 512×512 | uint16 | reversed | 1 | 16.2271 | 3.2496 | 4.99x |
| filters.threshold_sauvola | 512×512 | uint16 | reversed | 1 | 22.8691 | 6.6433 | 3.44x |
| filters.threshold_niblack | 512×512 | uint16 | strided | 1 | 22.3223 | 7.1974 | 3.10x |
| filters.threshold_sauvola | 512×512 | uint16 | strided | 1 | 29.7236 | 6.3819 | 4.66x |
| filters.threshold_niblack | 512×512 | uint16 | fortran | 1 | 32.6910 | 6.4097 | 5.10x |
| filters.threshold_sauvola | 512×512 | uint16 | fortran | 1 | 24.1658 | 5.1083 | 4.73x |
| filters.threshold_niblack | 1×512×512 | uint16 | volume | 1 | 27.6475 | 6.6144 | 4.18x |
| filters.threshold_sauvola | 1×512×512 | uint16 | volume | 1 | 20.5175 | 5.8721 | 3.49x |
| filters.threshold_niblack | 512×512 | int16 | window-15 | 1 | 9.9337 | 3.1992 | 3.11x |
| filters.threshold_sauvola | 512×512 | int16 | window-15 | 1 | 9.0163 | 2.0988 | 4.30x |
| filters.threshold_niblack | 512×512 | int16 | window-3 | 1 | 8.9823 | 1.8978 | 4.73x |
| filters.threshold_sauvola | 512×512 | int16 | window-3 | 1 | 9.1320 | 2.1308 | 4.29x |
| filters.threshold_niblack | 512×512 | int16 | window-31 | 1 | 14.4901 | 3.9684 | 3.65x |
| filters.threshold_sauvola | 512×512 | int16 | window-31 | 1 | 9.4785 | 2.4327 | 3.90x |
| filters.threshold_niblack | 512×512 | int16 | window-127 | 1 | 13.7841 | 5.2356 | 2.63x |
| filters.threshold_sauvola | 512×512 | int16 | window-127 | 1 | 12.1067 | 4.4846 | 2.70x |
| filters.threshold_niblack | 512×512 | int16 | window-(3, 31) | 1 | 10.3227 | 3.1753 | 3.25x |
| filters.threshold_sauvola | 512×512 | int16 | window-(3, 31) | 1 | 10.2279 | 3.0912 | 3.31x |
| filters.threshold_niblack | 512×512 | int16 | reversed | 1 | 9.0694 | 3.2181 | 2.82x |
| filters.threshold_sauvola | 512×512 | int16 | reversed | 1 | 12.0809 | 3.3407 | 3.62x |
| filters.threshold_niblack | 512×512 | int16 | strided | 1 | 9.2497 | 3.2901 | 2.81x |
| filters.threshold_sauvola | 512×512 | int16 | strided | 1 | 12.9213 | 3.7554 | 3.44x |
| filters.threshold_niblack | 512×512 | int16 | fortran | 1 | 15.8369 | 3.7343 | 4.24x |
| filters.threshold_sauvola | 512×512 | int16 | fortran | 1 | 17.7144 | 4.3436 | 4.08x |
| filters.threshold_niblack | 1×512×512 | int16 | volume | 1 | 23.9855 | 7.8610 | 3.05x |
| filters.threshold_sauvola | 1×512×512 | int16 | volume | 1 | 22.1318 | 5.6665 | 3.91x |
| filters.threshold_niblack | 512×512 | float32 | window-15 | 1 | 8.2234 | 2.6348 | 3.12x |
| filters.threshold_sauvola | 512×512 | float32 | window-15 | 1 | 9.0253 | 2.7990 | 3.22x |
| filters.threshold_niblack | 512×512 | float32 | window-3 | 1 | 7.7282 | 2.4636 | 3.14x |
| filters.threshold_sauvola | 512×512 | float32 | window-3 | 1 | 7.3055 | 1.8685 | 3.91x |
| filters.threshold_niblack | 512×512 | float32 | window-31 | 1 | 7.6895 | 3.2948 | 2.33x |
| filters.threshold_sauvola | 512×512 | float32 | window-31 | 1 | 8.0100 | 1.9890 | 4.03x |
| filters.threshold_niblack | 512×512 | float32 | window-127 | 1 | 12.4576 | 5.2082 | 2.39x |
| filters.threshold_sauvola | 512×512 | float32 | window-127 | 1 | 10.8580 | 4.4835 | 2.42x |
| filters.threshold_niblack | 512×512 | float32 | window-(3, 31) | 1 | 7.0565 | 2.0499 | 3.44x |
| filters.threshold_sauvola | 512×512 | float32 | window-(3, 31) | 1 | 7.5053 | 1.8821 | 3.99x |
| filters.threshold_niblack | 512×512 | float32 | reversed | 1 | 9.0416 | 2.8763 | 3.14x |
| filters.threshold_sauvola | 512×512 | float32 | reversed | 1 | 8.9258 | 2.9514 | 3.02x |
| filters.threshold_niblack | 512×512 | float32 | strided | 1 | 6.5697 | 2.3053 | 2.85x |
| filters.threshold_sauvola | 512×512 | float32 | strided | 1 | 7.7135 | 4.8733 | 1.58x |
| filters.threshold_niblack | 512×512 | float32 | fortran | 1 | 12.2068 | 3.1056 | 3.93x |
| filters.threshold_sauvola | 512×512 | float32 | fortran | 1 | 8.9043 | 2.9008 | 3.07x |
| filters.threshold_niblack | 1×512×512 | float32 | volume | 1 | 19.8347 | 6.0099 | 3.30x |
| filters.threshold_sauvola | 1×512×512 | float32 | volume | 1 | 17.8513 | 5.4388 | 3.28x |
| filters.threshold_niblack | 512×512 | float32 | int64-window | 1 | 8.5473 | 2.6636 | 3.21x |
| filters.threshold_sauvola | 512×512 | float32 | int64-window | 1 | 8.1725 | 2.4073 | 3.39x |
| filters.threshold_niblack | 512×512 | float32 | k-0.0 | 1 | 6.0247 | 2.1487 | 2.80x |
| filters.threshold_sauvola | 512×512 | float32 | k-0.0 | 1 | 6.4252 | 2.1634 | 2.97x |
| filters.threshold_niblack | 512×512 | float32 | k--0.2 | 1 | 7.3309 | 3.7747 | 1.94x |
| filters.threshold_sauvola | 512×512 | float32 | k--0.2 | 1 | 9.4184 | 4.5819 | 2.06x |
| filters.threshold_niblack | 512×512 | float32 | k-np.float32(0.17) | 1 | 7.3158 | 2.5194 | 2.90x |
| filters.threshold_sauvola | 512×512 | float32 | k-np.float32(0.17) | 1 | 7.2310 | 2.2399 | 3.23x |
| filters.threshold_niblack | 512×512 | float32 | k-np.float64(0.17) | 1 | 7.0922 | 2.0684 | 3.43x |
| filters.threshold_sauvola | 512×512 | float32 | k-np.float64(0.17) | 1 | 11.1525 | 3.4838 | 3.20x |
| filters.threshold_sauvola | 512×512 | float32 | r-0.5 | 1 | 11.0427 | 2.9905 | 3.69x |
| filters.threshold_sauvola | 512×512 | float32 | r-128.0 | 1 | 9.1347 | 2.5014 | 3.65x |
| filters.threshold_sauvola | 512×512 | float32 | r-np.float64(0.5) | 1 | 11.8870 | 3.6350 | 3.27x |
| filters.threshold_niblack | 512×512 | float64 | window-15 | 1 | 10.3716 | 2.8605 | 3.63x |
| filters.threshold_sauvola | 512×512 | float64 | window-15 | 1 | 9.0963 | 2.8606 | 3.18x |
| filters.threshold_niblack | 512×512 | float64 | window-3 | 1 | 11.1284 | 2.4470 | 4.55x |
| filters.threshold_sauvola | 512×512 | float64 | window-3 | 1 | 9.6603 | 3.9252 | 2.46x |
| filters.threshold_niblack | 512×512 | float64 | window-31 | 1 | 13.1279 | 3.1522 | 4.16x |
| filters.threshold_sauvola | 512×512 | float64 | window-31 | 1 | 13.8565 | 5.6677 | 2.44x |
| filters.threshold_niblack | 512×512 | float64 | window-127 | 1 | 15.2360 | 5.2508 | 2.90x |
| filters.threshold_sauvola | 512×512 | float64 | window-127 | 1 | 16.3617 | 6.4586 | 2.53x |
| filters.threshold_niblack | 512×512 | float64 | window-(3, 31) | 1 | 10.3024 | 2.5537 | 4.03x |
| filters.threshold_sauvola | 512×512 | float64 | window-(3, 31) | 1 | 11.1287 | 2.6061 | 4.27x |
| filters.threshold_niblack | 512×512 | float64 | reversed | 1 | 9.2686 | 3.3696 | 2.75x |
| filters.threshold_sauvola | 512×512 | float64 | reversed | 1 | 10.9961 | 4.7769 | 2.30x |
| filters.threshold_niblack | 512×512 | float64 | strided | 1 | 11.0934 | 4.2580 | 2.61x |
| filters.threshold_sauvola | 512×512 | float64 | strided | 1 | 9.4072 | 4.2424 | 2.22x |
| filters.threshold_niblack | 512×512 | float64 | fortran | 1 | 15.8467 | 5.5453 | 2.86x |
| filters.threshold_sauvola | 512×512 | float64 | fortran | 1 | 14.2520 | 5.1803 | 2.75x |
| filters.threshold_niblack | 1×512×512 | float64 | volume | 1 | 18.2364 | 5.0639 | 3.60x |
| filters.threshold_sauvola | 1×512×512 | float64 | volume | 1 | 17.0902 | 5.0515 | 3.38x |
| filters.threshold_niblack | 512×512 | float64 | int64-window | 1 | 9.4156 | 3.3834 | 2.78x |
| filters.threshold_sauvola | 512×512 | float64 | int64-window | 1 | 8.4394 | 2.2887 | 3.69x |
| filters.threshold_niblack | 512×512 | float64 | k-0.0 | 1 | 9.9571 | 3.9764 | 2.50x |
| filters.threshold_sauvola | 512×512 | float64 | k-0.0 | 1 | 8.1978 | 2.2861 | 3.59x |
| filters.threshold_niblack | 512×512 | float64 | k--0.2 | 1 | 8.2377 | 3.2593 | 2.53x |
| filters.threshold_sauvola | 512×512 | float64 | k--0.2 | 1 | 8.4956 | 3.0594 | 2.78x |
| filters.threshold_niblack | 512×512 | float64 | k-np.float32(0.17) | 1 | 8.1156 | 3.0953 | 2.62x |
| filters.threshold_sauvola | 512×512 | float64 | k-np.float32(0.17) | 1 | 8.2552 | 3.0326 | 2.72x |
| filters.threshold_niblack | 512×512 | float64 | k-np.float64(0.17) | 1 | 10.1892 | 3.3753 | 3.02x |
| filters.threshold_sauvola | 512×512 | float64 | k-np.float64(0.17) | 1 | 10.4354 | 3.7663 | 2.77x |
| filters.threshold_sauvola | 512×512 | float64 | r-0.5 | 1 | 8.0548 | 2.1864 | 3.68x |
| filters.threshold_sauvola | 512×512 | float64 | r-128.0 | 1 | 8.9313 | 3.2497 | 2.75x |
| filters.threshold_sauvola | 512×512 | float64 | r-np.float64(0.5) | 1 | 7.5593 | 2.1088 | 3.58x |
| filters.threshold_niblack | 512×512 | bool | additional-width | 1 | 8.0327 | 2.9426 | 2.73x |
| filters.threshold_sauvola | 512×512 | bool | additional-width | 1 | 10.1709 | 3.1508 | 3.23x |
| filters.threshold_niblack | 512×512 | int8 | additional-width | 1 | 8.2763 | 3.1244 | 2.65x |
| filters.threshold_sauvola | 512×512 | int8 | additional-width | 1 | 9.9198 | 2.9447 | 3.37x |
| filters.threshold_niblack | 512×512 | uint32 | additional-width | 1 | 8.9737 | 3.0566 | 2.94x |
| filters.threshold_sauvola | 512×512 | uint32 | additional-width | 1 | 8.8946 | 2.4017 | 3.70x |
| filters.threshold_niblack | 512×512 | int32 | additional-width | 1 | 7.5733 | 2.7484 | 2.76x |
| filters.threshold_sauvola | 512×512 | int32 | additional-width | 1 | 8.6048 | 2.0787 | 4.14x |
| filters.threshold_niblack | 512×512 | uint64 | additional-width | 1 | 7.7892 | 4.1570 | 1.87x |
| filters.threshold_sauvola | 512×512 | uint64 | additional-width | 1 | 9.5788 | 3.5337 | 2.71x |
| filters.threshold_niblack | 512×512 | int64 | additional-width | 1 | 7.6158 | 2.7979 | 2.72x |
| filters.threshold_sauvola | 512×512 | int64 | additional-width | 1 | 8.7393 | 2.0944 | 4.17x |
