# Local threshold performance

Reference: original scikit-image 0.26.0. Reference thread pools use one worker; native workers are explicit. Each case checks dtype, shape, warnings and numerical parity with fallback disabled. Timings include dispatch, layout copying, reflected padding, both integral images, local statistics and output allocation. Both scalar and NumPy int64 window metadata are covered, as are strong float64 weights and ranges. All raw samples and slower cases are retained in the JSON.

| API | Shape | Input | Workload | Workers | Original ms | C++ ms | Speedup |
|---|---|---|---|---:|---:|---:|---:|
| filters.threshold_niblack | 512×512 | uint8 | window-15 | 1 | 10.0516 | 2.3662 | 4.25x |
| filters.threshold_sauvola | 512×512 | uint8 | window-15 | 1 | 12.4491 | 3.7614 | 3.31x |
| filters.threshold_niblack | 512×512 | uint8 | window-3 | 1 | 14.7514 | 2.7491 | 5.37x |
| filters.threshold_sauvola | 512×512 | uint8 | window-3 | 1 | 12.2660 | 3.8224 | 3.21x |
| filters.threshold_niblack | 512×512 | uint8 | window-31 | 1 | 10.2077 | 2.4537 | 4.16x |
| filters.threshold_sauvola | 512×512 | uint8 | window-31 | 1 | 14.6724 | 4.2579 | 3.45x |
| filters.threshold_niblack | 512×512 | uint8 | window-127 | 1 | 14.8467 | 3.9848 | 3.73x |
| filters.threshold_sauvola | 512×512 | uint8 | window-127 | 1 | 15.3270 | 5.6580 | 2.71x |
| filters.threshold_niblack | 512×512 | uint8 | window-(3, 31) | 1 | 14.5332 | 4.0180 | 3.62x |
| filters.threshold_sauvola | 512×512 | uint8 | window-(3, 31) | 1 | 9.7844 | 4.2618 | 2.30x |
| filters.threshold_niblack | 512×512 | uint8 | reversed | 1 | 12.2043 | 3.0138 | 4.05x |
| filters.threshold_sauvola | 512×512 | uint8 | reversed | 1 | 14.1502 | 5.7827 | 2.45x |
| filters.threshold_niblack | 512×512 | uint8 | strided | 1 | 9.2011 | 2.4238 | 3.80x |
| filters.threshold_sauvola | 512×512 | uint8 | strided | 1 | 11.1491 | 3.3716 | 3.31x |
| filters.threshold_niblack | 512×512 | uint8 | fortran | 1 | 13.3254 | 2.3319 | 5.71x |
| filters.threshold_sauvola | 512×512 | uint8 | fortran | 1 | 16.0025 | 2.6351 | 6.07x |
| filters.threshold_niblack | 1×512×512 | uint8 | volume | 1 | 25.5955 | 6.2710 | 4.08x |
| filters.threshold_sauvola | 1×512×512 | uint8 | volume | 1 | 24.1701 | 5.9798 | 4.04x |
| filters.threshold_niblack | 512×512 | uint16 | window-15 | 1 | 14.6290 | 3.5186 | 4.16x |
| filters.threshold_sauvola | 512×512 | uint16 | window-15 | 1 | 17.0167 | 5.4989 | 3.09x |
| filters.threshold_niblack | 512×512 | uint16 | window-3 | 1 | 8.7055 | 2.2104 | 3.94x |
| filters.threshold_sauvola | 512×512 | uint16 | window-3 | 1 | 14.3925 | 3.8538 | 3.73x |
| filters.threshold_niblack | 512×512 | uint16 | window-31 | 1 | 12.9556 | 2.6017 | 4.98x |
| filters.threshold_sauvola | 512×512 | uint16 | window-31 | 1 | 19.9630 | 4.9045 | 4.07x |
| filters.threshold_niblack | 512×512 | uint16 | window-127 | 1 | 13.4523 | 4.2069 | 3.20x |
| filters.threshold_sauvola | 512×512 | uint16 | window-127 | 1 | 13.0569 | 5.0479 | 2.59x |
| filters.threshold_niblack | 512×512 | uint16 | window-(3, 31) | 1 | 11.0946 | 3.1111 | 3.57x |
| filters.threshold_sauvola | 512×512 | uint16 | window-(3, 31) | 1 | 9.4102 | 2.7567 | 3.41x |
| filters.threshold_niblack | 512×512 | uint16 | reversed | 1 | 10.0866 | 3.5366 | 2.85x |
| filters.threshold_sauvola | 512×512 | uint16 | reversed | 1 | 12.3001 | 2.6488 | 4.64x |
| filters.threshold_niblack | 512×512 | uint16 | strided | 1 | 9.3126 | 4.7081 | 1.98x |
| filters.threshold_sauvola | 512×512 | uint16 | strided | 1 | 10.0001 | 3.1100 | 3.22x |
| filters.threshold_niblack | 512×512 | uint16 | fortran | 1 | 15.9191 | 3.5138 | 4.53x |
| filters.threshold_sauvola | 512×512 | uint16 | fortran | 1 | 19.4084 | 3.8747 | 5.01x |
| filters.threshold_niblack | 1×512×512 | uint16 | volume | 1 | 25.4564 | 7.8200 | 3.26x |
| filters.threshold_sauvola | 1×512×512 | uint16 | volume | 1 | 22.5555 | 7.4647 | 3.02x |
| filters.threshold_niblack | 512×512 | int16 | window-15 | 1 | 11.3834 | 4.1732 | 2.73x |
| filters.threshold_sauvola | 512×512 | int16 | window-15 | 1 | 14.5225 | 3.0368 | 4.78x |
| filters.threshold_niblack | 512×512 | int16 | window-3 | 1 | 11.4440 | 2.3324 | 4.91x |
| filters.threshold_sauvola | 512×512 | int16 | window-3 | 1 | 10.4986 | 2.5637 | 4.10x |
| filters.threshold_niblack | 512×512 | int16 | window-31 | 1 | 15.8987 | 5.0412 | 3.15x |
| filters.threshold_sauvola | 512×512 | int16 | window-31 | 1 | 12.9278 | 2.9246 | 4.42x |
| filters.threshold_niblack | 512×512 | int16 | window-127 | 1 | 14.8545 | 4.4725 | 3.32x |
| filters.threshold_sauvola | 512×512 | int16 | window-127 | 1 | 15.6675 | 4.6380 | 3.38x |
| filters.threshold_niblack | 512×512 | int16 | window-(3, 31) | 1 | 13.8744 | 4.0710 | 3.41x |
| filters.threshold_sauvola | 512×512 | int16 | window-(3, 31) | 1 | 11.1066 | 3.9731 | 2.80x |
| filters.threshold_niblack | 512×512 | int16 | reversed | 1 | 15.7898 | 4.0885 | 3.86x |
| filters.threshold_sauvola | 512×512 | int16 | reversed | 1 | 16.9754 | 4.5769 | 3.71x |
| filters.threshold_niblack | 512×512 | int16 | strided | 1 | 11.5595 | 4.3389 | 2.66x |
| filters.threshold_sauvola | 512×512 | int16 | strided | 1 | 14.1568 | 4.7459 | 2.98x |
| filters.threshold_niblack | 512×512 | int16 | fortran | 1 | 16.7912 | 2.2636 | 7.42x |
| filters.threshold_sauvola | 512×512 | int16 | fortran | 1 | 17.8707 | 3.7331 | 4.79x |
| filters.threshold_niblack | 1×512×512 | int16 | volume | 1 | 29.0930 | 8.3444 | 3.49x |
| filters.threshold_sauvola | 1×512×512 | int16 | volume | 1 | 25.1075 | 9.0203 | 2.78x |
| filters.threshold_niblack | 512×512 | float32 | window-15 | 1 | 9.4038 | 3.1388 | 3.00x |
| filters.threshold_sauvola | 512×512 | float32 | window-15 | 1 | 10.6538 | 2.2277 | 4.78x |
| filters.threshold_niblack | 512×512 | float32 | window-3 | 1 | 8.0640 | 2.8039 | 2.88x |
| filters.threshold_sauvola | 512×512 | float32 | window-3 | 1 | 11.0330 | 2.4539 | 4.50x |
| filters.threshold_niblack | 512×512 | float32 | window-31 | 1 | 10.8814 | 3.8418 | 2.83x |
| filters.threshold_sauvola | 512×512 | float32 | window-31 | 1 | 11.5471 | 4.3433 | 2.66x |
| filters.threshold_niblack | 512×512 | float32 | window-127 | 1 | 13.5520 | 6.1688 | 2.20x |
| filters.threshold_sauvola | 512×512 | float32 | window-127 | 1 | 13.2698 | 4.8447 | 2.74x |
| filters.threshold_niblack | 512×512 | float32 | window-(3, 31) | 1 | 10.0664 | 2.6059 | 3.86x |
| filters.threshold_sauvola | 512×512 | float32 | window-(3, 31) | 1 | 12.7740 | 6.7420 | 1.89x |
| filters.threshold_niblack | 512×512 | float32 | reversed | 1 | 28.6866 | 6.3189 | 4.54x |
| filters.threshold_sauvola | 512×512 | float32 | reversed | 1 | 19.0060 | 7.4151 | 2.56x |
| filters.threshold_niblack | 512×512 | float32 | strided | 1 | 12.2004 | 4.8707 | 2.50x |
| filters.threshold_sauvola | 512×512 | float32 | strided | 1 | 11.2483 | 4.7278 | 2.38x |
| filters.threshold_niblack | 512×512 | float32 | fortran | 1 | 17.9993 | 3.8699 | 4.65x |
| filters.threshold_sauvola | 512×512 | float32 | fortran | 1 | 20.9475 | 4.5683 | 4.59x |
| filters.threshold_niblack | 1×512×512 | float32 | volume | 1 | 24.2384 | 7.7753 | 3.12x |
| filters.threshold_sauvola | 1×512×512 | float32 | volume | 1 | 20.4050 | 6.5810 | 3.10x |
| filters.threshold_niblack | 512×512 | float32 | int64-window | 1 | 7.8756 | 3.2049 | 2.46x |
| filters.threshold_sauvola | 512×512 | float32 | int64-window | 1 | 10.3114 | 3.7859 | 2.72x |
| filters.threshold_niblack | 512×512 | float32 | k-0.0 | 1 | 9.5436 | 3.2395 | 2.95x |
| filters.threshold_sauvola | 512×512 | float32 | k-0.0 | 1 | 9.8364 | 3.0132 | 3.26x |
| filters.threshold_niblack | 512×512 | float32 | k--0.2 | 1 | 11.7640 | 3.6205 | 3.25x |
| filters.threshold_sauvola | 512×512 | float32 | k--0.2 | 1 | 14.9974 | 4.1504 | 3.61x |
| filters.threshold_niblack | 512×512 | float32 | k-np.float32(0.17) | 1 | 8.8677 | 2.9817 | 2.97x |
| filters.threshold_sauvola | 512×512 | float32 | k-np.float32(0.17) | 1 | 9.5977 | 3.5476 | 2.71x |
| filters.threshold_niblack | 512×512 | float32 | k-np.float64(0.17) | 1 | 9.4957 | 2.7770 | 3.42x |
| filters.threshold_sauvola | 512×512 | float32 | k-np.float64(0.17) | 1 | 11.9148 | 3.1202 | 3.82x |
| filters.threshold_sauvola | 512×512 | float32 | r-0.5 | 1 | 9.0777 | 2.9016 | 3.13x |
| filters.threshold_sauvola | 512×512 | float32 | r-128.0 | 1 | 9.2823 | 2.4485 | 3.79x |
| filters.threshold_sauvola | 512×512 | float32 | r-np.float64(0.5) | 1 | 12.1796 | 3.0970 | 3.93x |
| filters.threshold_niblack | 512×512 | float64 | window-15 | 1 | 13.2329 | 2.8610 | 4.63x |
| filters.threshold_sauvola | 512×512 | float64 | window-15 | 1 | 10.6733 | 2.6580 | 4.02x |
| filters.threshold_niblack | 512×512 | float64 | window-3 | 1 | 11.2065 | 2.9134 | 3.85x |
| filters.threshold_sauvola | 512×512 | float64 | window-3 | 1 | 10.3304 | 2.9812 | 3.47x |
| filters.threshold_niblack | 512×512 | float64 | window-31 | 1 | 11.2706 | 3.3108 | 3.40x |
| filters.threshold_sauvola | 512×512 | float64 | window-31 | 1 | 13.1598 | 3.8300 | 3.44x |
| filters.threshold_niblack | 512×512 | float64 | window-127 | 1 | 16.0820 | 3.7867 | 4.25x |
| filters.threshold_sauvola | 512×512 | float64 | window-127 | 1 | 15.0325 | 5.1904 | 2.90x |
| filters.threshold_niblack | 512×512 | float64 | window-(3, 31) | 1 | 13.3445 | 3.0989 | 4.31x |
| filters.threshold_sauvola | 512×512 | float64 | window-(3, 31) | 1 | 13.2732 | 3.0590 | 4.34x |
| filters.threshold_niblack | 512×512 | float64 | reversed | 1 | 16.2652 | 6.2304 | 2.61x |
| filters.threshold_sauvola | 512×512 | float64 | reversed | 1 | 16.3427 | 5.8277 | 2.80x |
| filters.threshold_niblack | 512×512 | float64 | strided | 1 | 16.2558 | 4.6916 | 3.46x |
| filters.threshold_sauvola | 512×512 | float64 | strided | 1 | 14.7091 | 5.7793 | 2.55x |
| filters.threshold_niblack | 512×512 | float64 | fortran | 1 | 22.1651 | 6.8577 | 3.23x |
| filters.threshold_sauvola | 512×512 | float64 | fortran | 1 | 23.2216 | 7.9143 | 2.93x |
| filters.threshold_niblack | 1×512×512 | float64 | volume | 1 | 21.9020 | 8.2116 | 2.67x |
| filters.threshold_sauvola | 1×512×512 | float64 | volume | 1 | 23.1821 | 7.8297 | 2.96x |
| filters.threshold_niblack | 512×512 | float64 | int64-window | 1 | 13.3897 | 4.3098 | 3.11x |
| filters.threshold_sauvola | 512×512 | float64 | int64-window | 1 | 10.6561 | 3.6174 | 2.95x |
| filters.threshold_niblack | 512×512 | float64 | k-0.0 | 1 | 9.4343 | 3.4880 | 2.70x |
| filters.threshold_sauvola | 512×512 | float64 | k-0.0 | 1 | 10.1320 | 3.8480 | 2.63x |
| filters.threshold_niblack | 512×512 | float64 | k--0.2 | 1 | 9.0887 | 3.8002 | 2.39x |
| filters.threshold_sauvola | 512×512 | float64 | k--0.2 | 1 | 11.8942 | 3.8283 | 3.11x |
| filters.threshold_niblack | 512×512 | float64 | k-np.float32(0.17) | 1 | 9.9691 | 3.5748 | 2.79x |
| filters.threshold_sauvola | 512×512 | float64 | k-np.float32(0.17) | 1 | 9.3835 | 2.3986 | 3.91x |
| filters.threshold_niblack | 512×512 | float64 | k-np.float64(0.17) | 1 | 13.3613 | 4.0270 | 3.32x |
| filters.threshold_sauvola | 512×512 | float64 | k-np.float64(0.17) | 1 | 15.8487 | 4.2529 | 3.73x |
| filters.threshold_sauvola | 512×512 | float64 | r-0.5 | 1 | 14.7758 | 4.6554 | 3.17x |
| filters.threshold_sauvola | 512×512 | float64 | r-128.0 | 1 | 16.0934 | 4.5563 | 3.53x |
| filters.threshold_sauvola | 512×512 | float64 | r-np.float64(0.5) | 1 | 19.0349 | 5.5944 | 3.40x |
| filters.threshold_niblack | 512×512 | bool | additional-width | 1 | 13.0564 | 2.7902 | 4.68x |
| filters.threshold_sauvola | 512×512 | bool | additional-width | 1 | 11.6113 | 3.1717 | 3.66x |
| filters.threshold_niblack | 512×512 | int8 | additional-width | 1 | 13.2553 | 2.6456 | 5.01x |
| filters.threshold_sauvola | 512×512 | int8 | additional-width | 1 | 18.8292 | 5.6822 | 3.31x |
| filters.threshold_niblack | 512×512 | uint32 | additional-width | 1 | 13.2903 | 3.3224 | 4.00x |
| filters.threshold_sauvola | 512×512 | uint32 | additional-width | 1 | 15.4557 | 2.8947 | 5.34x |
| filters.threshold_niblack | 512×512 | int32 | additional-width | 1 | 11.8631 | 3.6892 | 3.22x |
| filters.threshold_sauvola | 512×512 | int32 | additional-width | 1 | 10.4909 | 2.5808 | 4.06x |
| filters.threshold_niblack | 512×512 | uint64 | additional-width | 1 | 9.7283 | 4.6286 | 2.10x |
| filters.threshold_sauvola | 512×512 | uint64 | additional-width | 1 | 13.2283 | 4.4794 | 2.95x |
| filters.threshold_niblack | 512×512 | int64 | additional-width | 1 | 12.7833 | 3.1223 | 4.09x |
| filters.threshold_sauvola | 512×512 | int64 | additional-width | 1 | 13.9654 | 3.2143 | 4.34x |
