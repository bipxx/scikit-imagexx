# Local threshold performance

Reference: original scikit-image 0.26.0. Reference thread pools use one worker; native workers are explicit. Each case checks dtype, shape, warnings and numerical parity with fallback disabled. Timings include dispatch, layout copying, reflected padding, both integral images, local statistics and output allocation. Both scalar and NumPy int64 window metadata are covered, as are strong float64 weights and ranges. All raw samples and slower cases are retained in the JSON.

| API | Shape | Input | Workload | Workers | Original ms | C++ ms | Speedup |
|---|---|---|---|---:|---:|---:|---:|
| filters.threshold_niblack | 512×512 | uint8 | window-15 | 1 | 8.8320 | 2.1961 | 4.02x |
| filters.threshold_sauvola | 512×512 | uint8 | window-15 | 1 | 8.5067 | 1.9559 | 4.35x |
| filters.threshold_niblack | 512×512 | uint8 | window-3 | 1 | 9.1722 | 2.2180 | 4.14x |
| filters.threshold_sauvola | 512×512 | uint8 | window-3 | 1 | 9.1589 | 3.2530 | 2.82x |
| filters.threshold_niblack | 512×512 | uint8 | window-31 | 1 | 8.8345 | 2.4863 | 3.55x |
| filters.threshold_sauvola | 512×512 | uint8 | window-31 | 1 | 9.9170 | 3.5737 | 2.77x |
| filters.threshold_niblack | 512×512 | uint8 | window-127 | 1 | 11.7018 | 3.2864 | 3.56x |
| filters.threshold_sauvola | 512×512 | uint8 | window-127 | 1 | 18.1636 | 5.6933 | 3.19x |
| filters.threshold_niblack | 512×512 | uint8 | window-(3, 31) | 1 | 8.9319 | 2.2725 | 3.93x |
| filters.threshold_sauvola | 512×512 | uint8 | window-(3, 31) | 1 | 9.7425 | 3.4249 | 2.84x |
| filters.threshold_niblack | 512×512 | uint8 | reversed | 1 | 10.7958 | 3.0095 | 3.59x |
| filters.threshold_sauvola | 512×512 | uint8 | reversed | 1 | 14.0676 | 3.9594 | 3.55x |
| filters.threshold_niblack | 512×512 | uint8 | strided | 1 | 9.3230 | 2.4342 | 3.83x |
| filters.threshold_sauvola | 512×512 | uint8 | strided | 1 | 11.2804 | 3.6135 | 3.12x |
| filters.threshold_niblack | 512×512 | uint8 | fortran | 1 | 12.6427 | 2.3387 | 5.41x |
| filters.threshold_sauvola | 512×512 | uint8 | fortran | 1 | 17.2875 | 3.6926 | 4.68x |
| filters.threshold_niblack | 1×512×512 | uint8 | volume | 1 | 19.5189 | 5.2165 | 3.74x |
| filters.threshold_sauvola | 1×512×512 | uint8 | volume | 1 | 18.7730 | 5.8046 | 3.23x |
| filters.threshold_niblack | 512×512 | uint16 | window-15 | 1 | 8.5419 | 2.2176 | 3.85x |
| filters.threshold_sauvola | 512×512 | uint16 | window-15 | 1 | 10.9444 | 2.3856 | 4.59x |
| filters.threshold_niblack | 512×512 | uint16 | window-3 | 1 | 8.4160 | 2.3245 | 3.62x |
| filters.threshold_sauvola | 512×512 | uint16 | window-3 | 1 | 10.0063 | 2.5924 | 3.86x |
| filters.threshold_niblack | 512×512 | uint16 | window-31 | 1 | 8.7567 | 2.7142 | 3.23x |
| filters.threshold_sauvola | 512×512 | uint16 | window-31 | 1 | 10.6495 | 2.5524 | 4.17x |
| filters.threshold_niblack | 512×512 | uint16 | window-127 | 1 | 11.8301 | 3.8847 | 3.05x |
| filters.threshold_sauvola | 512×512 | uint16 | window-127 | 1 | 12.8087 | 4.7202 | 2.71x |
| filters.threshold_niblack | 512×512 | uint16 | window-(3, 31) | 1 | 8.4248 | 2.2284 | 3.78x |
| filters.threshold_sauvola | 512×512 | uint16 | window-(3, 31) | 1 | 8.5130 | 2.2401 | 3.80x |
| filters.threshold_niblack | 512×512 | uint16 | reversed | 1 | 8.6472 | 3.1806 | 2.72x |
| filters.threshold_sauvola | 512×512 | uint16 | reversed | 1 | 8.0313 | 2.2168 | 3.62x |
| filters.threshold_niblack | 512×512 | uint16 | strided | 1 | 11.0704 | 4.0553 | 2.73x |
| filters.threshold_sauvola | 512×512 | uint16 | strided | 1 | 8.4861 | 2.3721 | 3.58x |
| filters.threshold_niblack | 512×512 | uint16 | fortran | 1 | 13.6124 | 3.3359 | 4.08x |
| filters.threshold_sauvola | 512×512 | uint16 | fortran | 1 | 13.8077 | 2.1206 | 6.51x |
| filters.threshold_niblack | 1×512×512 | uint16 | volume | 1 | 19.7921 | 5.0210 | 3.94x |
| filters.threshold_sauvola | 1×512×512 | uint16 | volume | 1 | 18.6318 | 4.9771 | 3.74x |
| filters.threshold_niblack | 512×512 | int16 | window-15 | 1 | 9.1530 | 3.4250 | 2.67x |
| filters.threshold_sauvola | 512×512 | int16 | window-15 | 1 | 12.3840 | 3.0511 | 4.06x |
| filters.threshold_niblack | 512×512 | int16 | window-3 | 1 | 9.6126 | 2.9723 | 3.23x |
| filters.threshold_sauvola | 512×512 | int16 | window-3 | 1 | 8.1252 | 2.1059 | 3.86x |
| filters.threshold_niblack | 512×512 | int16 | window-31 | 1 | 9.8162 | 2.1624 | 4.54x |
| filters.threshold_sauvola | 512×512 | int16 | window-31 | 1 | 10.1768 | 3.5242 | 2.89x |
| filters.threshold_niblack | 512×512 | int16 | window-127 | 1 | 11.9791 | 4.1994 | 2.85x |
| filters.threshold_sauvola | 512×512 | int16 | window-127 | 1 | 11.8321 | 4.5103 | 2.62x |
| filters.threshold_niblack | 512×512 | int16 | window-(3, 31) | 1 | 8.2710 | 1.8368 | 4.50x |
| filters.threshold_sauvola | 512×512 | int16 | window-(3, 31) | 1 | 8.9704 | 3.4707 | 2.58x |
| filters.threshold_niblack | 512×512 | int16 | reversed | 1 | 8.7271 | 2.0766 | 4.20x |
| filters.threshold_sauvola | 512×512 | int16 | reversed | 1 | 11.2498 | 3.4696 | 3.24x |
| filters.threshold_niblack | 512×512 | int16 | strided | 1 | 8.7462 | 3.3217 | 2.63x |
| filters.threshold_sauvola | 512×512 | int16 | strided | 1 | 10.6487 | 3.3858 | 3.15x |
| filters.threshold_niblack | 512×512 | int16 | fortran | 1 | 16.2874 | 3.6742 | 4.43x |
| filters.threshold_sauvola | 512×512 | int16 | fortran | 1 | 15.4695 | 3.2179 | 4.81x |
| filters.threshold_niblack | 1×512×512 | int16 | volume | 1 | 19.7967 | 5.9902 | 3.30x |
| filters.threshold_sauvola | 1×512×512 | int16 | volume | 1 | 22.7598 | 5.7532 | 3.96x |
| filters.threshold_niblack | 512×512 | float32 | window-15 | 1 | 8.6447 | 2.8159 | 3.07x |
| filters.threshold_sauvola | 512×512 | float32 | window-15 | 1 | 7.0442 | 1.8850 | 3.74x |
| filters.threshold_niblack | 512×512 | float32 | window-3 | 1 | 6.4352 | 2.0502 | 3.14x |
| filters.threshold_sauvola | 512×512 | float32 | window-3 | 1 | 6.8507 | 1.8326 | 3.74x |
| filters.threshold_niblack | 512×512 | float32 | window-31 | 1 | 7.4993 | 1.8732 | 4.00x |
| filters.threshold_sauvola | 512×512 | float32 | window-31 | 1 | 7.2133 | 2.0351 | 3.54x |
| filters.threshold_niblack | 512×512 | float32 | window-127 | 1 | 10.7812 | 4.3253 | 2.49x |
| filters.threshold_sauvola | 512×512 | float32 | window-127 | 1 | 10.7172 | 4.6382 | 2.31x |
| filters.threshold_niblack | 512×512 | float32 | window-(3, 31) | 1 | 6.8145 | 1.8013 | 3.78x |
| filters.threshold_sauvola | 512×512 | float32 | window-(3, 31) | 1 | 6.8167 | 2.5792 | 2.64x |
| filters.threshold_niblack | 512×512 | float32 | reversed | 1 | 6.7155 | 2.2934 | 2.93x |
| filters.threshold_sauvola | 512×512 | float32 | reversed | 1 | 6.6822 | 2.2899 | 2.92x |
| filters.threshold_niblack | 512×512 | float32 | strided | 1 | 7.1077 | 3.0780 | 2.31x |
| filters.threshold_sauvola | 512×512 | float32 | strided | 1 | 7.0627 | 2.5253 | 2.80x |
| filters.threshold_niblack | 512×512 | float32 | fortran | 1 | 10.9715 | 2.6049 | 4.21x |
| filters.threshold_sauvola | 512×512 | float32 | fortran | 1 | 15.7978 | 3.2796 | 4.82x |
| filters.threshold_niblack | 1×512×512 | float32 | volume | 1 | 15.6906 | 5.0308 | 3.12x |
| filters.threshold_sauvola | 1×512×512 | float32 | volume | 1 | 15.6881 | 4.6651 | 3.36x |
| filters.threshold_niblack | 512×512 | float32 | int64-window | 1 | 7.6208 | 2.3689 | 3.22x |
| filters.threshold_sauvola | 512×512 | float32 | int64-window | 1 | 7.5480 | 2.3298 | 3.24x |
| filters.threshold_niblack | 512×512 | float32 | k-0.0 | 1 | 6.1696 | 2.3321 | 2.65x |
| filters.threshold_sauvola | 512×512 | float32 | k-0.0 | 1 | 7.4189 | 2.1824 | 3.40x |
| filters.threshold_niblack | 512×512 | float32 | k--0.2 | 1 | 6.7127 | 2.3494 | 2.86x |
| filters.threshold_sauvola | 512×512 | float32 | k--0.2 | 1 | 7.2687 | 2.2290 | 3.26x |
| filters.threshold_niblack | 512×512 | float32 | k-np.float32(0.17) | 1 | 6.4230 | 2.1887 | 2.93x |
| filters.threshold_sauvola | 512×512 | float32 | k-np.float32(0.17) | 1 | 7.5715 | 2.4682 | 3.07x |
| filters.threshold_niblack | 512×512 | float32 | k-np.float64(0.17) | 1 | 7.7138 | 2.3062 | 3.34x |
| filters.threshold_sauvola | 512×512 | float32 | k-np.float64(0.17) | 1 | 7.7722 | 1.9886 | 3.91x |
| filters.threshold_sauvola | 512×512 | float32 | r-0.5 | 1 | 7.7356 | 1.9145 | 4.04x |
| filters.threshold_sauvola | 512×512 | float32 | r-128.0 | 1 | 7.0055 | 2.3835 | 2.94x |
| filters.threshold_sauvola | 512×512 | float32 | r-np.float64(0.5) | 1 | 8.4557 | 2.2506 | 3.76x |
| filters.threshold_niblack | 512×512 | float64 | window-15 | 1 | 7.7505 | 2.0221 | 3.83x |
| filters.threshold_sauvola | 512×512 | float64 | window-15 | 1 | 9.2349 | 2.4724 | 3.74x |
| filters.threshold_niblack | 512×512 | float64 | window-3 | 1 | 7.5182 | 1.7341 | 4.34x |
| filters.threshold_sauvola | 512×512 | float64 | window-3 | 1 | 8.2907 | 2.2291 | 3.72x |
| filters.threshold_niblack | 512×512 | float64 | window-31 | 1 | 10.2120 | 2.3465 | 4.35x |
| filters.threshold_sauvola | 512×512 | float64 | window-31 | 1 | 11.1403 | 2.8095 | 3.97x |
| filters.threshold_niblack | 512×512 | float64 | window-127 | 1 | 12.1005 | 2.7910 | 4.34x |
| filters.threshold_sauvola | 512×512 | float64 | window-127 | 1 | 11.3532 | 4.8556 | 2.34x |
| filters.threshold_niblack | 512×512 | float64 | window-(3, 31) | 1 | 9.3077 | 2.1257 | 4.38x |
| filters.threshold_sauvola | 512×512 | float64 | window-(3, 31) | 1 | 9.1848 | 2.2085 | 4.16x |
| filters.threshold_niblack | 512×512 | float64 | reversed | 1 | 9.0418 | 3.3688 | 2.68x |
| filters.threshold_sauvola | 512×512 | float64 | reversed | 1 | 9.3955 | 3.9063 | 2.41x |
| filters.threshold_niblack | 512×512 | float64 | strided | 1 | 9.3047 | 3.4259 | 2.72x |
| filters.threshold_sauvola | 512×512 | float64 | strided | 1 | 9.9192 | 4.1495 | 2.39x |
| filters.threshold_niblack | 512×512 | float64 | fortran | 1 | 13.4818 | 4.6171 | 2.92x |
| filters.threshold_sauvola | 512×512 | float64 | fortran | 1 | 12.7452 | 5.3859 | 2.37x |
| filters.threshold_niblack | 1×512×512 | float64 | volume | 1 | 18.3242 | 5.1623 | 3.55x |
| filters.threshold_sauvola | 1×512×512 | float64 | volume | 1 | 17.5116 | 5.2581 | 3.33x |
| filters.threshold_niblack | 512×512 | float64 | int64-window | 1 | 8.7508 | 3.4286 | 2.55x |
| filters.threshold_sauvola | 512×512 | float64 | int64-window | 1 | 9.0126 | 3.3984 | 2.65x |
| filters.threshold_niblack | 512×512 | float64 | k-0.0 | 1 | 8.8210 | 3.6500 | 2.42x |
| filters.threshold_sauvola | 512×512 | float64 | k-0.0 | 1 | 9.1398 | 3.3623 | 2.72x |
| filters.threshold_niblack | 512×512 | float64 | k--0.2 | 1 | 9.1955 | 3.3137 | 2.77x |
| filters.threshold_sauvola | 512×512 | float64 | k--0.2 | 1 | 10.9407 | 4.0150 | 2.72x |
| filters.threshold_niblack | 512×512 | float64 | k-np.float32(0.17) | 1 | 12.1774 | 4.1879 | 2.91x |
| filters.threshold_sauvola | 512×512 | float64 | k-np.float32(0.17) | 1 | 13.1687 | 4.6368 | 2.84x |
| filters.threshold_niblack | 512×512 | float64 | k-np.float64(0.17) | 1 | 10.3996 | 3.9452 | 2.64x |
| filters.threshold_sauvola | 512×512 | float64 | k-np.float64(0.17) | 1 | 12.1405 | 4.3319 | 2.80x |
| filters.threshold_sauvola | 512×512 | float64 | r-0.5 | 1 | 16.3121 | 5.6804 | 2.87x |
| filters.threshold_sauvola | 512×512 | float64 | r-128.0 | 1 | 12.8595 | 4.0330 | 3.19x |
| filters.threshold_sauvola | 512×512 | float64 | r-np.float64(0.5) | 1 | 9.7652 | 3.7726 | 2.59x |
| filters.threshold_niblack | 512×512 | bool | additional-width | 1 | 9.8456 | 2.1529 | 4.57x |
| filters.threshold_sauvola | 512×512 | bool | additional-width | 1 | 9.0607 | 2.1983 | 4.12x |
| filters.threshold_niblack | 512×512 | int8 | additional-width | 1 | 9.0765 | 1.8345 | 4.95x |
| filters.threshold_sauvola | 512×512 | int8 | additional-width | 1 | 9.2974 | 2.1729 | 4.28x |
| filters.threshold_niblack | 512×512 | uint32 | additional-width | 1 | 9.1971 | 2.1202 | 4.34x |
| filters.threshold_sauvola | 512×512 | uint32 | additional-width | 1 | 10.0851 | 2.3261 | 4.34x |
| filters.threshold_niblack | 512×512 | int32 | additional-width | 1 | 13.1666 | 4.0999 | 3.21x |
| filters.threshold_sauvola | 512×512 | int32 | additional-width | 1 | 12.3803 | 2.6341 | 4.70x |
| filters.threshold_niblack | 512×512 | uint64 | additional-width | 1 | 10.0552 | 3.7212 | 2.70x |
| filters.threshold_sauvola | 512×512 | uint64 | additional-width | 1 | 11.2561 | 3.6879 | 3.05x |
| filters.threshold_niblack | 512×512 | int64 | additional-width | 1 | 9.5172 | 2.2361 | 4.26x |
| filters.threshold_sauvola | 512×512 | int64 | additional-width | 1 | 10.7262 | 2.2290 | 4.81x |
