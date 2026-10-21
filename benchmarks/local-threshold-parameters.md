# Local threshold performance

Reference: original scikit-image 0.26.0. Reference thread pools use one worker; native workers are explicit. Each case checks dtype, shape, warnings and numerical parity with fallback disabled. Timings include dispatch, layout copying, reflected padding, both integral images, local statistics and output allocation. Both scalar and NumPy int64 window metadata are covered, as are strong float64 weights and ranges. All raw samples and slower cases are retained in the JSON.

| API | Shape | Input | Workload | Workers | Original ms | C++ ms | Speedup |
|---|---|---|---|---:|---:|---:|---:|
| filters.threshold_niblack | 512×512 | uint8 | window-15 | 1 | 12.5033 | 2.5839 | 4.84x |
| filters.threshold_sauvola | 512×512 | uint8 | window-15 | 1 | 9.9938 | 3.7085 | 2.69x |
| filters.threshold_niblack | 512×512 | uint8 | window-3 | 1 | 8.6833 | 2.4638 | 3.52x |
| filters.threshold_sauvola | 512×512 | uint8 | window-3 | 1 | 9.9729 | 3.3885 | 2.94x |
| filters.threshold_niblack | 512×512 | uint8 | window-31 | 1 | 9.6677 | 2.5021 | 3.86x |
| filters.threshold_sauvola | 512×512 | uint8 | window-31 | 1 | 10.1485 | 3.6805 | 2.76x |
| filters.threshold_niblack | 512×512 | uint8 | window-127 | 1 | 13.8030 | 4.0646 | 3.40x |
| filters.threshold_sauvola | 512×512 | uint8 | window-127 | 1 | 18.9008 | 4.9434 | 3.82x |
| filters.threshold_niblack | 512×512 | uint8 | window-(3, 31) | 1 | 10.9884 | 3.0284 | 3.63x |
| filters.threshold_sauvola | 512×512 | uint8 | window-(3, 31) | 1 | 11.3536 | 4.1109 | 2.76x |
| filters.threshold_niblack | 512×512 | uint8 | reversed | 1 | 11.3696 | 2.3295 | 4.88x |
| filters.threshold_sauvola | 512×512 | uint8 | reversed | 1 | 14.1191 | 3.7725 | 3.74x |
| filters.threshold_niblack | 512×512 | uint8 | strided | 1 | 9.6217 | 2.4380 | 3.95x |
| filters.threshold_sauvola | 512×512 | uint8 | strided | 1 | 12.3005 | 3.8320 | 3.21x |
| filters.threshold_niblack | 512×512 | uint8 | fortran | 1 | 19.9833 | 3.1551 | 6.33x |
| filters.threshold_sauvola | 512×512 | uint8 | fortran | 1 | 17.2711 | 2.9881 | 5.78x |
| filters.threshold_niblack | 1×512×512 | uint8 | volume | 1 | 24.5870 | 7.1217 | 3.45x |
| filters.threshold_sauvola | 1×512×512 | uint8 | volume | 1 | 22.8216 | 7.0018 | 3.26x |
| filters.threshold_niblack | 512×512 | uint16 | window-15 | 1 | 11.4438 | 2.8724 | 3.98x |
| filters.threshold_sauvola | 512×512 | uint16 | window-15 | 1 | 11.6723 | 3.2736 | 3.57x |
| filters.threshold_niblack | 512×512 | uint16 | window-3 | 1 | 7.5036 | 2.0861 | 3.60x |
| filters.threshold_sauvola | 512×512 | uint16 | window-3 | 1 | 11.3970 | 3.7858 | 3.01x |
| filters.threshold_niblack | 512×512 | uint16 | window-31 | 1 | 9.8778 | 2.4396 | 4.05x |
| filters.threshold_sauvola | 512×512 | uint16 | window-31 | 1 | 18.1654 | 5.5286 | 3.29x |
| filters.threshold_niblack | 512×512 | uint16 | window-127 | 1 | 13.7154 | 6.0629 | 2.26x |
| filters.threshold_sauvola | 512×512 | uint16 | window-127 | 1 | 20.3788 | 7.3350 | 2.78x |
| filters.threshold_niblack | 512×512 | uint16 | window-(3, 31) | 1 | 13.2298 | 3.3677 | 3.93x |
| filters.threshold_sauvola | 512×512 | uint16 | window-(3, 31) | 1 | 15.5117 | 3.7788 | 4.10x |
| filters.threshold_niblack | 512×512 | uint16 | reversed | 1 | 11.9140 | 4.9531 | 2.41x |
| filters.threshold_sauvola | 512×512 | uint16 | reversed | 1 | 11.1703 | 3.2507 | 3.44x |
| filters.threshold_niblack | 512×512 | uint16 | strided | 1 | 12.6633 | 4.4460 | 2.85x |
| filters.threshold_sauvola | 512×512 | uint16 | strided | 1 | 12.9114 | 3.0769 | 4.20x |
| filters.threshold_niblack | 512×512 | uint16 | fortran | 1 | 20.5698 | 4.8228 | 4.27x |
| filters.threshold_sauvola | 512×512 | uint16 | fortran | 1 | 23.1256 | 5.2927 | 4.37x |
| filters.threshold_niblack | 1×512×512 | uint16 | volume | 1 | 29.1279 | 8.2715 | 3.52x |
| filters.threshold_sauvola | 1×512×512 | uint16 | volume | 1 | 30.4270 | 8.3515 | 3.64x |
| filters.threshold_niblack | 512×512 | int16 | window-15 | 1 | 19.3971 | 5.6474 | 3.43x |
| filters.threshold_sauvola | 512×512 | int16 | window-15 | 1 | 17.5740 | 3.7955 | 4.63x |
| filters.threshold_niblack | 512×512 | int16 | window-3 | 1 | 9.2176 | 2.1989 | 4.19x |
| filters.threshold_sauvola | 512×512 | int16 | window-3 | 1 | 12.4483 | 2.7341 | 4.55x |
| filters.threshold_niblack | 512×512 | int16 | window-31 | 1 | 13.5146 | 4.4718 | 3.02x |
| filters.threshold_sauvola | 512×512 | int16 | window-31 | 1 | 12.8415 | 3.1059 | 4.13x |
| filters.threshold_niblack | 512×512 | int16 | window-127 | 1 | 14.0454 | 7.6455 | 1.84x |
| filters.threshold_sauvola | 512×512 | int16 | window-127 | 1 | 15.3506 | 5.3578 | 2.87x |
| filters.threshold_niblack | 512×512 | int16 | window-(3, 31) | 1 | 11.0699 | 2.9657 | 3.73x |
| filters.threshold_sauvola | 512×512 | int16 | window-(3, 31) | 1 | 15.5256 | 4.3158 | 3.60x |
| filters.threshold_niblack | 512×512 | int16 | reversed | 1 | 12.5499 | 3.0635 | 4.10x |
| filters.threshold_sauvola | 512×512 | int16 | reversed | 1 | 13.8829 | 4.8738 | 2.85x |
| filters.threshold_niblack | 512×512 | int16 | strided | 1 | 12.2461 | 3.0030 | 4.08x |
| filters.threshold_sauvola | 512×512 | int16 | strided | 1 | 15.7734 | 4.6000 | 3.43x |
| filters.threshold_niblack | 512×512 | int16 | fortran | 1 | 20.7229 | 3.9890 | 5.20x |
| filters.threshold_sauvola | 512×512 | int16 | fortran | 1 | 23.8633 | 5.3872 | 4.43x |
| filters.threshold_niblack | 1×512×512 | int16 | volume | 1 | 28.3374 | 6.7329 | 4.21x |
| filters.threshold_sauvola | 1×512×512 | int16 | volume | 1 | 24.3532 | 7.6978 | 3.16x |
| filters.threshold_niblack | 512×512 | float32 | window-15 | 1 | 19.0508 | 7.1161 | 2.68x |
| filters.threshold_sauvola | 512×512 | float32 | window-15 | 1 | 15.6201 | 4.5568 | 3.43x |
| filters.threshold_niblack | 512×512 | float32 | window-3 | 1 | 16.2838 | 5.6510 | 2.88x |
| filters.threshold_sauvola | 512×512 | float32 | window-3 | 1 | 11.6717 | 4.0202 | 2.90x |
| filters.threshold_niblack | 512×512 | float32 | window-31 | 1 | 8.4643 | 3.3553 | 2.52x |
| filters.threshold_sauvola | 512×512 | float32 | window-31 | 1 | 10.5426 | 3.5517 | 2.97x |
| filters.threshold_niblack | 512×512 | float32 | window-127 | 1 | 16.1634 | 6.6151 | 2.44x |
| filters.threshold_sauvola | 512×512 | float32 | window-127 | 1 | 14.2960 | 5.1806 | 2.76x |
| filters.threshold_niblack | 512×512 | float32 | window-(3, 31) | 1 | 8.0516 | 2.1191 | 3.80x |
| filters.threshold_sauvola | 512×512 | float32 | window-(3, 31) | 1 | 8.5512 | 2.5511 | 3.35x |
| filters.threshold_niblack | 512×512 | float32 | reversed | 1 | 7.4979 | 3.1832 | 2.36x |
| filters.threshold_sauvola | 512×512 | float32 | reversed | 1 | 7.7812 | 2.9210 | 2.66x |
| filters.threshold_niblack | 512×512 | float32 | strided | 1 | 8.3683 | 3.1135 | 2.69x |
| filters.threshold_sauvola | 512×512 | float32 | strided | 1 | 10.8632 | 4.4124 | 2.46x |
| filters.threshold_niblack | 512×512 | float32 | fortran | 1 | 21.8700 | 3.8693 | 5.65x |
| filters.threshold_sauvola | 512×512 | float32 | fortran | 1 | 27.5591 | 5.3066 | 5.19x |
| filters.threshold_niblack | 1×512×512 | float32 | volume | 1 | 35.3158 | 12.7471 | 2.77x |
| filters.threshold_sauvola | 1×512×512 | float32 | volume | 1 | 24.2237 | 6.9497 | 3.49x |
| filters.threshold_niblack | 512×512 | float32 | int64-window | 1 | 7.2896 | 2.7382 | 2.66x |
| filters.threshold_sauvola | 512×512 | float32 | int64-window | 1 | 10.9572 | 3.1542 | 3.47x |
| filters.threshold_niblack | 512×512 | float32 | k-0.0 | 1 | 10.3826 | 3.0500 | 3.40x |
| filters.threshold_sauvola | 512×512 | float32 | k-0.0 | 1 | 9.4528 | 3.5047 | 2.70x |
| filters.threshold_niblack | 512×512 | float32 | k--0.2 | 1 | 7.5547 | 2.3939 | 3.16x |
| filters.threshold_sauvola | 512×512 | float32 | k--0.2 | 1 | 6.5130 | 2.0520 | 3.17x |
| filters.threshold_niblack | 512×512 | float32 | k-np.float32(0.17) | 1 | 6.3274 | 2.4170 | 2.62x |
| filters.threshold_sauvola | 512×512 | float32 | k-np.float32(0.17) | 1 | 7.6484 | 2.0464 | 3.74x |
| filters.threshold_niblack | 512×512 | float32 | k-np.float64(0.17) | 1 | 7.4108 | 2.2242 | 3.33x |
| filters.threshold_sauvola | 512×512 | float32 | k-np.float64(0.17) | 1 | 7.3936 | 2.3223 | 3.18x |
| filters.threshold_sauvola | 512×512 | float32 | r-0.5 | 1 | 8.5498 | 1.9939 | 4.29x |
| filters.threshold_sauvola | 512×512 | float32 | r-128.0 | 1 | 7.7524 | 2.3688 | 3.27x |
| filters.threshold_sauvola | 512×512 | float32 | r-np.float64(0.5) | 1 | 8.3935 | 2.4144 | 3.48x |
| filters.threshold_niblack | 512×512 | float64 | window-15 | 1 | 9.5214 | 2.2196 | 4.29x |
| filters.threshold_sauvola | 512×512 | float64 | window-15 | 1 | 9.5813 | 2.7898 | 3.43x |
| filters.threshold_niblack | 512×512 | float64 | window-3 | 1 | 8.9384 | 2.2206 | 4.03x |
| filters.threshold_sauvola | 512×512 | float64 | window-3 | 1 | 8.7311 | 2.2738 | 3.84x |
| filters.threshold_niblack | 512×512 | float64 | window-31 | 1 | 8.1299 | 2.2981 | 3.54x |
| filters.threshold_sauvola | 512×512 | float64 | window-31 | 1 | 8.8090 | 2.2023 | 4.00x |
| filters.threshold_niblack | 512×512 | float64 | window-127 | 1 | 11.7224 | 3.3259 | 3.52x |
| filters.threshold_sauvola | 512×512 | float64 | window-127 | 1 | 11.7834 | 4.6982 | 2.51x |
| filters.threshold_niblack | 512×512 | float64 | window-(3, 31) | 1 | 11.8538 | 2.5261 | 4.69x |
| filters.threshold_sauvola | 512×512 | float64 | window-(3, 31) | 1 | 12.8285 | 4.2435 | 3.02x |
| filters.threshold_niblack | 512×512 | float64 | reversed | 1 | 10.4190 | 4.0622 | 2.56x |
| filters.threshold_sauvola | 512×512 | float64 | reversed | 1 | 11.2927 | 4.9068 | 2.30x |
| filters.threshold_niblack | 512×512 | float64 | strided | 1 | 10.6880 | 5.2900 | 2.02x |
| filters.threshold_sauvola | 512×512 | float64 | strided | 1 | 12.2954 | 4.8738 | 2.52x |
| filters.threshold_niblack | 512×512 | float64 | fortran | 1 | 20.8013 | 6.3092 | 3.30x |
| filters.threshold_sauvola | 512×512 | float64 | fortran | 1 | 20.7935 | 6.9467 | 2.99x |
| filters.threshold_niblack | 1×512×512 | float64 | volume | 1 | 22.6791 | 6.4857 | 3.50x |
| filters.threshold_sauvola | 1×512×512 | float64 | volume | 1 | 20.7356 | 8.9523 | 2.32x |
| filters.threshold_niblack | 512×512 | float64 | int64-window | 1 | 11.9487 | 3.8340 | 3.12x |
| filters.threshold_sauvola | 512×512 | float64 | int64-window | 1 | 12.8767 | 4.0659 | 3.17x |
| filters.threshold_niblack | 512×512 | float64 | k-0.0 | 1 | 10.1487 | 4.3172 | 2.35x |
| filters.threshold_sauvola | 512×512 | float64 | k-0.0 | 1 | 14.2362 | 4.8616 | 2.93x |
| filters.threshold_niblack | 512×512 | float64 | k--0.2 | 1 | 13.6194 | 3.9245 | 3.47x |
| filters.threshold_sauvola | 512×512 | float64 | k--0.2 | 1 | 12.8838 | 2.9624 | 4.35x |
| filters.threshold_niblack | 512×512 | float64 | k-np.float32(0.17) | 1 | 15.0554 | 4.4731 | 3.37x |
| filters.threshold_sauvola | 512×512 | float64 | k-np.float32(0.17) | 1 | 12.7968 | 3.4251 | 3.74x |
| filters.threshold_niblack | 512×512 | float64 | k-np.float64(0.17) | 1 | 14.9016 | 3.9986 | 3.73x |
| filters.threshold_sauvola | 512×512 | float64 | k-np.float64(0.17) | 1 | 13.9304 | 3.6900 | 3.78x |
| filters.threshold_sauvola | 512×512 | float64 | r-0.5 | 1 | 14.4771 | 5.7843 | 2.50x |
| filters.threshold_sauvola | 512×512 | float64 | r-128.0 | 1 | 11.6472 | 3.3548 | 3.47x |
| filters.threshold_sauvola | 512×512 | float64 | r-np.float64(0.5) | 1 | 12.8716 | 3.0288 | 4.25x |
| filters.threshold_niblack | 512×512 | bool | additional-width | 1 | 10.7370 | 4.6995 | 2.28x |
| filters.threshold_sauvola | 512×512 | bool | additional-width | 1 | 17.6681 | 4.7795 | 3.70x |
| filters.threshold_niblack | 512×512 | int8 | additional-width | 1 | 15.8207 | 6.1425 | 2.58x |
| filters.threshold_sauvola | 512×512 | int8 | additional-width | 1 | 14.6492 | 3.7898 | 3.87x |
| filters.threshold_niblack | 512×512 | uint32 | additional-width | 1 | 11.2178 | 2.9024 | 3.86x |
| filters.threshold_sauvola | 512×512 | uint32 | additional-width | 1 | 12.3550 | 2.8812 | 4.29x |
| filters.threshold_niblack | 512×512 | int32 | additional-width | 1 | 11.1704 | 3.5957 | 3.11x |
| filters.threshold_sauvola | 512×512 | int32 | additional-width | 1 | 8.3702 | 3.4306 | 2.44x |
| filters.threshold_niblack | 512×512 | uint64 | additional-width | 1 | 9.4826 | 4.0599 | 2.34x |
| filters.threshold_sauvola | 512×512 | uint64 | additional-width | 1 | 11.6205 | 3.7666 | 3.09x |
| filters.threshold_niblack | 512×512 | int64 | additional-width | 1 | 8.3507 | 2.0579 | 4.06x |
| filters.threshold_sauvola | 512×512 | int64 | additional-width | 1 | 13.0163 | 2.5332 | 5.14x |
