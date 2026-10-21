# Alpha and grayscale performance

Original scikit-image 0.26.0. Reference thread pools use one worker; native workers are explicit. Timings include dispatch, integer normalization, validation, layout handling and output allocation. Input construction is excluded. All samples and slower cases are retained in the JSON.

| API | Input | Layout / workload | Workers | Original ms | C++ ms | Speedup |
|---|---|---|---:|---:|---:|---:|
| color.rgba2rgb | bool | contiguous | 1 | 16.8521 | 0.5881 | 28.65x |
| color.rgba2rgb | bool | reversed | 1 | 16.0161 | 2.1059 | 7.61x |
| color.rgba2rgb | bool | strided | 1 | 15.4414 | 2.3378 | 6.61x |
| color.rgba2rgb | bool | fortran | 1 | 15.7231 | 2.2459 | 7.00x |
| color.rgba2rgb | bool | planar | 1 | 13.0638 | 2.4939 | 5.24x |
| color.rgba2rgb | uint8 | contiguous | 1 | 19.4919 | 0.8017 | 24.31x |
| color.rgba2rgb | uint8 | reversed | 1 | 20.1386 | 2.5419 | 7.92x |
| color.rgba2rgb | uint8 | strided | 1 | 20.2299 | 2.5141 | 8.05x |
| color.rgba2rgb | uint8 | fortran | 1 | 13.8696 | 2.6819 | 5.17x |
| color.rgba2rgb | uint8 | planar | 1 | 11.3829 | 2.4411 | 4.66x |
| color.rgba2rgb | int8 | contiguous | 1 | 18.7064 | 1.9152 | 9.77x |
| color.rgba2rgb | int8 | reversed | 1 | 20.0764 | 4.2297 | 4.75x |
| color.rgba2rgb | int8 | strided | 1 | 20.2437 | 3.0490 | 6.64x |
| color.rgba2rgb | int8 | fortran | 1 | 18.0847 | 4.2298 | 4.28x |
| color.rgba2rgb | int8 | planar | 1 | 14.5233 | 4.2814 | 3.39x |
| color.rgba2rgb | uint16 | contiguous | 1 | 19.3135 | 0.9383 | 20.58x |
| color.rgba2rgb | uint16 | reversed | 1 | 19.6750 | 2.2403 | 8.78x |
| color.rgba2rgb | uint16 | strided | 1 | 16.3797 | 2.3834 | 6.87x |
| color.rgba2rgb | uint16 | fortran | 1 | 14.3285 | 2.8855 | 4.97x |
| color.rgba2rgb | uint16 | planar | 1 | 10.5397 | 2.4699 | 4.27x |
| color.rgba2rgb | int16 | contiguous | 1 | 17.4655 | 1.4799 | 11.80x |
| color.rgba2rgb | int16 | reversed | 1 | 18.6125 | 2.7395 | 6.79x |
| color.rgba2rgb | int16 | strided | 1 | 17.8041 | 2.7674 | 6.43x |
| color.rgba2rgb | int16 | fortran | 1 | 15.5830 | 2.6825 | 5.81x |
| color.rgba2rgb | int16 | planar | 1 | 11.8436 | 2.8987 | 4.09x |
| color.rgba2rgb | uint32 | contiguous | 1 | 16.4083 | 2.4357 | 6.74x |
| color.rgba2rgb | uint32 | reversed | 1 | 20.6385 | 2.3935 | 8.62x |
| color.rgba2rgb | uint32 | strided | 1 | 19.9087 | 2.4155 | 8.24x |
| color.rgba2rgb | uint32 | fortran | 1 | 14.5016 | 2.3725 | 6.11x |
| color.rgba2rgb | uint32 | planar | 1 | 10.6651 | 2.4623 | 4.33x |
| color.rgba2rgb | int32 | contiguous | 1 | 16.8859 | 2.8582 | 5.91x |
| color.rgba2rgb | int32 | reversed | 1 | 19.9522 | 3.1743 | 6.29x |
| color.rgba2rgb | int32 | strided | 1 | 19.6246 | 3.1392 | 6.25x |
| color.rgba2rgb | int32 | fortran | 1 | 15.7171 | 2.9562 | 5.32x |
| color.rgba2rgb | int32 | planar | 1 | 10.5561 | 2.7674 | 3.81x |
| color.rgba2rgb | uint64 | contiguous | 1 | 20.0358 | 4.8495 | 4.13x |
| color.rgba2rgb | uint64 | reversed | 1 | 22.5247 | 6.5882 | 3.42x |
| color.rgba2rgb | uint64 | strided | 1 | 20.9757 | 6.6778 | 3.14x |
| color.rgba2rgb | uint64 | fortran | 1 | 16.4550 | 5.0386 | 3.27x |
| color.rgba2rgb | uint64 | planar | 1 | 14.4480 | 5.0070 | 2.89x |
| color.rgba2rgb | int64 | contiguous | 1 | 16.4874 | 3.0709 | 5.37x |
| color.rgba2rgb | int64 | reversed | 1 | 17.9039 | 2.9715 | 6.03x |
| color.rgba2rgb | int64 | strided | 1 | 18.6134 | 3.4446 | 5.40x |
| color.rgba2rgb | int64 | fortran | 1 | 18.5173 | 3.0611 | 6.05x |
| color.rgba2rgb | int64 | planar | 1 | 13.3821 | 3.3063 | 4.05x |
| color.rgba2rgb | float32 | contiguous | 1 | 12.4889 | 0.8198 | 15.23x |
| color.rgba2rgb | float32 | reversed | 1 | 10.3961 | 2.4385 | 4.26x |
| color.rgba2rgb | float32 | strided | 1 | 10.2236 | 2.5694 | 3.98x |
| color.rgba2rgb | float32 | fortran | 1 | 6.0373 | 0.7026 | 8.59x |
| color.rgba2rgb | float32 | planar | 1 | 5.0920 | 0.7713 | 6.60x |
| color.rgba2rgb | float64 | contiguous | 1 | 13.2453 | 1.0845 | 12.21x |
| color.rgba2rgb | float64 | reversed | 1 | 16.3880 | 6.9884 | 2.35x |
| color.rgba2rgb | float64 | strided | 1 | 18.2117 | 9.4643 | 1.92x |
| color.rgba2rgb | float64 | fortran | 1 | 13.0888 | 2.0918 | 6.26x |
| color.rgba2rgb | float64 | planar | 1 | 6.9795 | 1.1111 | 6.28x |
| color.gray2rgb | bool | contiguous | 1 | 0.3078 | 0.1138 | 2.70x |
| color.gray2rgba | bool | contiguous | 1 | 0.3390 | 0.0815 | 4.16x |
| color.gray2rgb | bool | reversed | 1 | 0.2281 | 0.0988 | 2.31x |
| color.gray2rgba | bool | reversed | 1 | 0.3105 | 0.0861 | 3.61x |
| color.gray2rgb | bool | strided | 1 | 0.3329 | 0.2642 | 1.26x |
| color.gray2rgba | bool | strided | 1 | 0.3279 | 0.0939 | 3.49x |
| color.gray2rgb | bool | fortran | 1 | 0.3362 | 0.2815 | 1.19x |
| color.gray2rgba | bool | fortran | 1 | 0.5115 | 0.1001 | 5.11x |
| color.gray2rgb | uint8 | contiguous | 1 | 0.3482 | 0.2290 | 1.52x |
| color.gray2rgba | uint8 | contiguous | 1 | 0.3049 | 0.1043 | 2.92x |
| color.gray2rgb | uint8 | reversed | 1 | 0.3479 | 0.2261 | 1.54x |
| color.gray2rgba | uint8 | reversed | 1 | 0.2841 | 0.0891 | 3.19x |
| color.gray2rgb | uint8 | strided | 1 | 0.4042 | 0.2650 | 1.53x |
| color.gray2rgba | uint8 | strided | 1 | 0.5546 | 0.1296 | 4.28x |
| color.gray2rgb | uint8 | fortran | 1 | 0.4526 | 0.2904 | 1.56x |
| color.gray2rgba | uint8 | fortran | 1 | 0.6547 | 0.1010 | 6.48x |
| color.gray2rgb | int8 | contiguous | 1 | 0.5715 | 0.3130 | 1.83x |
| color.gray2rgba | int8 | contiguous | 1 | 0.2853 | 0.0822 | 3.47x |
| color.gray2rgb | int8 | reversed | 1 | 0.3722 | 0.2862 | 1.30x |
| color.gray2rgba | int8 | reversed | 1 | 0.2740 | 0.0872 | 3.14x |
| color.gray2rgb | int8 | strided | 1 | 0.4054 | 0.2824 | 1.44x |
| color.gray2rgba | int8 | strided | 1 | 0.4814 | 0.1155 | 4.17x |
| color.gray2rgb | int8 | fortran | 1 | 0.4938 | 0.3223 | 1.53x |
| color.gray2rgba | int8 | fortran | 1 | 0.5850 | 0.1116 | 5.24x |
| color.gray2rgb | uint16 | contiguous | 1 | 0.2483 | 0.1609 | 1.54x |
| color.gray2rgba | uint16 | contiguous | 1 | 0.8609 | 0.5995 | 1.44x |
| color.gray2rgb | uint16 | reversed | 1 | 0.6716 | 0.5159 | 1.30x |
| color.gray2rgba | uint16 | reversed | 1 | 0.5864 | 0.1920 | 3.05x |
| color.gray2rgb | uint16 | strided | 1 | 0.6666 | 0.4610 | 1.45x |
| color.gray2rgba | uint16 | strided | 1 | 0.5448 | 0.1735 | 3.14x |
| color.gray2rgb | uint16 | fortran | 1 | 0.2556 | 0.1778 | 1.44x |
| color.gray2rgba | uint16 | fortran | 1 | 0.7310 | 0.2221 | 3.29x |
| color.gray2rgb | int16 | contiguous | 1 | 0.2491 | 0.1997 | 1.25x |
| color.gray2rgba | int16 | contiguous | 1 | 0.3229 | 0.1233 | 2.62x |
| color.gray2rgb | int16 | reversed | 1 | 0.2152 | 0.1469 | 1.46x |
| color.gray2rgba | int16 | reversed | 1 | 0.2683 | 0.1392 | 1.93x |
| color.gray2rgb | int16 | strided | 1 | 0.2958 | 0.1942 | 1.52x |
| color.gray2rgba | int16 | strided | 1 | 0.5018 | 0.1878 | 2.67x |
| color.gray2rgb | int16 | fortran | 1 | 0.3044 | 0.1624 | 1.87x |
| color.gray2rgba | int16 | fortran | 1 | 0.6378 | 0.1478 | 4.31x |
| color.gray2rgb | uint32 | contiguous | 1 | 1.3261 | 0.8408 | 1.58x |
| color.gray2rgba | uint32 | contiguous | 1 | 0.5715 | 0.2787 | 2.05x |
| color.gray2rgb | uint32 | reversed | 1 | 1.1890 | 1.0236 | 1.16x |
| color.gray2rgba | uint32 | reversed | 1 | 0.6344 | 0.2233 | 2.84x |
| color.gray2rgb | uint32 | strided | 1 | 1.4600 | 1.0178 | 1.43x |
| color.gray2rgba | uint32 | strided | 1 | 0.5700 | 0.3141 | 1.81x |
| color.gray2rgb | uint32 | fortran | 1 | 1.1009 | 0.9739 | 1.13x |
| color.gray2rgba | uint32 | fortran | 1 | 0.9255 | 0.2260 | 4.09x |
| color.gray2rgb | int32 | contiguous | 1 | 0.3314 | 0.1431 | 2.32x |
| color.gray2rgba | int32 | contiguous | 1 | 0.5433 | 0.2315 | 2.35x |
| color.gray2rgb | int32 | reversed | 1 | 0.2970 | 0.1523 | 1.95x |
| color.gray2rgba | int32 | reversed | 1 | 0.5325 | 0.2238 | 2.38x |
| color.gray2rgb | int32 | strided | 1 | 0.4437 | 0.2307 | 1.92x |
| color.gray2rgba | int32 | strided | 1 | 0.7196 | 0.3723 | 1.93x |
| color.gray2rgb | int32 | fortran | 1 | 0.3042 | 0.1632 | 1.86x |
| color.gray2rgba | int32 | fortran | 1 | 0.9468 | 0.2336 | 4.05x |
| color.gray2rgb | uint64 | contiguous | 1 | 2.4086 | 1.8304 | 1.32x |
| color.gray2rgba | uint64 | contiguous | 1 | 3.5561 | 2.3903 | 1.49x |
| color.gray2rgb | uint64 | reversed | 1 | 2.9908 | 1.9380 | 1.54x |
| color.gray2rgba | uint64 | reversed | 1 | 4.7513 | 2.6565 | 1.79x |
| color.gray2rgb | uint64 | strided | 1 | 4.7852 | 2.6043 | 1.84x |
| color.gray2rgba | uint64 | strided | 1 | 4.3404 | 2.5514 | 1.70x |
| color.gray2rgb | uint64 | fortran | 1 | 2.1388 | 0.5068 | 4.22x |
| color.gray2rgba | uint64 | fortran | 1 | 5.1753 | 2.2162 | 2.34x |
| color.gray2rgb | int64 | contiguous | 1 | 0.6918 | 0.4253 | 1.63x |
| color.gray2rgba | int64 | contiguous | 1 | 3.6366 | 2.5471 | 1.43x |
| color.gray2rgb | int64 | reversed | 1 | 0.7359 | 0.3610 | 2.04x |
| color.gray2rgba | int64 | reversed | 1 | 3.7547 | 2.2698 | 1.65x |
| color.gray2rgb | int64 | strided | 1 | 1.0659 | 0.5901 | 1.81x |
| color.gray2rgba | int64 | strided | 1 | 5.3886 | 2.9915 | 1.80x |
| color.gray2rgb | int64 | fortran | 1 | 0.8257 | 0.4173 | 1.98x |
| color.gray2rgba | int64 | fortran | 1 | 6.0311 | 2.5415 | 2.37x |
| color.gray2rgb | float32 | contiguous | 1 | 0.3015 | 0.1664 | 1.81x |
| color.gray2rgba | float32 | contiguous | 1 | 0.4906 | 0.2514 | 1.95x |
| color.gray2rgb | float32 | reversed | 1 | 0.3657 | 0.1949 | 1.88x |
| color.gray2rgba | float32 | reversed | 1 | 0.5145 | 0.2225 | 2.31x |
| color.gray2rgb | float32 | strided | 1 | 0.4036 | 0.2075 | 1.94x |
| color.gray2rgba | float32 | strided | 1 | 0.6196 | 0.2977 | 2.08x |
| color.gray2rgb | float32 | fortran | 1 | 0.3268 | 0.1582 | 2.07x |
| color.gray2rgba | float32 | fortran | 1 | 0.9071 | 0.2360 | 3.84x |
| color.gray2rgb | float64 | contiguous | 1 | 2.2541 | 1.6945 | 1.33x |
| color.gray2rgba | float64 | contiguous | 1 | 5.0136 | 3.0832 | 1.63x |
| color.gray2rgb | float64 | reversed | 1 | 2.0994 | 1.5578 | 1.35x |
| color.gray2rgba | float64 | reversed | 1 | 3.5128 | 2.2774 | 1.54x |
| color.gray2rgb | float64 | strided | 1 | 3.1193 | 1.9288 | 1.62x |
| color.gray2rgba | float64 | strided | 1 | 4.2427 | 2.4513 | 1.73x |
| color.gray2rgb | float64 | fortran | 1 | 2.0355 | 1.5079 | 1.35x |
| color.gray2rgba | float64 | fortran | 1 | 4.9037 | 2.2490 | 2.18x |
| color.gray2rgb | float16 | contiguous | 1 | 0.1972 | 0.1476 | 1.34x |
| color.gray2rgba | float16 | contiguous | 1 | 0.7547 | 0.5771 | 1.31x |
| color.gray2rgb | float16 | reversed | 1 | 0.5728 | 0.4843 | 1.18x |
| color.gray2rgba | float16 | reversed | 1 | 0.8255 | 0.6366 | 1.30x |
| color.gray2rgb | float16 | strided | 1 | 0.5255 | 0.4485 | 1.17x |
| color.gray2rgba | float16 | strided | 1 | 0.8982 | 0.6862 | 1.31x |
| color.gray2rgb | float16 | fortran | 1 | 0.6296 | 0.2618 | 2.40x |
| color.gray2rgba | float16 | fortran | 1 | 1.3757 | 0.8014 | 1.72x |
| color.rgba2rgb | uint8 | background-black | 1 | 16.5287 | 0.9344 | 17.69x |
| color.rgba2rgb | uint8 | background-color | 1 | 15.4022 | 0.7719 | 19.95x |
| color.rgba2rgb | uint8 | background-nan | 1 | 15.3027 | 0.6725 | 22.75x |
| color.gray2rgba | uint8 | alpha-scalar | 1 | 0.2790 | 0.0791 | 3.53x |
| color.gray2rgba | uint8 | alpha-typed-scalar | 1 | 0.3555 | 0.0868 | 4.09x |
| color.gray2rgba | uint8 | alpha-array | 1 | 0.3553 | 0.0794 | 4.47x |
| color.gray2rgba | uint8 | alpha-broadcast | 1 | 0.3075 | 0.0689 | 4.46x |
| color.rgba2rgb | float32 | background-black | 1 | 9.3385 | 0.7465 | 12.51x |
| color.rgba2rgb | float32 | background-color | 1 | 10.8559 | 0.8191 | 13.25x |
| color.rgba2rgb | float32 | background-nan | 1 | 9.2785 | 0.6750 | 13.75x |
| color.gray2rgba | float32 | alpha-scalar | 1 | 0.5963 | 0.2926 | 2.04x |
| color.gray2rgba | float32 | alpha-typed-scalar | 1 | 0.5618 | 0.2326 | 2.42x |
| color.gray2rgba | float32 | alpha-array | 1 | 0.6855 | 0.6585 | 1.04x |
| color.gray2rgba | float32 | alpha-broadcast | 1 | 0.5279 | 0.2325 | 2.27x |
| color.rgba2rgb | float64 | background-black | 1 | 13.5217 | 1.2833 | 10.54x |
| color.rgba2rgb | float64 | background-color | 1 | 12.6350 | 1.1658 | 10.84x |
| color.rgba2rgb | float64 | background-nan | 1 | 15.5882 | 2.3190 | 6.72x |
| color.gray2rgba | float64 | alpha-scalar | 1 | 3.5669 | 2.4011 | 1.49x |
| color.gray2rgba | float64 | alpha-typed-scalar | 1 | 3.9074 | 2.4402 | 1.60x |
| color.gray2rgba | float64 | alpha-array | 1 | 6.1507 | 3.9917 | 1.54x |
| color.gray2rgba | float64 | alpha-broadcast | 1 | 4.5528 | 2.9380 | 1.55x |
| color.gray2rgb | >u2 | endian-contiguous | 1 | 0.3401 | 0.1967 | 1.73x |
| color.gray2rgba | >u2 | endian-contiguous | 1 | 0.3579 | 0.2280 | 1.57x |
| color.gray2rgb | >u2 | endian-reversed | 1 | 0.2282 | 0.1597 | 1.43x |
| color.gray2rgba | >u2 | endian-reversed | 1 | 0.3060 | 0.1367 | 2.24x |
| color.gray2rgb | >f4 | endian-contiguous | 1 | 0.2904 | 0.2140 | 1.36x |
| color.gray2rgba | >f4 | endian-contiguous | 1 | 0.5989 | 0.5579 | 1.07x |
| color.gray2rgb | >f4 | endian-reversed | 1 | 0.3271 | 0.2274 | 1.44x |
| color.gray2rgba | >f4 | endian-reversed | 1 | 0.5432 | 0.4257 | 1.28x |
| color.gray2rgb | >f8 | endian-contiguous | 1 | 2.3323 | 1.7679 | 1.32x |
| color.gray2rgba | >f8 | endian-contiguous | 1 | 3.4311 | 2.5304 | 1.36x |
| color.gray2rgb | >f8 | endian-reversed | 1 | 2.7598 | 1.8973 | 1.45x |
| color.gray2rgba | >f8 | endian-reversed | 1 | 3.4965 | 2.5711 | 1.36x |
| color.gray2rgb | [('x', '>i2'), ('y', '>f4')] | record-6-contiguous | 1 | 1.1999 | 0.2049 | 5.86x |
| color.gray2rgba | [('x', '>i2'), ('y', '>f4')] | record-6-contiguous | 1 | 4.5376 | 3.0228 | 1.50x |
| color.gray2rgb | [('x', '>i2'), ('y', '>f4')] | record-6-reversed | 1 | 1.1990 | 0.2568 | 4.67x |
| color.gray2rgba | [('x', '>i2'), ('y', '>f4')] | record-6-reversed | 1 | 4.3497 | 2.7276 | 1.59x |
| color.gray2rgb | [('x', '>i2'), ('y', '>f4')] | record-6-strided | 1 | 1.4041 | 0.4982 | 2.82x |
| color.gray2rgba | [('x', '>i2'), ('y', '>f4')] | record-6-strided | 1 | 5.0626 | 2.9268 | 1.73x |
| color.gray2rgb | [('x', '>i2'), ('y', '>f4')] | record-6-fortran | 1 | 1.1969 | 0.2432 | 4.92x |
| color.gray2rgba | [('x', '>i2'), ('y', '>f4')] | record-6-fortran | 1 | 4.9549 | 2.8621 | 1.73x |
| color.gray2rgb | [('record', [('a', '>i2'), ('b', '>u4')]), ('vec', '>f4', (3,))] | record-18-contiguous | 1 | 9.6674 | 6.4116 | 1.51x |
| color.gray2rgba | [('record', [('a', '>i2'), ('b', '>u4')]), ('vec', '>f4', (3,))] | record-18-contiguous | 1 | 24.5042 | 15.1368 | 1.62x |
| color.gray2rgb | [('record', [('a', '>i2'), ('b', '>u4')]), ('vec', '>f4', (3,))] | record-18-reversed | 1 | 9.3843 | 7.7307 | 1.21x |
| color.gray2rgba | [('record', [('a', '>i2'), ('b', '>u4')]), ('vec', '>f4', (3,))] | record-18-reversed | 1 | 19.7063 | 14.8415 | 1.33x |
| color.gray2rgb | [('record', [('a', '>i2'), ('b', '>u4')]), ('vec', '>f4', (3,))] | record-18-strided | 1 | 14.2280 | 8.8231 | 1.61x |
| color.gray2rgba | [('record', [('a', '>i2'), ('b', '>u4')]), ('vec', '>f4', (3,))] | record-18-strided | 1 | 23.4249 | 15.6522 | 1.50x |
| color.gray2rgb | [('record', [('a', '>i2'), ('b', '>u4')]), ('vec', '>f4', (3,))] | record-18-fortran | 1 | 10.7778 | 8.6119 | 1.25x |
| color.gray2rgba | [('record', [('a', '>i2'), ('b', '>u4')]), ('vec', '>f4', (3,))] | record-18-fortran | 1 | 22.9577 | 16.4061 | 1.40x |
| color.gray2rgb | [('z', '>c16'), ('when', '>M8[ns]'), ('dt', '>m8[ms]')] | record-32-contiguous | 1 | 17.5039 | 9.4446 | 1.85x |
| color.gray2rgba | [('z', '>c16'), ('when', '>M8[ns]'), ('dt', '>m8[ms]')] | record-32-contiguous | 1 | 31.2663 | 17.2285 | 1.81x |
| color.gray2rgb | [('z', '>c16'), ('when', '>M8[ns]'), ('dt', '>m8[ms]')] | record-32-reversed | 1 | 14.7897 | 6.7237 | 2.20x |
| color.gray2rgba | [('z', '>c16'), ('when', '>M8[ns]'), ('dt', '>m8[ms]')] | record-32-reversed | 1 | 32.9550 | 15.1975 | 2.17x |
| color.gray2rgb | [('z', '>c16'), ('when', '>M8[ns]'), ('dt', '>m8[ms]')] | record-32-strided | 1 | 15.3874 | 8.2809 | 1.86x |
| color.gray2rgba | [('z', '>c16'), ('when', '>M8[ns]'), ('dt', '>m8[ms]')] | record-32-strided | 1 | 31.1833 | 19.2077 | 1.62x |
| color.gray2rgb | [('z', '>c16'), ('when', '>M8[ns]'), ('dt', '>m8[ms]')] | record-32-fortran | 1 | 13.1397 | 7.2206 | 1.82x |
| color.gray2rgba | [('z', '>c16'), ('when', '>M8[ns]'), ('dt', '>m8[ms]')] | record-32-fortran | 1 | 27.5294 | 15.3062 | 1.80x |
