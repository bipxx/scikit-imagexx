# Alpha and grayscale performance

Original scikit-image 0.26.0. Reference thread pools use one worker; native workers are explicit. Timings include dispatch, integer normalization, validation, layout handling and output allocation. Input construction is excluded. All samples and slower cases are retained in the JSON.

| API | Input | Layout / workload | Workers | Original ms | C++ ms | Speedup |
|---|---|---|---:|---:|---:|---:|
| color.rgba2rgb | bool | contiguous | 1 | 13.1753 | 0.5337 | 24.68x |
| color.rgba2rgb | bool | reversed | 1 | 13.3568 | 2.0953 | 6.37x |
| color.rgba2rgb | bool | strided | 1 | 13.7457 | 2.1655 | 6.35x |
| color.rgba2rgb | bool | fortran | 1 | 11.0686 | 2.1986 | 5.03x |
| color.rgba2rgb | bool | planar | 1 | 7.8825 | 2.1934 | 3.59x |
| color.rgba2rgb | uint8 | contiguous | 1 | 15.6683 | 0.5943 | 26.36x |
| color.rgba2rgb | uint8 | reversed | 1 | 16.7693 | 2.2605 | 7.42x |
| color.rgba2rgb | uint8 | strided | 1 | 14.9583 | 2.1500 | 6.96x |
| color.rgba2rgb | uint8 | fortran | 1 | 11.8280 | 2.1425 | 5.52x |
| color.rgba2rgb | uint8 | planar | 1 | 8.1476 | 2.3146 | 3.52x |
| color.rgba2rgb | int8 | contiguous | 1 | 15.1049 | 1.6779 | 9.00x |
| color.rgba2rgb | int8 | reversed | 1 | 19.7912 | 3.7510 | 5.28x |
| color.rgba2rgb | int8 | strided | 1 | 15.9385 | 3.4151 | 4.67x |
| color.rgba2rgb | int8 | fortran | 1 | 13.7109 | 2.6928 | 5.09x |
| color.rgba2rgb | int8 | planar | 1 | 10.9722 | 3.8013 | 2.89x |
| color.rgba2rgb | uint16 | contiguous | 1 | 14.8658 | 0.7166 | 20.74x |
| color.rgba2rgb | uint16 | reversed | 1 | 14.5825 | 2.0895 | 6.98x |
| color.rgba2rgb | uint16 | strided | 1 | 15.1356 | 2.2565 | 6.71x |
| color.rgba2rgb | uint16 | fortran | 1 | 12.7932 | 2.3189 | 5.52x |
| color.rgba2rgb | uint16 | planar | 1 | 10.4120 | 2.1452 | 4.85x |
| color.rgba2rgb | int16 | contiguous | 1 | 20.4551 | 1.6193 | 12.63x |
| color.rgba2rgb | int16 | reversed | 1 | 21.8119 | 3.5254 | 6.19x |
| color.rgba2rgb | int16 | strided | 1 | 19.6221 | 3.2624 | 6.01x |
| color.rgba2rgb | int16 | fortran | 1 | 13.3183 | 2.5244 | 5.28x |
| color.rgba2rgb | int16 | planar | 1 | 11.3477 | 2.8431 | 3.99x |
| color.rgba2rgb | uint32 | contiguous | 1 | 17.9009 | 2.1795 | 8.21x |
| color.rgba2rgb | uint32 | reversed | 1 | 14.6632 | 2.2492 | 6.52x |
| color.rgba2rgb | uint32 | strided | 1 | 15.1529 | 2.1025 | 7.21x |
| color.rgba2rgb | uint32 | fortran | 1 | 11.9762 | 2.1561 | 5.55x |
| color.rgba2rgb | uint32 | planar | 1 | 8.9198 | 1.9901 | 4.48x |
| color.rgba2rgb | int32 | contiguous | 1 | 14.3163 | 2.5280 | 5.66x |
| color.rgba2rgb | int32 | reversed | 1 | 14.0995 | 2.3661 | 5.96x |
| color.rgba2rgb | int32 | strided | 1 | 14.2987 | 2.4706 | 5.79x |
| color.rgba2rgb | int32 | fortran | 1 | 11.3669 | 2.1795 | 5.22x |
| color.rgba2rgb | int32 | planar | 1 | 7.6329 | 2.3917 | 3.19x |
| color.rgba2rgb | uint64 | contiguous | 1 | 17.7881 | 4.7748 | 3.73x |
| color.rgba2rgb | uint64 | reversed | 1 | 18.7364 | 4.8397 | 3.87x |
| color.rgba2rgb | uint64 | strided | 1 | 18.0025 | 4.2258 | 4.26x |
| color.rgba2rgb | uint64 | fortran | 1 | 16.8120 | 4.9842 | 3.37x |
| color.rgba2rgb | uint64 | planar | 1 | 13.6522 | 4.7561 | 2.87x |
| color.rgba2rgb | int64 | contiguous | 1 | 17.0123 | 2.9440 | 5.78x |
| color.rgba2rgb | int64 | reversed | 1 | 15.4710 | 2.4410 | 6.34x |
| color.rgba2rgb | int64 | strided | 1 | 16.9499 | 2.8899 | 5.87x |
| color.rgba2rgb | int64 | fortran | 1 | 14.7778 | 3.0843 | 4.79x |
| color.rgba2rgb | int64 | planar | 1 | 9.2045 | 2.6122 | 3.52x |
| color.rgba2rgb | float32 | contiguous | 1 | 11.3222 | 0.6928 | 16.34x |
| color.rgba2rgb | float32 | reversed | 1 | 8.2696 | 1.9488 | 4.24x |
| color.rgba2rgb | float32 | strided | 1 | 9.3029 | 2.9517 | 3.15x |
| color.rgba2rgb | float32 | fortran | 1 | 4.9082 | 0.5721 | 8.58x |
| color.rgba2rgb | float32 | planar | 1 | 3.6740 | 0.8691 | 4.23x |
| color.rgba2rgb | float64 | contiguous | 1 | 11.6168 | 0.8604 | 13.50x |
| color.rgba2rgb | float64 | reversed | 1 | 12.6994 | 4.7095 | 2.70x |
| color.rgba2rgb | float64 | strided | 1 | 13.3381 | 6.0663 | 2.20x |
| color.rgba2rgb | float64 | fortran | 1 | 10.1533 | 1.0606 | 9.57x |
| color.rgba2rgb | float64 | planar | 1 | 6.9363 | 0.8385 | 8.27x |
| color.gray2rgb | bool | contiguous | 1 | 0.3311 | 0.2493 | 1.33x |
| color.gray2rgba | bool | contiguous | 1 | 0.3156 | 0.0708 | 4.46x |
| color.gray2rgb | bool | reversed | 1 | 0.3271 | 0.2196 | 1.49x |
| color.gray2rgba | bool | reversed | 1 | 0.2599 | 0.1221 | 2.13x |
| color.gray2rgb | bool | strided | 1 | 0.3570 | 0.2746 | 1.30x |
| color.gray2rgba | bool | strided | 1 | 0.3137 | 0.0878 | 3.57x |
| color.gray2rgb | bool | fortran | 1 | 0.3518 | 0.2541 | 1.38x |
| color.gray2rgba | bool | fortran | 1 | 0.7097 | 0.1385 | 5.12x |
| color.gray2rgb | uint8 | contiguous | 1 | 0.5740 | 0.3114 | 1.84x |
| color.gray2rgba | uint8 | contiguous | 1 | 0.2693 | 0.0724 | 3.72x |
| color.gray2rgb | uint8 | reversed | 1 | 0.3613 | 0.2466 | 1.47x |
| color.gray2rgba | uint8 | reversed | 1 | 0.2512 | 0.0738 | 3.41x |
| color.gray2rgb | uint8 | strided | 1 | 0.4011 | 0.2830 | 1.42x |
| color.gray2rgba | uint8 | strided | 1 | 0.3404 | 0.0883 | 3.86x |
| color.gray2rgb | uint8 | fortran | 1 | 0.4168 | 0.2321 | 1.80x |
| color.gray2rgba | uint8 | fortran | 1 | 0.7354 | 0.1930 | 3.81x |
| color.gray2rgb | int8 | contiguous | 1 | 0.3649 | 0.2541 | 1.44x |
| color.gray2rgba | int8 | contiguous | 1 | 0.3085 | 0.0805 | 3.83x |
| color.gray2rgb | int8 | reversed | 1 | 0.3470 | 0.2440 | 1.42x |
| color.gray2rgba | int8 | reversed | 1 | 0.3292 | 0.1072 | 3.07x |
| color.gray2rgb | int8 | strided | 1 | 0.3596 | 0.2686 | 1.34x |
| color.gray2rgba | int8 | strided | 1 | 0.3717 | 0.1704 | 2.18x |
| color.gray2rgb | int8 | fortran | 1 | 0.3433 | 0.2498 | 1.37x |
| color.gray2rgba | int8 | fortran | 1 | 0.5855 | 0.1967 | 2.98x |
| color.gray2rgb | uint16 | contiguous | 1 | 0.2472 | 0.2295 | 1.08x |
| color.gray2rgba | uint16 | contiguous | 1 | 1.0301 | 0.6666 | 1.55x |
| color.gray2rgb | uint16 | reversed | 1 | 0.6988 | 0.5590 | 1.25x |
| color.gray2rgba | uint16 | reversed | 1 | 0.2796 | 0.1554 | 1.80x |
| color.gray2rgb | uint16 | strided | 1 | 0.2319 | 0.2080 | 1.11x |
| color.gray2rgba | uint16 | strided | 1 | 0.3698 | 0.1962 | 1.89x |
| color.gray2rgb | uint16 | fortran | 1 | 0.2141 | 0.2040 | 1.05x |
| color.gray2rgba | uint16 | fortran | 1 | 0.5893 | 0.1849 | 3.19x |
| color.gray2rgb | int16 | contiguous | 1 | 0.1893 | 0.1796 | 1.05x |
| color.gray2rgba | int16 | contiguous | 1 | 0.2932 | 0.1191 | 2.46x |
| color.gray2rgb | int16 | reversed | 1 | 0.1875 | 0.1597 | 1.17x |
| color.gray2rgba | int16 | reversed | 1 | 0.2679 | 0.1786 | 1.50x |
| color.gray2rgb | int16 | strided | 1 | 0.2349 | 0.2104 | 1.12x |
| color.gray2rgba | int16 | strided | 1 | 0.3229 | 0.1874 | 1.72x |
| color.gray2rgb | int16 | fortran | 1 | 0.1893 | 0.1938 | 0.98x |
| color.gray2rgba | int16 | fortran | 1 | 0.6197 | 0.1792 | 3.46x |
| color.gray2rgb | uint32 | contiguous | 1 | 0.9097 | 0.7477 | 1.22x |
| color.gray2rgba | uint32 | contiguous | 1 | 0.5052 | 0.2291 | 2.20x |
| color.gray2rgb | uint32 | reversed | 1 | 1.1259 | 0.8319 | 1.35x |
| color.gray2rgba | uint32 | reversed | 1 | 0.5831 | 0.2617 | 2.23x |
| color.gray2rgb | uint32 | strided | 1 | 1.4182 | 1.0335 | 1.37x |
| color.gray2rgba | uint32 | strided | 1 | 0.7547 | 0.3512 | 2.15x |
| color.gray2rgb | uint32 | fortran | 1 | 0.9567 | 0.8490 | 1.13x |
| color.gray2rgba | uint32 | fortran | 1 | 0.9252 | 0.2792 | 3.31x |
| color.gray2rgb | int32 | contiguous | 1 | 0.2943 | 0.1629 | 1.81x |
| color.gray2rgba | int32 | contiguous | 1 | 0.6789 | 0.3229 | 2.10x |
| color.gray2rgb | int32 | reversed | 1 | 0.3228 | 0.2477 | 1.30x |
| color.gray2rgba | int32 | reversed | 1 | 0.6585 | 0.3214 | 2.05x |
| color.gray2rgb | int32 | strided | 1 | 0.4033 | 0.1856 | 2.17x |
| color.gray2rgba | int32 | strided | 1 | 0.6475 | 0.3030 | 2.14x |
| color.gray2rgb | int32 | fortran | 1 | 0.2702 | 0.1518 | 1.78x |
| color.gray2rgba | int32 | fortran | 1 | 0.8134 | 0.3544 | 2.30x |
| color.gray2rgb | uint64 | contiguous | 1 | 2.0262 | 1.5142 | 1.34x |
| color.gray2rgba | uint64 | contiguous | 1 | 3.2909 | 2.2807 | 1.44x |
| color.gray2rgb | uint64 | reversed | 1 | 2.1463 | 1.6553 | 1.30x |
| color.gray2rgba | uint64 | reversed | 1 | 2.7515 | 2.1025 | 1.31x |
| color.gray2rgb | uint64 | strided | 1 | 2.3501 | 1.7040 | 1.38x |
| color.gray2rgba | uint64 | strided | 1 | 3.0701 | 2.0482 | 1.50x |
| color.gray2rgb | uint64 | fortran | 1 | 1.9492 | 1.6256 | 1.20x |
| color.gray2rgba | uint64 | fortran | 1 | 4.7689 | 2.1891 | 2.18x |
| color.gray2rgb | int64 | contiguous | 1 | 2.1119 | 1.6724 | 1.26x |
| color.gray2rgba | int64 | contiguous | 1 | 3.1683 | 2.0935 | 1.51x |
| color.gray2rgb | int64 | reversed | 1 | 1.9714 | 1.5088 | 1.31x |
| color.gray2rgba | int64 | reversed | 1 | 2.8891 | 2.0457 | 1.41x |
| color.gray2rgb | int64 | strided | 1 | 2.1045 | 1.6804 | 1.25x |
| color.gray2rgba | int64 | strided | 1 | 3.2830 | 2.1493 | 1.53x |
| color.gray2rgb | int64 | fortran | 1 | 1.8297 | 1.4658 | 1.25x |
| color.gray2rgba | int64 | fortran | 1 | 4.2053 | 2.0661 | 2.04x |
| color.gray2rgb | float32 | contiguous | 1 | 0.3050 | 0.1521 | 2.00x |
| color.gray2rgba | float32 | contiguous | 1 | 0.5219 | 0.2286 | 2.28x |
| color.gray2rgb | float32 | reversed | 1 | 0.3213 | 0.1654 | 1.94x |
| color.gray2rgba | float32 | reversed | 1 | 0.4841 | 0.2413 | 2.01x |
| color.gray2rgb | float32 | strided | 1 | 0.4091 | 0.2079 | 1.97x |
| color.gray2rgba | float32 | strided | 1 | 0.7069 | 0.3564 | 1.98x |
| color.gray2rgb | float32 | fortran | 1 | 0.3010 | 0.2016 | 1.49x |
| color.gray2rgba | float32 | fortran | 1 | 1.1683 | 0.2441 | 4.79x |
| color.gray2rgb | float64 | contiguous | 1 | 2.0784 | 1.4829 | 1.40x |
| color.gray2rgba | float64 | contiguous | 1 | 3.5693 | 2.3231 | 1.54x |
| color.gray2rgb | float64 | reversed | 1 | 2.1102 | 1.6304 | 1.29x |
| color.gray2rgba | float64 | reversed | 1 | 4.2492 | 2.5299 | 1.68x |
| color.gray2rgb | float64 | strided | 1 | 2.9245 | 2.2276 | 1.31x |
| color.gray2rgba | float64 | strided | 1 | 4.1975 | 2.5118 | 1.67x |
| color.gray2rgb | float64 | fortran | 1 | 2.4453 | 1.6122 | 1.52x |
| color.gray2rgba | float64 | fortran | 1 | 4.6848 | 1.9352 | 2.42x |
| color.gray2rgb | float16 | contiguous | 1 | 0.1885 | 0.1418 | 1.33x |
| color.gray2rgba | float16 | contiguous | 1 | 0.6984 | 0.5338 | 1.31x |
| color.gray2rgb | float16 | reversed | 1 | 0.4714 | 0.1638 | 2.88x |
| color.gray2rgba | float16 | reversed | 1 | 0.7425 | 0.6004 | 1.24x |
| color.gray2rgb | float16 | strided | 1 | 0.5460 | 0.1745 | 3.13x |
| color.gray2rgba | float16 | strided | 1 | 0.9444 | 0.7536 | 1.25x |
| color.gray2rgb | float16 | fortran | 1 | 0.4538 | 0.1918 | 2.37x |
| color.gray2rgba | float16 | fortran | 1 | 1.1291 | 0.5952 | 1.90x |
| color.rgba2rgb | uint8 | background-black | 1 | 15.8330 | 0.6372 | 24.85x |
| color.rgba2rgb | uint8 | background-color | 1 | 15.3047 | 0.7183 | 21.31x |
| color.rgba2rgb | uint8 | background-nan | 1 | 15.2468 | 0.6015 | 25.35x |
| color.gray2rgba | uint8 | alpha-scalar | 1 | 0.2731 | 0.0813 | 3.36x |
| color.gray2rgba | uint8 | alpha-typed-scalar | 1 | 0.5184 | 0.0830 | 6.25x |
| color.gray2rgba | uint8 | alpha-array | 1 | 0.4164 | 0.0975 | 4.27x |
| color.gray2rgba | uint8 | alpha-broadcast | 1 | 0.3368 | 0.0837 | 4.02x |
| color.rgba2rgb | float32 | background-black | 1 | 10.4912 | 0.9045 | 11.60x |
| color.rgba2rgb | float32 | background-color | 1 | 9.4709 | 0.8460 | 11.20x |
| color.rgba2rgb | float32 | background-nan | 1 | 9.7128 | 0.7909 | 12.28x |
| color.gray2rgba | float32 | alpha-scalar | 1 | 0.6889 | 0.2870 | 2.40x |
| color.gray2rgba | float32 | alpha-typed-scalar | 1 | 0.5801 | 0.2532 | 2.29x |
| color.gray2rgba | float32 | alpha-array | 1 | 0.7360 | 0.6915 | 1.06x |
| color.gray2rgba | float32 | alpha-broadcast | 1 | 0.6846 | 0.3055 | 2.24x |
| color.rgba2rgb | float64 | background-black | 1 | 12.5694 | 1.1542 | 10.89x |
| color.rgba2rgb | float64 | background-color | 1 | 12.8853 | 1.1006 | 11.71x |
| color.rgba2rgb | float64 | background-nan | 1 | 12.4673 | 0.9231 | 13.51x |
| color.gray2rgba | float64 | alpha-scalar | 1 | 5.4834 | 3.5243 | 1.56x |
| color.gray2rgba | float64 | alpha-typed-scalar | 1 | 3.8150 | 2.7878 | 1.37x |
| color.gray2rgba | float64 | alpha-array | 1 | 3.4281 | 3.1162 | 1.10x |
| color.gray2rgba | float64 | alpha-broadcast | 1 | 3.2354 | 2.5251 | 1.28x |
| color.gray2rgb | >u2 | endian-contiguous | 1 | 0.2820 | 0.1605 | 1.76x |
| color.gray2rgba | >u2 | endian-contiguous | 1 | 0.3808 | 0.1352 | 2.82x |
| color.gray2rgb | >u2 | endian-reversed | 1 | 0.2989 | 0.1765 | 1.69x |
| color.gray2rgba | >u2 | endian-reversed | 1 | 0.3479 | 0.1409 | 2.47x |
| color.gray2rgb | >f4 | endian-contiguous | 1 | 0.3172 | 0.2861 | 1.11x |
| color.gray2rgba | >f4 | endian-contiguous | 1 | 0.7511 | 0.4636 | 1.62x |
| color.gray2rgb | >f4 | endian-reversed | 1 | 0.3631 | 0.2537 | 1.43x |
| color.gray2rgba | >f4 | endian-reversed | 1 | 0.9206 | 0.4440 | 2.07x |
| color.gray2rgb | >f8 | endian-contiguous | 1 | 2.3749 | 1.7646 | 1.35x |
| color.gray2rgba | >f8 | endian-contiguous | 1 | 4.0598 | 2.8092 | 1.45x |
| color.gray2rgb | >f8 | endian-reversed | 1 | 2.8018 | 2.0739 | 1.35x |
| color.gray2rgba | >f8 | endian-reversed | 1 | 3.5937 | 2.4053 | 1.49x |
| color.gray2rgb | [('x', '>i2'), ('y', '>f4')] | record-6-contiguous | 1 | 1.3634 | 0.3334 | 4.09x |
| color.gray2rgba | [('x', '>i2'), ('y', '>f4')] | record-6-contiguous | 1 | 3.4206 | 2.0449 | 1.67x |
| color.gray2rgb | [('x', '>i2'), ('y', '>f4')] | record-6-reversed | 1 | 1.3785 | 0.3021 | 4.56x |
| color.gray2rgba | [('x', '>i2'), ('y', '>f4')] | record-6-reversed | 1 | 4.4087 | 1.6151 | 2.73x |
| color.gray2rgb | [('x', '>i2'), ('y', '>f4')] | record-6-strided | 1 | 1.8467 | 0.7322 | 2.52x |
| color.gray2rgba | [('x', '>i2'), ('y', '>f4')] | record-6-strided | 1 | 6.3804 | 2.0591 | 3.10x |
| color.gray2rgb | [('x', '>i2'), ('y', '>f4')] | record-6-fortran | 1 | 1.5442 | 0.2402 | 6.43x |
| color.gray2rgba | [('x', '>i2'), ('y', '>f4')] | record-6-fortran | 1 | 3.5961 | 1.4160 | 2.54x |
| color.gray2rgb | [('record', [('a', '>i2'), ('b', '>u4')]), ('vec', '>f4', (3,))] | record-18-contiguous | 1 | 11.2829 | 9.6498 | 1.17x |
| color.gray2rgba | [('record', [('a', '>i2'), ('b', '>u4')]), ('vec', '>f4', (3,))] | record-18-contiguous | 1 | 26.8291 | 17.9414 | 1.50x |
| color.gray2rgb | [('record', [('a', '>i2'), ('b', '>u4')]), ('vec', '>f4', (3,))] | record-18-reversed | 1 | 13.8346 | 8.5518 | 1.62x |
| color.gray2rgba | [('record', [('a', '>i2'), ('b', '>u4')]), ('vec', '>f4', (3,))] | record-18-reversed | 1 | 28.8026 | 18.9762 | 1.52x |
| color.gray2rgb | [('record', [('a', '>i2'), ('b', '>u4')]), ('vec', '>f4', (3,))] | record-18-strided | 1 | 16.4680 | 9.1244 | 1.80x |
| color.gray2rgba | [('record', [('a', '>i2'), ('b', '>u4')]), ('vec', '>f4', (3,))] | record-18-strided | 1 | 26.0061 | 17.3423 | 1.50x |
| color.gray2rgb | [('record', [('a', '>i2'), ('b', '>u4')]), ('vec', '>f4', (3,))] | record-18-fortran | 1 | 8.9038 | 6.5359 | 1.36x |
| color.gray2rgba | [('record', [('a', '>i2'), ('b', '>u4')]), ('vec', '>f4', (3,))] | record-18-fortran | 1 | 23.5761 | 16.1013 | 1.46x |
| color.gray2rgb | [('z', '>c16'), ('when', '>M8[ns]'), ('dt', '>m8[ms]')] | record-32-contiguous | 1 | 15.4124 | 7.1855 | 2.14x |
| color.gray2rgba | [('z', '>c16'), ('when', '>M8[ns]'), ('dt', '>m8[ms]')] | record-32-contiguous | 1 | 37.9204 | 19.2320 | 1.97x |
| color.gray2rgb | [('z', '>c16'), ('when', '>M8[ns]'), ('dt', '>m8[ms]')] | record-32-reversed | 1 | 15.6525 | 8.2315 | 1.90x |
| color.gray2rgba | [('z', '>c16'), ('when', '>M8[ns]'), ('dt', '>m8[ms]')] | record-32-reversed | 1 | 34.6108 | 20.3860 | 1.70x |
| color.gray2rgb | [('z', '>c16'), ('when', '>M8[ns]'), ('dt', '>m8[ms]')] | record-32-strided | 1 | 22.0734 | 10.0047 | 2.21x |
| color.gray2rgba | [('z', '>c16'), ('when', '>M8[ns]'), ('dt', '>m8[ms]')] | record-32-strided | 1 | 46.1702 | 24.7656 | 1.86x |
| color.gray2rgb | [('z', '>c16'), ('when', '>M8[ns]'), ('dt', '>m8[ms]')] | record-32-fortran | 1 | 24.0892 | 9.7879 | 2.46x |
| color.gray2rgba | [('z', '>c16'), ('when', '>M8[ns]'), ('dt', '>m8[ms]')] | record-32-fortran | 1 | 39.1293 | 20.2989 | 1.93x |
