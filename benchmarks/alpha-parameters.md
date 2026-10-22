# Alpha and grayscale performance

Original scikit-image 0.26.0. Reference thread pools use one worker; native workers are explicit. Timings include dispatch, integer normalization, validation, layout handling and output allocation. Input construction is excluded. All samples and slower cases are retained in the JSON.

| API | Input | Layout / workload | Workers | Original ms | C++ ms | Speedup |
|---|---|---|---:|---:|---:|---:|
| color.rgba2rgb | bool | contiguous | 1 | 14.1477 | 0.5794 | 24.42x |
| color.rgba2rgb | bool | reversed | 1 | 16.0491 | 2.4350 | 6.59x |
| color.rgba2rgb | bool | strided | 1 | 17.2962 | 2.3896 | 7.24x |
| color.rgba2rgb | bool | fortran | 1 | 14.4294 | 2.3956 | 6.02x |
| color.rgba2rgb | bool | planar | 1 | 15.4518 | 3.0082 | 5.14x |
| color.rgba2rgb | uint8 | contiguous | 1 | 20.1651 | 0.9169 | 21.99x |
| color.rgba2rgb | uint8 | reversed | 1 | 21.8723 | 2.9743 | 7.35x |
| color.rgba2rgb | uint8 | strided | 1 | 24.8087 | 2.7602 | 8.99x |
| color.rgba2rgb | uint8 | fortran | 1 | 18.0775 | 2.6609 | 6.79x |
| color.rgba2rgb | uint8 | planar | 1 | 12.4934 | 2.9498 | 4.24x |
| color.rgba2rgb | int8 | contiguous | 1 | 23.3124 | 1.7362 | 13.43x |
| color.rgba2rgb | int8 | reversed | 1 | 21.4926 | 4.3109 | 4.99x |
| color.rgba2rgb | int8 | strided | 1 | 22.5002 | 4.2915 | 5.24x |
| color.rgba2rgb | int8 | fortran | 1 | 20.5502 | 3.3922 | 6.06x |
| color.rgba2rgb | int8 | planar | 1 | 14.6663 | 4.2300 | 3.47x |
| color.rgba2rgb | uint16 | contiguous | 1 | 18.6822 | 0.9962 | 18.75x |
| color.rgba2rgb | uint16 | reversed | 1 | 19.9532 | 2.9674 | 6.72x |
| color.rgba2rgb | uint16 | strided | 1 | 17.6916 | 2.7078 | 6.53x |
| color.rgba2rgb | uint16 | fortran | 1 | 13.8426 | 2.5968 | 5.33x |
| color.rgba2rgb | uint16 | planar | 1 | 9.9588 | 2.2113 | 4.50x |
| color.rgba2rgb | int16 | contiguous | 1 | 16.5149 | 0.8998 | 18.35x |
| color.rgba2rgb | int16 | reversed | 1 | 22.3922 | 3.0302 | 7.39x |
| color.rgba2rgb | int16 | strided | 1 | 19.3700 | 2.9240 | 6.62x |
| color.rgba2rgb | int16 | fortran | 1 | 15.8554 | 2.7556 | 5.75x |
| color.rgba2rgb | int16 | planar | 1 | 13.6782 | 3.0436 | 4.49x |
| color.rgba2rgb | uint32 | contiguous | 1 | 19.5710 | 2.5156 | 7.78x |
| color.rgba2rgb | uint32 | reversed | 1 | 17.8412 | 2.5301 | 7.05x |
| color.rgba2rgb | uint32 | strided | 1 | 17.1136 | 2.6199 | 6.53x |
| color.rgba2rgb | uint32 | fortran | 1 | 14.7587 | 2.2754 | 6.49x |
| color.rgba2rgb | uint32 | planar | 1 | 9.9386 | 2.2305 | 4.46x |
| color.rgba2rgb | int32 | contiguous | 1 | 19.4822 | 2.7180 | 7.17x |
| color.rgba2rgb | int32 | reversed | 1 | 17.0136 | 2.5057 | 6.79x |
| color.rgba2rgb | int32 | strided | 1 | 18.0083 | 2.6541 | 6.79x |
| color.rgba2rgb | int32 | fortran | 1 | 14.0282 | 2.4444 | 5.74x |
| color.rgba2rgb | int32 | planar | 1 | 13.2804 | 2.7588 | 4.81x |
| color.rgba2rgb | uint64 | contiguous | 1 | 21.7558 | 5.5046 | 3.95x |
| color.rgba2rgb | uint64 | reversed | 1 | 23.2783 | 5.1678 | 4.50x |
| color.rgba2rgb | uint64 | strided | 1 | 26.1550 | 7.5665 | 3.46x |
| color.rgba2rgb | uint64 | fortran | 1 | 20.4561 | 5.7343 | 3.57x |
| color.rgba2rgb | uint64 | planar | 1 | 16.1435 | 7.8032 | 2.07x |
| color.rgba2rgb | int64 | contiguous | 1 | 19.4444 | 3.2803 | 5.93x |
| color.rgba2rgb | int64 | reversed | 1 | 24.6226 | 3.6917 | 6.67x |
| color.rgba2rgb | int64 | strided | 1 | 26.9350 | 4.9341 | 5.46x |
| color.rgba2rgb | int64 | fortran | 1 | 20.6939 | 3.2919 | 6.29x |
| color.rgba2rgb | int64 | planar | 1 | 12.8409 | 3.7713 | 3.40x |
| color.rgba2rgb | float32 | contiguous | 1 | 13.0481 | 1.5002 | 8.70x |
| color.rgba2rgb | float32 | reversed | 1 | 12.1610 | 3.0327 | 4.01x |
| color.rgba2rgb | float32 | strided | 1 | 12.8339 | 3.4571 | 3.71x |
| color.rgba2rgb | float32 | fortran | 1 | 7.9861 | 1.2583 | 6.35x |
| color.rgba2rgb | float32 | planar | 1 | 4.7362 | 0.8579 | 5.52x |
| color.rgba2rgb | float64 | contiguous | 1 | 17.1360 | 3.2144 | 5.33x |
| color.rgba2rgb | float64 | reversed | 1 | 16.5678 | 7.2969 | 2.27x |
| color.rgba2rgb | float64 | strided | 1 | 18.6823 | 8.9125 | 2.10x |
| color.rgba2rgb | float64 | fortran | 1 | 13.4131 | 2.2336 | 6.01x |
| color.rgba2rgb | float64 | planar | 1 | 10.2989 | 2.3101 | 4.46x |
| color.gray2rgb | bool | contiguous | 1 | 0.3768 | 0.1374 | 2.74x |
| color.gray2rgba | bool | contiguous | 1 | 0.3617 | 0.1828 | 1.98x |
| color.gray2rgb | bool | reversed | 1 | 0.4049 | 0.2421 | 1.67x |
| color.gray2rgba | bool | reversed | 1 | 0.3768 | 0.1367 | 2.76x |
| color.gray2rgb | bool | strided | 1 | 0.5553 | 0.3475 | 1.60x |
| color.gray2rgba | bool | strided | 1 | 0.3683 | 0.1224 | 3.01x |
| color.gray2rgb | bool | fortran | 1 | 0.5002 | 0.3108 | 1.61x |
| color.gray2rgba | bool | fortran | 1 | 0.6513 | 0.1413 | 4.61x |
| color.gray2rgb | uint8 | contiguous | 1 | 0.4820 | 0.2737 | 1.76x |
| color.gray2rgba | uint8 | contiguous | 1 | 0.4532 | 0.1052 | 4.31x |
| color.gray2rgb | uint8 | reversed | 1 | 0.4425 | 0.3180 | 1.39x |
| color.gray2rgba | uint8 | reversed | 1 | 0.3147 | 0.1615 | 1.95x |
| color.gray2rgb | uint8 | strided | 1 | 0.5016 | 0.3426 | 1.46x |
| color.gray2rgba | uint8 | strided | 1 | 0.4930 | 0.1335 | 3.69x |
| color.gray2rgb | uint8 | fortran | 1 | 0.5611 | 0.3161 | 1.78x |
| color.gray2rgba | uint8 | fortran | 1 | 0.7495 | 0.1810 | 4.14x |
| color.gray2rgb | int8 | contiguous | 1 | 0.4988 | 0.3061 | 1.63x |
| color.gray2rgba | int8 | contiguous | 1 | 0.4556 | 0.1359 | 3.35x |
| color.gray2rgb | int8 | reversed | 1 | 0.5291 | 0.3346 | 1.58x |
| color.gray2rgba | int8 | reversed | 1 | 0.4249 | 0.1571 | 2.70x |
| color.gray2rgb | int8 | strided | 1 | 0.5766 | 0.3789 | 1.52x |
| color.gray2rgba | int8 | strided | 1 | 0.5868 | 0.2140 | 2.74x |
| color.gray2rgb | int8 | fortran | 1 | 0.5969 | 0.3444 | 1.73x |
| color.gray2rgba | int8 | fortran | 1 | 0.8010 | 0.1685 | 4.75x |
| color.gray2rgb | uint16 | contiguous | 1 | 0.3547 | 0.2773 | 1.28x |
| color.gray2rgba | uint16 | contiguous | 1 | 0.6338 | 0.3406 | 1.86x |
| color.gray2rgb | uint16 | reversed | 1 | 1.0545 | 0.6959 | 1.52x |
| color.gray2rgba | uint16 | reversed | 1 | 0.5163 | 0.3529 | 1.46x |
| color.gray2rgb | uint16 | strided | 1 | 1.2207 | 0.9608 | 1.27x |
| color.gray2rgba | uint16 | strided | 1 | 1.2722 | 0.7403 | 1.72x |
| color.gray2rgb | uint16 | fortran | 1 | 0.5969 | 0.3544 | 1.68x |
| color.gray2rgba | uint16 | fortran | 1 | 0.8554 | 0.1805 | 4.74x |
| color.gray2rgb | int16 | contiguous | 1 | 0.3292 | 0.2074 | 1.59x |
| color.gray2rgba | int16 | contiguous | 1 | 0.4778 | 0.3288 | 1.45x |
| color.gray2rgb | int16 | reversed | 1 | 0.2775 | 0.2515 | 1.10x |
| color.gray2rgba | int16 | reversed | 1 | 0.5105 | 0.2345 | 2.18x |
| color.gray2rgb | int16 | strided | 1 | 0.3570 | 0.3200 | 1.12x |
| color.gray2rgba | int16 | strided | 1 | 0.7850 | 0.4198 | 1.87x |
| color.gray2rgb | int16 | fortran | 1 | 0.3317 | 0.2415 | 1.37x |
| color.gray2rgba | int16 | fortran | 1 | 0.8609 | 0.2330 | 3.70x |
| color.gray2rgb | uint32 | contiguous | 1 | 1.6850 | 1.1414 | 1.48x |
| color.gray2rgba | uint32 | contiguous | 1 | 0.7241 | 0.3380 | 2.14x |
| color.gray2rgb | uint32 | reversed | 1 | 1.5647 | 1.1524 | 1.36x |
| color.gray2rgba | uint32 | reversed | 1 | 0.8775 | 0.4265 | 2.06x |
| color.gray2rgb | uint32 | strided | 1 | 2.8602 | 1.8607 | 1.54x |
| color.gray2rgba | uint32 | strided | 1 | 2.1500 | 1.0355 | 2.08x |
| color.gray2rgb | uint32 | fortran | 1 | 1.5565 | 0.6524 | 2.39x |
| color.gray2rgba | uint32 | fortran | 1 | 1.5813 | 0.4485 | 3.53x |
| color.gray2rgb | int32 | contiguous | 1 | 0.6542 | 0.2440 | 2.68x |
| color.gray2rgba | int32 | contiguous | 1 | 0.9978 | 0.3748 | 2.66x |
| color.gray2rgb | int32 | reversed | 1 | 0.5790 | 0.2577 | 2.25x |
| color.gray2rgba | int32 | reversed | 1 | 0.9032 | 0.4814 | 1.88x |
| color.gray2rgb | int32 | strided | 1 | 0.6819 | 0.3721 | 1.83x |
| color.gray2rgba | int32 | strided | 1 | 0.9434 | 0.4220 | 2.24x |
| color.gray2rgb | int32 | fortran | 1 | 0.4317 | 0.2246 | 1.92x |
| color.gray2rgba | int32 | fortran | 1 | 1.8484 | 0.3471 | 5.32x |
| color.gray2rgb | uint64 | contiguous | 1 | 2.8531 | 2.0021 | 1.43x |
| color.gray2rgba | uint64 | contiguous | 1 | 6.5043 | 3.6071 | 1.80x |
| color.gray2rgb | uint64 | reversed | 1 | 1.1014 | 0.6782 | 1.62x |
| color.gray2rgba | uint64 | reversed | 1 | 5.2319 | 3.2541 | 1.61x |
| color.gray2rgb | uint64 | strided | 1 | 2.5125 | 1.2118 | 2.07x |
| color.gray2rgba | uint64 | strided | 1 | 9.6091 | 3.9492 | 2.43x |
| color.gray2rgb | uint64 | fortran | 1 | 3.0238 | 2.1915 | 1.38x |
| color.gray2rgba | uint64 | fortran | 1 | 7.5736 | 3.1037 | 2.44x |
| color.gray2rgb | int64 | contiguous | 1 | 3.1701 | 2.2969 | 1.38x |
| color.gray2rgba | int64 | contiguous | 1 | 7.0941 | 3.1475 | 2.25x |
| color.gray2rgb | int64 | reversed | 1 | 3.5479 | 2.4640 | 1.44x |
| color.gray2rgba | int64 | reversed | 1 | 4.7451 | 2.8649 | 1.66x |
| color.gray2rgb | int64 | strided | 1 | 3.3363 | 1.5196 | 2.20x |
| color.gray2rgba | int64 | strided | 1 | 5.4257 | 3.5105 | 1.55x |
| color.gray2rgb | int64 | fortran | 1 | 4.3604 | 1.4259 | 3.06x |
| color.gray2rgba | int64 | fortran | 1 | 8.2826 | 3.3060 | 2.51x |
| color.gray2rgb | float32 | contiguous | 1 | 0.5544 | 0.2022 | 2.74x |
| color.gray2rgba | float32 | contiguous | 1 | 0.7855 | 0.3509 | 2.24x |
| color.gray2rgb | float32 | reversed | 1 | 0.3816 | 0.2185 | 1.75x |
| color.gray2rgba | float32 | reversed | 1 | 0.8321 | 0.4723 | 1.76x |
| color.gray2rgb | float32 | strided | 1 | 0.5898 | 0.3253 | 1.81x |
| color.gray2rgba | float32 | strided | 1 | 0.9954 | 0.4362 | 2.28x |
| color.gray2rgb | float32 | fortran | 1 | 0.4045 | 0.2395 | 1.69x |
| color.gray2rgba | float32 | fortran | 1 | 1.3583 | 0.4630 | 2.93x |
| color.gray2rgb | float64 | contiguous | 1 | 2.1319 | 0.5516 | 3.86x |
| color.gray2rgba | float64 | contiguous | 1 | 4.0860 | 2.5625 | 1.59x |
| color.gray2rgb | float64 | reversed | 1 | 1.1498 | 0.4611 | 2.49x |
| color.gray2rgba | float64 | reversed | 1 | 4.8511 | 2.9274 | 1.66x |
| color.gray2rgb | float64 | strided | 1 | 1.7627 | 0.8326 | 2.12x |
| color.gray2rgba | float64 | strided | 1 | 6.1367 | 3.1473 | 1.95x |
| color.gray2rgb | float64 | fortran | 1 | 0.9822 | 0.4825 | 2.04x |
| color.gray2rgba | float64 | fortran | 1 | 8.7470 | 3.1448 | 2.78x |
| color.gray2rgb | float16 | contiguous | 1 | 0.3545 | 0.2317 | 1.53x |
| color.gray2rgba | float16 | contiguous | 1 | 1.4195 | 1.1394 | 1.25x |
| color.gray2rgb | float16 | reversed | 1 | 0.3265 | 0.1651 | 1.98x |
| color.gray2rgba | float16 | reversed | 1 | 1.4614 | 1.0009 | 1.46x |
| color.gray2rgb | float16 | strided | 1 | 0.4548 | 0.3238 | 1.40x |
| color.gray2rgba | float16 | strided | 1 | 1.0759 | 0.9517 | 1.13x |
| color.gray2rgb | float16 | fortran | 1 | 0.3189 | 0.2351 | 1.36x |
| color.gray2rgba | float16 | fortran | 1 | 1.2292 | 0.8069 | 1.52x |
| color.rgba2rgb | uint8 | background-black | 1 | 20.6210 | 0.8934 | 23.08x |
| color.rgba2rgb | uint8 | background-color | 1 | 20.5301 | 0.8160 | 25.16x |
| color.rgba2rgb | uint8 | background-nan | 1 | 25.4268 | 1.7838 | 14.25x |
| color.gray2rgba | uint8 | alpha-scalar | 1 | 0.6075 | 0.2259 | 2.69x |
| color.gray2rgba | uint8 | alpha-typed-scalar | 1 | 0.4383 | 0.1230 | 3.56x |
| color.gray2rgba | uint8 | alpha-array | 1 | 0.4456 | 0.1210 | 3.68x |
| color.gray2rgba | uint8 | alpha-broadcast | 1 | 0.5627 | 0.1225 | 4.59x |
| color.rgba2rgb | float32 | background-black | 1 | 12.0443 | 0.9756 | 12.35x |
| color.rgba2rgb | float32 | background-color | 1 | 11.5659 | 0.8654 | 13.36x |
| color.rgba2rgb | float32 | background-nan | 1 | 13.2430 | 1.4680 | 9.02x |
| color.gray2rgba | float32 | alpha-scalar | 1 | 0.8937 | 0.4018 | 2.22x |
| color.gray2rgba | float32 | alpha-typed-scalar | 1 | 0.6725 | 0.3390 | 1.98x |
| color.gray2rgba | float32 | alpha-array | 1 | 1.4188 | 1.0012 | 1.42x |
| color.gray2rgba | float32 | alpha-broadcast | 1 | 0.7520 | 0.3409 | 2.21x |
| color.rgba2rgb | float64 | background-black | 1 | 16.7402 | 2.5814 | 6.48x |
| color.rgba2rgb | float64 | background-color | 1 | 19.1910 | 3.3102 | 5.80x |
| color.rgba2rgb | float64 | background-nan | 1 | 18.5982 | 2.8119 | 6.61x |
| color.gray2rgba | float64 | alpha-scalar | 1 | 5.2939 | 3.2099 | 1.65x |
| color.gray2rgba | float64 | alpha-typed-scalar | 1 | 4.8840 | 3.1319 | 1.56x |
| color.gray2rgba | float64 | alpha-array | 1 | 6.5530 | 3.6890 | 1.78x |
| color.gray2rgba | float64 | alpha-broadcast | 1 | 4.0727 | 2.6691 | 1.53x |
| color.gray2rgb | >u2 | endian-contiguous | 1 | 0.3126 | 0.1641 | 1.90x |
| color.gray2rgba | >u2 | endian-contiguous | 1 | 0.5176 | 0.2696 | 1.92x |
| color.gray2rgb | >u2 | endian-reversed | 1 | 0.3997 | 0.2477 | 1.61x |
| color.gray2rgba | >u2 | endian-reversed | 1 | 0.5504 | 0.2803 | 1.96x |
| color.gray2rgb | >f4 | endian-contiguous | 1 | 0.4622 | 0.3465 | 1.33x |
| color.gray2rgba | >f4 | endian-contiguous | 1 | 2.1614 | 1.3839 | 1.56x |
| color.gray2rgb | >f4 | endian-reversed | 1 | 2.5856 | 0.6819 | 3.79x |
| color.gray2rgba | >f4 | endian-reversed | 1 | 2.8458 | 1.5059 | 1.89x |
| color.gray2rgb | >f8 | endian-contiguous | 1 | 5.5414 | 3.1778 | 1.74x |
| color.gray2rgba | >f8 | endian-contiguous | 1 | 20.7076 | 13.5358 | 1.53x |
| color.gray2rgb | >f8 | endian-reversed | 1 | 8.2376 | 2.5730 | 3.20x |
| color.gray2rgba | >f8 | endian-reversed | 1 | 14.3919 | 9.7251 | 1.48x |
| color.gray2rgb | [('x', '>i2'), ('y', '>f4')] | record-6-contiguous | 1 | 6.6412 | 1.8462 | 3.60x |
| color.gray2rgba | [('x', '>i2'), ('y', '>f4')] | record-6-contiguous | 1 | 14.1275 | 5.9227 | 2.39x |
| color.gray2rgb | [('x', '>i2'), ('y', '>f4')] | record-6-reversed | 1 | 4.6385 | 1.8611 | 2.49x |
| color.gray2rgba | [('x', '>i2'), ('y', '>f4')] | record-6-reversed | 1 | 25.7155 | 6.4608 | 3.98x |
| color.gray2rgb | [('x', '>i2'), ('y', '>f4')] | record-6-strided | 1 | 6.1896 | 2.3666 | 2.62x |
| color.gray2rgba | [('x', '>i2'), ('y', '>f4')] | record-6-strided | 1 | 25.0516 | 8.0999 | 3.09x |
| color.gray2rgb | [('x', '>i2'), ('y', '>f4')] | record-6-fortran | 1 | 5.3191 | 0.9221 | 5.77x |
| color.gray2rgba | [('x', '>i2'), ('y', '>f4')] | record-6-fortran | 1 | 16.9868 | 4.1757 | 4.07x |
| color.gray2rgb | [('record', [('a', '>i2'), ('b', '>u4')]), ('vec', '>f4', (3,))] | record-18-contiguous | 1 | 28.4699 | 15.0873 | 1.89x |
| color.gray2rgba | [('record', [('a', '>i2'), ('b', '>u4')]), ('vec', '>f4', (3,))] | record-18-contiguous | 1 | 44.1999 | 23.6969 | 1.87x |
| color.gray2rgb | [('record', [('a', '>i2'), ('b', '>u4')]), ('vec', '>f4', (3,))] | record-18-reversed | 1 | 15.7851 | 9.3342 | 1.69x |
| color.gray2rgba | [('record', [('a', '>i2'), ('b', '>u4')]), ('vec', '>f4', (3,))] | record-18-reversed | 1 | 38.6999 | 20.4504 | 1.89x |
| color.gray2rgb | [('record', [('a', '>i2'), ('b', '>u4')]), ('vec', '>f4', (3,))] | record-18-strided | 1 | 19.5620 | 10.3339 | 1.89x |
| color.gray2rgba | [('record', [('a', '>i2'), ('b', '>u4')]), ('vec', '>f4', (3,))] | record-18-strided | 1 | 32.8966 | 20.4497 | 1.61x |
| color.gray2rgb | [('record', [('a', '>i2'), ('b', '>u4')]), ('vec', '>f4', (3,))] | record-18-fortran | 1 | 14.2744 | 8.2083 | 1.74x |
| color.gray2rgba | [('record', [('a', '>i2'), ('b', '>u4')]), ('vec', '>f4', (3,))] | record-18-fortran | 1 | 33.6411 | 20.9993 | 1.60x |
| color.gray2rgb | [('z', '>c16'), ('when', '>M8[ns]'), ('dt', '>m8[ms]')] | record-32-contiguous | 1 | 21.7158 | 8.8863 | 2.44x |
| color.gray2rgba | [('z', '>c16'), ('when', '>M8[ns]'), ('dt', '>m8[ms]')] | record-32-contiguous | 1 | 38.6454 | 19.7150 | 1.96x |
| color.gray2rgb | [('z', '>c16'), ('when', '>M8[ns]'), ('dt', '>m8[ms]')] | record-32-reversed | 1 | 21.4059 | 10.0627 | 2.13x |
| color.gray2rgba | [('z', '>c16'), ('when', '>M8[ns]'), ('dt', '>m8[ms]')] | record-32-reversed | 1 | 41.6984 | 21.4401 | 1.94x |
| color.gray2rgb | [('z', '>c16'), ('when', '>M8[ns]'), ('dt', '>m8[ms]')] | record-32-strided | 1 | 22.9554 | 11.1648 | 2.06x |
| color.gray2rgba | [('z', '>c16'), ('when', '>M8[ns]'), ('dt', '>m8[ms]')] | record-32-strided | 1 | 60.5665 | 30.0063 | 2.02x |
| color.gray2rgb | [('z', '>c16'), ('when', '>M8[ns]'), ('dt', '>m8[ms]')] | record-32-fortran | 1 | 24.0338 | 10.4805 | 2.29x |
| color.gray2rgba | [('z', '>c16'), ('when', '>M8[ns]'), ('dt', '>m8[ms]')] | record-32-fortran | 1 | 43.7934 | 19.8508 | 2.21x |
