# Alpha and grayscale performance

Original scikit-image 0.26.0. Reference thread pools use one worker; native workers are explicit. Timings include dispatch, integer normalization, validation, layout handling and output allocation. Input construction is excluded. All samples and slower cases are retained in the JSON.

| API | Input | Layout / workload | Workers | Original ms | C++ ms | Speedup |
|---|---|---|---:|---:|---:|---:|
| color.rgba2rgb | bool | contiguous | 1 | 14.4722 | 0.6212 | 23.30x |
| color.rgba2rgb | bool | reversed | 1 | 17.2754 | 2.1194 | 8.15x |
| color.rgba2rgb | bool | strided | 1 | 14.6489 | 2.1765 | 6.73x |
| color.rgba2rgb | bool | fortran | 1 | 11.6201 | 2.0616 | 5.64x |
| color.rgba2rgb | bool | planar | 1 | 8.2516 | 2.1351 | 3.86x |
| color.rgba2rgb | uint8 | contiguous | 1 | 21.0831 | 0.7920 | 26.62x |
| color.rgba2rgb | uint8 | reversed | 1 | 19.6606 | 2.3101 | 8.51x |
| color.rgba2rgb | uint8 | strided | 1 | 18.3562 | 2.2960 | 7.99x |
| color.rgba2rgb | uint8 | fortran | 1 | 13.9829 | 2.1245 | 6.58x |
| color.rgba2rgb | uint8 | planar | 1 | 12.0577 | 2.0856 | 5.78x |
| color.rgba2rgb | int8 | contiguous | 1 | 20.0300 | 2.1522 | 9.31x |
| color.rgba2rgb | int8 | reversed | 1 | 21.9652 | 3.6227 | 6.06x |
| color.rgba2rgb | int8 | strided | 1 | 16.7358 | 2.5550 | 6.55x |
| color.rgba2rgb | int8 | fortran | 1 | 14.2197 | 3.3118 | 4.29x |
| color.rgba2rgb | int8 | planar | 1 | 12.7604 | 2.6124 | 4.88x |
| color.rgba2rgb | uint16 | contiguous | 1 | 20.1789 | 1.0043 | 20.09x |
| color.rgba2rgb | uint16 | reversed | 1 | 19.9366 | 2.4252 | 8.22x |
| color.rgba2rgb | uint16 | strided | 1 | 21.6601 | 2.6846 | 8.07x |
| color.rgba2rgb | uint16 | fortran | 1 | 16.1057 | 2.2411 | 7.19x |
| color.rgba2rgb | uint16 | planar | 1 | 13.6608 | 2.6095 | 5.24x |
| color.rgba2rgb | int16 | contiguous | 1 | 23.0162 | 2.3101 | 9.96x |
| color.rgba2rgb | int16 | reversed | 1 | 22.0724 | 2.7854 | 7.92x |
| color.rgba2rgb | int16 | strided | 1 | 20.4727 | 2.6426 | 7.75x |
| color.rgba2rgb | int16 | fortran | 1 | 16.7206 | 2.6316 | 6.35x |
| color.rgba2rgb | int16 | planar | 1 | 14.8706 | 2.8030 | 5.31x |
| color.rgba2rgb | uint32 | contiguous | 1 | 20.0381 | 2.2488 | 8.91x |
| color.rgba2rgb | uint32 | reversed | 1 | 19.1693 | 2.5094 | 7.64x |
| color.rgba2rgb | uint32 | strided | 1 | 20.0299 | 2.6911 | 7.44x |
| color.rgba2rgb | uint32 | fortran | 1 | 16.5870 | 2.5067 | 6.62x |
| color.rgba2rgb | uint32 | planar | 1 | 13.2684 | 2.4591 | 5.40x |
| color.rgba2rgb | int32 | contiguous | 1 | 21.1692 | 2.8129 | 7.53x |
| color.rgba2rgb | int32 | reversed | 1 | 19.2577 | 2.7813 | 6.92x |
| color.rgba2rgb | int32 | strided | 1 | 18.9293 | 3.0368 | 6.23x |
| color.rgba2rgb | int32 | fortran | 1 | 15.8645 | 2.5774 | 6.16x |
| color.rgba2rgb | int32 | planar | 1 | 13.5026 | 2.6277 | 5.14x |
| color.rgba2rgb | uint64 | contiguous | 1 | 23.8370 | 5.1783 | 4.60x |
| color.rgba2rgb | uint64 | reversed | 1 | 24.6412 | 5.1956 | 4.74x |
| color.rgba2rgb | uint64 | strided | 1 | 25.6776 | 5.5501 | 4.63x |
| color.rgba2rgb | uint64 | fortran | 1 | 18.6882 | 4.9207 | 3.80x |
| color.rgba2rgb | uint64 | planar | 1 | 19.3079 | 5.0329 | 3.84x |
| color.rgba2rgb | int64 | contiguous | 1 | 18.0167 | 2.7296 | 6.60x |
| color.rgba2rgb | int64 | reversed | 1 | 20.3693 | 3.5905 | 5.67x |
| color.rgba2rgb | int64 | strided | 1 | 22.1823 | 4.2937 | 5.17x |
| color.rgba2rgb | int64 | fortran | 1 | 18.5463 | 3.1663 | 5.86x |
| color.rgba2rgb | int64 | planar | 1 | 13.4393 | 2.6733 | 5.03x |
| color.rgba2rgb | float32 | contiguous | 1 | 10.9040 | 0.7998 | 13.63x |
| color.rgba2rgb | float32 | reversed | 1 | 9.9289 | 2.8382 | 3.50x |
| color.rgba2rgb | float32 | strided | 1 | 11.4090 | 4.1647 | 2.74x |
| color.rgba2rgb | float32 | fortran | 1 | 6.2581 | 1.0519 | 5.95x |
| color.rgba2rgb | float32 | planar | 1 | 6.7980 | 1.5177 | 4.48x |
| color.rgba2rgb | float64 | contiguous | 1 | 16.9166 | 3.1141 | 5.43x |
| color.rgba2rgb | float64 | reversed | 1 | 15.7085 | 6.7422 | 2.33x |
| color.rgba2rgb | float64 | strided | 1 | 21.8596 | 12.7651 | 1.71x |
| color.rgba2rgb | float64 | fortran | 1 | 15.3317 | 3.9163 | 3.91x |
| color.rgba2rgb | float64 | planar | 1 | 21.5013 | 5.5475 | 3.88x |
| color.gray2rgb | bool | contiguous | 1 | 0.4557 | 0.3064 | 1.49x |
| color.gray2rgba | bool | contiguous | 1 | 0.6008 | 0.3383 | 1.78x |
| color.gray2rgb | bool | reversed | 1 | 0.6556 | 1.0524 | 0.62x |
| color.gray2rgba | bool | reversed | 1 | 0.3089 | 0.0739 | 4.18x |
| color.gray2rgb | bool | strided | 1 | 0.4383 | 0.2848 | 1.54x |
| color.gray2rgba | bool | strided | 1 | 0.3085 | 0.1169 | 2.64x |
| color.gray2rgb | bool | fortran | 1 | 0.4108 | 0.3250 | 1.26x |
| color.gray2rgba | bool | fortran | 1 | 0.5316 | 0.0725 | 7.33x |
| color.gray2rgb | uint8 | contiguous | 1 | 0.6631 | 0.4734 | 1.40x |
| color.gray2rgba | uint8 | contiguous | 1 | 0.5005 | 0.2653 | 1.89x |
| color.gray2rgb | uint8 | reversed | 1 | 0.7936 | 0.5156 | 1.54x |
| color.gray2rgba | uint8 | reversed | 1 | 0.4979 | 0.2550 | 1.95x |
| color.gray2rgb | uint8 | strided | 1 | 0.8312 | 0.6232 | 1.33x |
| color.gray2rgba | uint8 | strided | 1 | 0.6385 | 0.2907 | 2.20x |
| color.gray2rgb | uint8 | fortran | 1 | 0.7476 | 0.5067 | 1.48x |
| color.gray2rgba | uint8 | fortran | 1 | 0.9779 | 0.2110 | 4.63x |
| color.gray2rgb | int8 | contiguous | 1 | 0.8808 | 0.7768 | 1.13x |
| color.gray2rgba | int8 | contiguous | 1 | 0.5619 | 0.2235 | 2.51x |
| color.gray2rgb | int8 | reversed | 1 | 1.0236 | 0.6318 | 1.62x |
| color.gray2rgba | int8 | reversed | 1 | 0.7968 | 0.3853 | 2.07x |
| color.gray2rgb | int8 | strided | 1 | 1.0572 | 0.8085 | 1.31x |
| color.gray2rgba | int8 | strided | 1 | 0.8861 | 0.5537 | 1.60x |
| color.gray2rgb | int8 | fortran | 1 | 0.5205 | 0.2144 | 2.43x |
| color.gray2rgba | int8 | fortran | 1 | 1.3461 | 0.5358 | 2.51x |
| color.gray2rgb | uint16 | contiguous | 1 | 0.6123 | 0.3067 | 2.00x |
| color.gray2rgba | uint16 | contiguous | 1 | 3.1256 | 0.8489 | 3.68x |
| color.gray2rgb | uint16 | reversed | 1 | 0.9354 | 0.4535 | 2.06x |
| color.gray2rgba | uint16 | reversed | 1 | 0.7259 | 0.7220 | 1.01x |
| color.gray2rgb | uint16 | strided | 1 | 0.7614 | 0.9355 | 0.81x |
| color.gray2rgba | uint16 | strided | 1 | 0.9245 | 0.5466 | 1.69x |
| color.gray2rgb | uint16 | fortran | 1 | 0.4144 | 0.3016 | 1.37x |
| color.gray2rgba | uint16 | fortran | 1 | 1.0596 | 0.3215 | 3.30x |
| color.gray2rgb | int16 | contiguous | 1 | 0.4588 | 0.2945 | 1.56x |
| color.gray2rgba | int16 | contiguous | 1 | 0.5414 | 0.3242 | 1.67x |
| color.gray2rgb | int16 | reversed | 1 | 0.3592 | 0.2890 | 1.24x |
| color.gray2rgba | int16 | reversed | 1 | 0.6103 | 0.3310 | 1.84x |
| color.gray2rgb | int16 | strided | 1 | 0.5226 | 0.3868 | 1.35x |
| color.gray2rgba | int16 | strided | 1 | 1.0048 | 0.4718 | 2.13x |
| color.gray2rgb | int16 | fortran | 1 | 0.3427 | 0.2509 | 1.37x |
| color.gray2rgba | int16 | fortran | 1 | 0.6816 | 0.1999 | 3.41x |
| color.gray2rgb | uint32 | contiguous | 1 | 1.1769 | 1.1232 | 1.05x |
| color.gray2rgba | uint32 | contiguous | 1 | 0.6309 | 0.2435 | 2.59x |
| color.gray2rgb | uint32 | reversed | 1 | 1.5714 | 1.0505 | 1.50x |
| color.gray2rgba | uint32 | reversed | 1 | 0.5911 | 0.2826 | 2.09x |
| color.gray2rgb | uint32 | strided | 1 | 1.5981 | 1.1561 | 1.38x |
| color.gray2rgba | uint32 | strided | 1 | 0.8835 | 0.3520 | 2.51x |
| color.gray2rgb | uint32 | fortran | 1 | 1.5436 | 1.3271 | 1.16x |
| color.gray2rgba | uint32 | fortran | 1 | 1.0969 | 0.4488 | 2.44x |
| color.gray2rgb | int32 | contiguous | 1 | 0.3826 | 0.2009 | 1.90x |
| color.gray2rgba | int32 | contiguous | 1 | 0.5533 | 0.2593 | 2.13x |
| color.gray2rgb | int32 | reversed | 1 | 0.3001 | 0.1849 | 1.62x |
| color.gray2rgba | int32 | reversed | 1 | 0.5090 | 0.2395 | 2.13x |
| color.gray2rgb | int32 | strided | 1 | 0.4016 | 0.2158 | 1.86x |
| color.gray2rgba | int32 | strided | 1 | 0.7143 | 0.3155 | 2.26x |
| color.gray2rgb | int32 | fortran | 1 | 0.3001 | 0.1639 | 1.83x |
| color.gray2rgba | int32 | fortran | 1 | 0.9065 | 0.2341 | 3.87x |
| color.gray2rgb | uint64 | contiguous | 1 | 0.8593 | 0.4944 | 1.74x |
| color.gray2rgba | uint64 | contiguous | 1 | 3.8344 | 2.6373 | 1.45x |
| color.gray2rgb | uint64 | reversed | 1 | 0.7115 | 0.4357 | 1.63x |
| color.gray2rgba | uint64 | reversed | 1 | 4.1245 | 2.3589 | 1.75x |
| color.gray2rgb | uint64 | strided | 1 | 1.1961 | 0.5839 | 2.05x |
| color.gray2rgba | uint64 | strided | 1 | 4.0469 | 2.7944 | 1.45x |
| color.gray2rgb | uint64 | fortran | 1 | 0.6698 | 0.4542 | 1.47x |
| color.gray2rgba | uint64 | fortran | 1 | 5.4389 | 2.4871 | 2.19x |
| color.gray2rgb | int64 | contiguous | 1 | 0.9199 | 0.3454 | 2.66x |
| color.gray2rgba | int64 | contiguous | 1 | 3.7112 | 2.3425 | 1.58x |
| color.gray2rgb | int64 | reversed | 1 | 1.0470 | 0.3941 | 2.66x |
| color.gray2rgba | int64 | reversed | 1 | 3.8666 | 2.6211 | 1.48x |
| color.gray2rgb | int64 | strided | 1 | 3.0954 | 2.0624 | 1.50x |
| color.gray2rgba | int64 | strided | 1 | 5.4102 | 3.4605 | 1.56x |
| color.gray2rgb | int64 | fortran | 1 | 2.3220 | 1.8669 | 1.24x |
| color.gray2rgba | int64 | fortran | 1 | 5.5050 | 2.3797 | 2.31x |
| color.gray2rgb | float32 | contiguous | 1 | 0.4551 | 0.2020 | 2.25x |
| color.gray2rgba | float32 | contiguous | 1 | 0.5954 | 0.3203 | 1.86x |
| color.gray2rgb | float32 | reversed | 1 | 0.3095 | 0.1584 | 1.95x |
| color.gray2rgba | float32 | reversed | 1 | 0.6080 | 0.2518 | 2.41x |
| color.gray2rgb | float32 | strided | 1 | 0.4428 | 0.2844 | 1.56x |
| color.gray2rgba | float32 | strided | 1 | 0.9267 | 0.3444 | 2.69x |
| color.gray2rgb | float32 | fortran | 1 | 0.5339 | 0.2362 | 2.26x |
| color.gray2rgba | float32 | fortran | 1 | 1.2763 | 0.4924 | 2.59x |
| color.gray2rgb | float64 | contiguous | 1 | 6.1212 | 1.9560 | 3.13x |
| color.gray2rgba | float64 | contiguous | 1 | 7.7444 | 3.4223 | 2.26x |
| color.gray2rgb | float64 | reversed | 1 | 2.4515 | 1.0381 | 2.36x |
| color.gray2rgba | float64 | reversed | 1 | 11.0884 | 4.9294 | 2.25x |
| color.gray2rgb | float64 | strided | 1 | 7.6100 | 4.1596 | 1.83x |
| color.gray2rgba | float64 | strided | 1 | 13.9056 | 5.2130 | 2.67x |
| color.gray2rgb | float64 | fortran | 1 | 7.1637 | 3.5689 | 2.01x |
| color.gray2rgba | float64 | fortran | 1 | 16.2496 | 5.0372 | 3.23x |
| color.gray2rgb | float16 | contiguous | 1 | 0.2079 | 0.1744 | 1.19x |
| color.gray2rgba | float16 | contiguous | 1 | 1.2514 | 0.8748 | 1.43x |
| color.gray2rgb | float16 | reversed | 1 | 0.7427 | 0.5879 | 1.26x |
| color.gray2rgba | float16 | reversed | 1 | 1.0128 | 0.7758 | 1.31x |
| color.gray2rgb | float16 | strided | 1 | 0.6673 | 0.5347 | 1.25x |
| color.gray2rgba | float16 | strided | 1 | 1.0641 | 0.8666 | 1.23x |
| color.gray2rgb | float16 | fortran | 1 | 0.5954 | 0.5335 | 1.12x |
| color.gray2rgba | float16 | fortran | 1 | 1.0771 | 0.6361 | 1.69x |
| color.rgba2rgb | uint8 | background-black | 1 | 16.6895 | 0.6166 | 27.07x |
| color.rgba2rgb | uint8 | background-color | 1 | 16.1761 | 0.6455 | 25.06x |
| color.rgba2rgb | uint8 | background-nan | 1 | 16.3421 | 0.6211 | 26.31x |
| color.gray2rgba | uint8 | alpha-scalar | 1 | 0.2651 | 0.0664 | 3.99x |
| color.gray2rgba | uint8 | alpha-typed-scalar | 1 | 0.2688 | 0.0640 | 4.20x |
| color.gray2rgba | uint8 | alpha-array | 1 | 0.2801 | 0.0614 | 4.56x |
| color.gray2rgba | uint8 | alpha-broadcast | 1 | 0.2919 | 0.0619 | 4.71x |
| color.rgba2rgb | float32 | background-black | 1 | 9.4191 | 0.7048 | 13.36x |
| color.rgba2rgb | float32 | background-color | 1 | 9.8289 | 0.8652 | 11.36x |
| color.rgba2rgb | float32 | background-nan | 1 | 9.1097 | 0.6290 | 14.48x |
| color.gray2rgba | float32 | alpha-scalar | 1 | 0.5330 | 0.2582 | 2.06x |
| color.gray2rgba | float32 | alpha-typed-scalar | 1 | 0.5554 | 0.2368 | 2.35x |
| color.gray2rgba | float32 | alpha-array | 1 | 0.7519 | 0.6422 | 1.17x |
| color.gray2rgba | float32 | alpha-broadcast | 1 | 0.6301 | 0.2151 | 2.93x |
| color.rgba2rgb | float64 | background-black | 1 | 13.7036 | 1.8502 | 7.41x |
| color.rgba2rgb | float64 | background-color | 1 | 14.8387 | 2.6110 | 5.68x |
| color.rgba2rgb | float64 | background-nan | 1 | 13.4278 | 1.3592 | 9.88x |
| color.gray2rgba | float64 | alpha-scalar | 1 | 4.8823 | 2.3406 | 2.09x |
| color.gray2rgba | float64 | alpha-typed-scalar | 1 | 4.0858 | 3.0281 | 1.35x |
| color.gray2rgba | float64 | alpha-array | 1 | 4.7345 | 2.7496 | 1.72x |
| color.gray2rgba | float64 | alpha-broadcast | 1 | 3.6907 | 2.4647 | 1.50x |
| color.gray2rgb | >u2 | endian-contiguous | 1 | 0.2320 | 0.1522 | 1.52x |
| color.gray2rgba | >u2 | endian-contiguous | 1 | 0.3147 | 0.1459 | 2.16x |
| color.gray2rgb | >u2 | endian-reversed | 1 | 0.2228 | 0.1422 | 1.57x |
| color.gray2rgba | >u2 | endian-reversed | 1 | 0.3207 | 0.1529 | 2.10x |
| color.gray2rgb | >f4 | endian-contiguous | 1 | 0.3181 | 0.2304 | 1.38x |
| color.gray2rgba | >f4 | endian-contiguous | 1 | 0.5287 | 0.3616 | 1.46x |
| color.gray2rgb | >f4 | endian-reversed | 1 | 0.3141 | 0.2017 | 1.56x |
| color.gray2rgba | >f4 | endian-reversed | 1 | 0.5470 | 0.4142 | 1.32x |
| color.gray2rgb | >f8 | endian-contiguous | 1 | 2.3554 | 1.8450 | 1.28x |
| color.gray2rgba | >f8 | endian-contiguous | 1 | 3.6812 | 2.5666 | 1.43x |
| color.gray2rgb | >f8 | endian-reversed | 1 | 2.9316 | 2.1111 | 1.39x |
| color.gray2rgba | >f8 | endian-reversed | 1 | 4.6358 | 2.6290 | 1.76x |
| color.gray2rgb | [('x', '>i2'), ('y', '>f4')] | record-6-contiguous | 1 | 1.1604 | 0.2332 | 4.98x |
| color.gray2rgba | [('x', '>i2'), ('y', '>f4')] | record-6-contiguous | 1 | 3.1099 | 1.5920 | 1.95x |
| color.gray2rgb | [('x', '>i2'), ('y', '>f4')] | record-6-reversed | 1 | 1.2134 | 0.2231 | 5.44x |
| color.gray2rgba | [('x', '>i2'), ('y', '>f4')] | record-6-reversed | 1 | 4.6469 | 2.0571 | 2.26x |
| color.gray2rgb | [('x', '>i2'), ('y', '>f4')] | record-6-strided | 1 | 1.7671 | 0.6112 | 2.89x |
| color.gray2rgba | [('x', '>i2'), ('y', '>f4')] | record-6-strided | 1 | 5.5340 | 2.3403 | 2.36x |
| color.gray2rgb | [('x', '>i2'), ('y', '>f4')] | record-6-fortran | 1 | 1.0829 | 0.2103 | 5.15x |
| color.gray2rgba | [('x', '>i2'), ('y', '>f4')] | record-6-fortran | 1 | 5.6677 | 1.9277 | 2.94x |
| color.gray2rgb | [('record', [('a', '>i2'), ('b', '>u4')]), ('vec', '>f4', (3,))] | record-18-contiguous | 1 | 12.0603 | 6.3577 | 1.90x |
| color.gray2rgba | [('record', [('a', '>i2'), ('b', '>u4')]), ('vec', '>f4', (3,))] | record-18-contiguous | 1 | 23.5046 | 15.4822 | 1.52x |
| color.gray2rgb | [('record', [('a', '>i2'), ('b', '>u4')]), ('vec', '>f4', (3,))] | record-18-reversed | 1 | 9.5659 | 6.2695 | 1.53x |
| color.gray2rgba | [('record', [('a', '>i2'), ('b', '>u4')]), ('vec', '>f4', (3,))] | record-18-reversed | 1 | 22.6731 | 15.4273 | 1.47x |
| color.gray2rgb | [('record', [('a', '>i2'), ('b', '>u4')]), ('vec', '>f4', (3,))] | record-18-strided | 1 | 12.5116 | 7.4204 | 1.69x |
| color.gray2rgba | [('record', [('a', '>i2'), ('b', '>u4')]), ('vec', '>f4', (3,))] | record-18-strided | 1 | 25.3976 | 16.2398 | 1.56x |
| color.gray2rgb | [('record', [('a', '>i2'), ('b', '>u4')]), ('vec', '>f4', (3,))] | record-18-fortran | 1 | 9.5675 | 6.6677 | 1.43x |
| color.gray2rgba | [('record', [('a', '>i2'), ('b', '>u4')]), ('vec', '>f4', (3,))] | record-18-fortran | 1 | 20.9299 | 14.6695 | 1.43x |
| color.gray2rgb | [('z', '>c16'), ('when', '>M8[ns]'), ('dt', '>m8[ms]')] | record-32-contiguous | 1 | 18.2015 | 9.3059 | 1.96x |
| color.gray2rgba | [('z', '>c16'), ('when', '>M8[ns]'), ('dt', '>m8[ms]')] | record-32-contiguous | 1 | 33.0105 | 16.9894 | 1.94x |
| color.gray2rgb | [('z', '>c16'), ('when', '>M8[ns]'), ('dt', '>m8[ms]')] | record-32-reversed | 1 | 20.2010 | 9.0837 | 2.22x |
| color.gray2rgba | [('z', '>c16'), ('when', '>M8[ns]'), ('dt', '>m8[ms]')] | record-32-reversed | 1 | 35.7450 | 17.2454 | 2.07x |
| color.gray2rgb | [('z', '>c16'), ('when', '>M8[ns]'), ('dt', '>m8[ms]')] | record-32-strided | 1 | 15.5988 | 7.8603 | 1.98x |
| color.gray2rgba | [('z', '>c16'), ('when', '>M8[ns]'), ('dt', '>m8[ms]')] | record-32-strided | 1 | 32.7015 | 16.3167 | 2.00x |
| color.gray2rgb | [('z', '>c16'), ('when', '>M8[ns]'), ('dt', '>m8[ms]')] | record-32-fortran | 1 | 14.8018 | 6.6958 | 2.21x |
| color.gray2rgba | [('z', '>c16'), ('when', '>M8[ns]'), ('dt', '>m8[ms]')] | record-32-fortran | 1 | 34.8911 | 15.5409 | 2.25x |
