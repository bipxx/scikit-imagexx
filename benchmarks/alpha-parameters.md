# Alpha and grayscale performance

Original scikit-image 0.26.0. Reference thread pools use one worker; native workers are explicit. Timings include dispatch, integer normalization, validation, layout handling and output allocation. Input construction is excluded. All samples and slower cases are retained in the JSON.

| API | Input | Layout / workload | Workers | Original ms | C++ ms | Speedup |
|---|---|---|---:|---:|---:|---:|
| color.rgba2rgb | bool | contiguous | 1 | 15.2209 | 0.5630 | 27.04x |
| color.rgba2rgb | bool | reversed | 1 | 15.9243 | 2.1416 | 7.44x |
| color.rgba2rgb | bool | strided | 1 | 16.9450 | 2.2664 | 7.48x |
| color.rgba2rgb | bool | fortran | 1 | 15.4942 | 2.1567 | 7.18x |
| color.rgba2rgb | bool | planar | 1 | 11.8250 | 2.3456 | 5.04x |
| color.rgba2rgb | uint8 | contiguous | 1 | 19.0544 | 0.6204 | 30.71x |
| color.rgba2rgb | uint8 | reversed | 1 | 17.4395 | 2.1016 | 8.30x |
| color.rgba2rgb | uint8 | strided | 1 | 18.3803 | 2.2435 | 8.19x |
| color.rgba2rgb | uint8 | fortran | 1 | 12.8349 | 2.1270 | 6.03x |
| color.rgba2rgb | uint8 | planar | 1 | 10.0024 | 2.2811 | 4.38x |
| color.rgba2rgb | int8 | contiguous | 1 | 21.7752 | 2.5645 | 8.49x |
| color.rgba2rgb | int8 | reversed | 1 | 19.3143 | 3.5030 | 5.51x |
| color.rgba2rgb | int8 | strided | 1 | 18.7398 | 2.9142 | 6.43x |
| color.rgba2rgb | int8 | fortran | 1 | 14.8420 | 3.7555 | 3.95x |
| color.rgba2rgb | int8 | planar | 1 | 11.5718 | 3.3765 | 3.43x |
| color.rgba2rgb | uint16 | contiguous | 1 | 16.3736 | 0.7670 | 21.35x |
| color.rgba2rgb | uint16 | reversed | 1 | 17.7236 | 2.2379 | 7.92x |
| color.rgba2rgb | uint16 | strided | 1 | 17.2780 | 2.3766 | 7.27x |
| color.rgba2rgb | uint16 | fortran | 1 | 15.0565 | 2.1585 | 6.98x |
| color.rgba2rgb | uint16 | planar | 1 | 10.0500 | 2.1315 | 4.72x |
| color.rgba2rgb | int16 | contiguous | 1 | 20.5885 | 1.0758 | 19.14x |
| color.rgba2rgb | int16 | reversed | 1 | 16.2848 | 2.4703 | 6.59x |
| color.rgba2rgb | int16 | strided | 1 | 18.2753 | 2.5914 | 7.05x |
| color.rgba2rgb | int16 | fortran | 1 | 13.9592 | 2.5193 | 5.54x |
| color.rgba2rgb | int16 | planar | 1 | 13.7127 | 2.4463 | 5.61x |
| color.rgba2rgb | uint32 | contiguous | 1 | 15.7773 | 1.9809 | 7.96x |
| color.rgba2rgb | uint32 | reversed | 1 | 17.4734 | 2.0634 | 8.47x |
| color.rgba2rgb | uint32 | strided | 1 | 15.5042 | 2.0824 | 7.45x |
| color.rgba2rgb | uint32 | fortran | 1 | 13.1351 | 2.0941 | 6.27x |
| color.rgba2rgb | uint32 | planar | 1 | 10.0771 | 2.1388 | 4.71x |
| color.rgba2rgb | int32 | contiguous | 1 | 16.5669 | 2.4918 | 6.65x |
| color.rgba2rgb | int32 | reversed | 1 | 16.3657 | 2.4755 | 6.61x |
| color.rgba2rgb | int32 | strided | 1 | 20.2555 | 3.8613 | 5.25x |
| color.rgba2rgb | int32 | fortran | 1 | 15.2741 | 3.8739 | 3.94x |
| color.rgba2rgb | int32 | planar | 1 | 7.9773 | 2.3740 | 3.36x |
| color.rgba2rgb | uint64 | contiguous | 1 | 22.2900 | 5.5943 | 3.98x |
| color.rgba2rgb | uint64 | reversed | 1 | 22.4025 | 6.7725 | 3.31x |
| color.rgba2rgb | uint64 | strided | 1 | 21.3946 | 5.0861 | 4.21x |
| color.rgba2rgb | uint64 | fortran | 1 | 19.6309 | 5.4712 | 3.59x |
| color.rgba2rgb | uint64 | planar | 1 | 14.8082 | 5.8506 | 2.53x |
| color.rgba2rgb | int64 | contiguous | 1 | 22.5579 | 3.3041 | 6.83x |
| color.rgba2rgb | int64 | reversed | 1 | 29.1426 | 5.0075 | 5.82x |
| color.rgba2rgb | int64 | strided | 1 | 29.3603 | 4.8367 | 6.07x |
| color.rgba2rgb | int64 | fortran | 1 | 20.3271 | 3.3363 | 6.09x |
| color.rgba2rgb | int64 | planar | 1 | 15.9144 | 3.5250 | 4.51x |
| color.rgba2rgb | float32 | contiguous | 1 | 11.2073 | 0.9098 | 12.32x |
| color.rgba2rgb | float32 | reversed | 1 | 11.2657 | 2.1202 | 5.31x |
| color.rgba2rgb | float32 | strided | 1 | 12.7421 | 2.7326 | 4.66x |
| color.rgba2rgb | float32 | fortran | 1 | 8.3833 | 0.8026 | 10.45x |
| color.rgba2rgb | float32 | planar | 1 | 3.7507 | 0.6386 | 5.87x |
| color.rgba2rgb | float64 | contiguous | 1 | 14.5695 | 2.2853 | 6.38x |
| color.rgba2rgb | float64 | reversed | 1 | 15.3281 | 6.8161 | 2.25x |
| color.rgba2rgb | float64 | strided | 1 | 17.5614 | 8.6165 | 2.04x |
| color.rgba2rgb | float64 | fortran | 1 | 16.8145 | 2.7659 | 6.08x |
| color.rgba2rgb | float64 | planar | 1 | 14.9956 | 3.9747 | 3.77x |
| color.gray2rgb | bool | contiguous | 1 | 0.4644 | 0.3793 | 1.22x |
| color.gray2rgba | bool | contiguous | 1 | 0.2623 | 0.1272 | 2.06x |
| color.gray2rgb | bool | reversed | 1 | 0.4046 | 0.3007 | 1.35x |
| color.gray2rgba | bool | reversed | 1 | 0.2852 | 0.1377 | 2.07x |
| color.gray2rgb | bool | strided | 1 | 0.4414 | 0.3314 | 1.33x |
| color.gray2rgba | bool | strided | 1 | 0.2916 | 0.1347 | 2.16x |
| color.gray2rgb | bool | fortran | 1 | 0.3614 | 0.1477 | 2.45x |
| color.gray2rgba | bool | fortran | 1 | 0.5330 | 0.1142 | 4.67x |
| color.gray2rgb | uint8 | contiguous | 1 | 0.2560 | 0.0838 | 3.06x |
| color.gray2rgba | uint8 | contiguous | 1 | 0.2846 | 0.0757 | 3.76x |
| color.gray2rgb | uint8 | reversed | 1 | 0.1951 | 0.1042 | 1.87x |
| color.gray2rgba | uint8 | reversed | 1 | 0.2894 | 0.0868 | 3.34x |
| color.gray2rgb | uint8 | strided | 1 | 0.4004 | 0.1443 | 2.77x |
| color.gray2rgba | uint8 | strided | 1 | 0.3316 | 0.1046 | 3.17x |
| color.gray2rgb | uint8 | fortran | 1 | 0.4284 | 0.2849 | 1.50x |
| color.gray2rgba | uint8 | fortran | 1 | 0.6241 | 0.0920 | 6.78x |
| color.gray2rgb | int8 | contiguous | 1 | 0.4283 | 0.2670 | 1.60x |
| color.gray2rgba | int8 | contiguous | 1 | 0.2727 | 0.1017 | 2.68x |
| color.gray2rgb | int8 | reversed | 1 | 0.3821 | 0.3500 | 1.09x |
| color.gray2rgba | int8 | reversed | 1 | 0.3083 | 0.0906 | 3.40x |
| color.gray2rgb | int8 | strided | 1 | 0.5351 | 0.3597 | 1.49x |
| color.gray2rgba | int8 | strided | 1 | 0.4516 | 0.1897 | 2.38x |
| color.gray2rgb | int8 | fortran | 1 | 0.4430 | 0.3253 | 1.36x |
| color.gray2rgba | int8 | fortran | 1 | 0.6122 | 0.1144 | 5.35x |
| color.gray2rgb | uint16 | contiguous | 1 | 0.3320 | 0.2146 | 1.55x |
| color.gray2rgba | uint16 | contiguous | 1 | 0.4363 | 0.1881 | 2.32x |
| color.gray2rgb | uint16 | reversed | 1 | 0.2711 | 0.1540 | 1.76x |
| color.gray2rgba | uint16 | reversed | 1 | 0.3272 | 0.2089 | 1.57x |
| color.gray2rgb | uint16 | strided | 1 | 0.3860 | 0.2740 | 1.41x |
| color.gray2rgba | uint16 | strided | 1 | 0.5286 | 0.2063 | 2.56x |
| color.gray2rgb | uint16 | fortran | 1 | 0.2409 | 0.1639 | 1.47x |
| color.gray2rgba | uint16 | fortran | 1 | 0.7129 | 0.1290 | 5.53x |
| color.gray2rgb | int16 | contiguous | 1 | 0.2714 | 0.2041 | 1.33x |
| color.gray2rgba | int16 | contiguous | 1 | 0.3265 | 0.1924 | 1.70x |
| color.gray2rgb | int16 | reversed | 1 | 0.1976 | 0.1454 | 1.36x |
| color.gray2rgba | int16 | reversed | 1 | 0.2896 | 0.1240 | 2.34x |
| color.gray2rgb | int16 | strided | 1 | 0.2571 | 0.2951 | 0.87x |
| color.gray2rgba | int16 | strided | 1 | 0.4207 | 0.2119 | 1.98x |
| color.gray2rgb | int16 | fortran | 1 | 0.3602 | 0.2147 | 1.68x |
| color.gray2rgba | int16 | fortran | 1 | 0.6029 | 0.1567 | 3.85x |
| color.gray2rgb | uint32 | contiguous | 1 | 0.4413 | 0.3244 | 1.36x |
| color.gray2rgba | uint32 | contiguous | 1 | 0.8399 | 0.3405 | 2.47x |
| color.gray2rgb | uint32 | reversed | 1 | 0.4428 | 0.2773 | 1.60x |
| color.gray2rgba | uint32 | reversed | 1 | 0.6853 | 0.3366 | 2.04x |
| color.gray2rgb | uint32 | strided | 1 | 1.3156 | 0.2866 | 4.59x |
| color.gray2rgba | uint32 | strided | 1 | 0.9114 | 0.6117 | 1.49x |
| color.gray2rgb | uint32 | fortran | 1 | 1.2519 | 0.2503 | 5.00x |
| color.gray2rgba | uint32 | fortran | 1 | 0.9349 | 0.2968 | 3.15x |
| color.gray2rgb | int32 | contiguous | 1 | 0.3059 | 0.1545 | 1.98x |
| color.gray2rgba | int32 | contiguous | 1 | 0.6907 | 0.3429 | 2.01x |
| color.gray2rgb | int32 | reversed | 1 | 0.3404 | 0.1846 | 1.84x |
| color.gray2rgba | int32 | reversed | 1 | 0.5858 | 0.2727 | 2.15x |
| color.gray2rgb | int32 | strided | 1 | 0.4667 | 0.2639 | 1.77x |
| color.gray2rgba | int32 | strided | 1 | 0.8936 | 0.6169 | 1.45x |
| color.gray2rgb | int32 | fortran | 1 | 0.4213 | 0.1861 | 2.26x |
| color.gray2rgba | int32 | fortran | 1 | 0.9198 | 0.2440 | 3.77x |
| color.gray2rgb | uint64 | contiguous | 1 | 2.4873 | 1.6993 | 1.46x |
| color.gray2rgba | uint64 | contiguous | 1 | 4.1935 | 3.1537 | 1.33x |
| color.gray2rgb | uint64 | reversed | 1 | 2.4884 | 1.9072 | 1.30x |
| color.gray2rgba | uint64 | reversed | 1 | 4.2748 | 2.9187 | 1.46x |
| color.gray2rgb | uint64 | strided | 1 | 2.6718 | 1.8692 | 1.43x |
| color.gray2rgba | uint64 | strided | 1 | 8.5090 | 4.1689 | 2.04x |
| color.gray2rgb | uint64 | fortran | 1 | 0.8461 | 0.6118 | 1.38x |
| color.gray2rgba | uint64 | fortran | 1 | 4.8989 | 2.9680 | 1.65x |
| color.gray2rgb | int64 | contiguous | 1 | 0.7163 | 0.3736 | 1.92x |
| color.gray2rgba | int64 | contiguous | 1 | 7.3379 | 4.1997 | 1.75x |
| color.gray2rgb | int64 | reversed | 1 | 0.9667 | 0.5180 | 1.87x |
| color.gray2rgba | int64 | reversed | 1 | 4.5912 | 2.8368 | 1.62x |
| color.gray2rgb | int64 | strided | 1 | 2.4313 | 1.6194 | 1.50x |
| color.gray2rgba | int64 | strided | 1 | 8.5759 | 3.7476 | 2.29x |
| color.gray2rgb | int64 | fortran | 1 | 2.6053 | 2.0352 | 1.28x |
| color.gray2rgba | int64 | fortran | 1 | 8.7386 | 3.0422 | 2.87x |
| color.gray2rgb | float32 | contiguous | 1 | 0.3897 | 0.1605 | 2.43x |
| color.gray2rgba | float32 | contiguous | 1 | 0.7358 | 0.3645 | 2.02x |
| color.gray2rgb | float32 | reversed | 1 | 0.4141 | 0.2150 | 1.93x |
| color.gray2rgba | float32 | reversed | 1 | 0.9254 | 0.3584 | 2.58x |
| color.gray2rgb | float32 | strided | 1 | 0.4625 | 0.2282 | 2.03x |
| color.gray2rgba | float32 | strided | 1 | 0.7856 | 0.5440 | 1.44x |
| color.gray2rgb | float32 | fortran | 1 | 0.3980 | 0.1871 | 2.13x |
| color.gray2rgba | float32 | fortran | 1 | 0.9333 | 0.2653 | 3.52x |
| color.gray2rgb | float64 | contiguous | 1 | 3.2978 | 2.2280 | 1.48x |
| color.gray2rgba | float64 | contiguous | 1 | 3.5485 | 2.8978 | 1.22x |
| color.gray2rgb | float64 | reversed | 1 | 2.3980 | 1.8625 | 1.29x |
| color.gray2rgba | float64 | reversed | 1 | 3.1518 | 2.4586 | 1.28x |
| color.gray2rgb | float64 | strided | 1 | 2.9774 | 1.9531 | 1.52x |
| color.gray2rgba | float64 | strided | 1 | 4.9931 | 3.2911 | 1.52x |
| color.gray2rgb | float64 | fortran | 1 | 1.2541 | 0.8579 | 1.46x |
| color.gray2rgba | float64 | fortran | 1 | 13.0260 | 3.1877 | 4.09x |
| color.gray2rgb | float16 | contiguous | 1 | 0.2458 | 0.1814 | 1.36x |
| color.gray2rgba | float16 | contiguous | 1 | 1.0119 | 0.7507 | 1.35x |
| color.gray2rgb | float16 | reversed | 1 | 0.2729 | 0.1976 | 1.38x |
| color.gray2rgba | float16 | reversed | 1 | 0.9247 | 0.6974 | 1.33x |
| color.gray2rgb | float16 | strided | 1 | 0.2761 | 0.1896 | 1.46x |
| color.gray2rgba | float16 | strided | 1 | 1.2365 | 0.6570 | 1.88x |
| color.gray2rgb | float16 | fortran | 1 | 0.2521 | 0.1644 | 1.53x |
| color.gray2rgba | float16 | fortran | 1 | 1.3571 | 0.7173 | 1.89x |
| color.rgba2rgb | uint8 | background-black | 1 | 18.5282 | 0.8443 | 21.95x |
| color.rgba2rgb | uint8 | background-color | 1 | 17.2725 | 0.8278 | 20.87x |
| color.rgba2rgb | uint8 | background-nan | 1 | 18.4766 | 1.0122 | 18.25x |
| color.gray2rgba | uint8 | alpha-scalar | 1 | 0.3761 | 0.0829 | 4.54x |
| color.gray2rgba | uint8 | alpha-typed-scalar | 1 | 0.2749 | 0.0898 | 3.06x |
| color.gray2rgba | uint8 | alpha-array | 1 | 0.4249 | 0.1041 | 4.08x |
| color.gray2rgba | uint8 | alpha-broadcast | 1 | 0.3760 | 0.0753 | 4.99x |
| color.rgba2rgb | float32 | background-black | 1 | 10.6942 | 0.9590 | 11.15x |
| color.rgba2rgb | float32 | background-color | 1 | 9.9335 | 0.8314 | 11.95x |
| color.rgba2rgb | float32 | background-nan | 1 | 11.6275 | 0.9372 | 12.41x |
| color.gray2rgba | float32 | alpha-scalar | 1 | 0.8698 | 0.3907 | 2.23x |
| color.gray2rgba | float32 | alpha-typed-scalar | 1 | 0.7139 | 0.3567 | 2.00x |
| color.gray2rgba | float32 | alpha-array | 1 | 1.5724 | 0.9394 | 1.67x |
| color.gray2rgba | float32 | alpha-broadcast | 1 | 0.6554 | 0.2416 | 2.71x |
| color.rgba2rgb | float64 | background-black | 1 | 16.8574 | 2.4127 | 6.99x |
| color.rgba2rgb | float64 | background-color | 1 | 15.6251 | 2.6258 | 5.95x |
| color.rgba2rgb | float64 | background-nan | 1 | 14.3427 | 2.0295 | 7.07x |
| color.gray2rgba | float64 | alpha-scalar | 1 | 4.4916 | 2.9107 | 1.54x |
| color.gray2rgba | float64 | alpha-typed-scalar | 1 | 4.3334 | 3.2851 | 1.32x |
| color.gray2rgba | float64 | alpha-array | 1 | 8.8158 | 4.2717 | 2.06x |
| color.gray2rgba | float64 | alpha-broadcast | 1 | 6.4638 | 2.8658 | 2.26x |
| color.gray2rgb | >u2 | endian-contiguous | 1 | 0.7286 | 0.5443 | 1.34x |
| color.gray2rgba | >u2 | endian-contiguous | 1 | 0.4692 | 0.1553 | 3.02x |
| color.gray2rgb | >u2 | endian-reversed | 1 | 0.2856 | 0.2124 | 1.34x |
| color.gray2rgba | >u2 | endian-reversed | 1 | 0.5153 | 0.1597 | 3.23x |
| color.gray2rgb | >f4 | endian-contiguous | 1 | 0.3717 | 0.2575 | 1.44x |
| color.gray2rgba | >f4 | endian-contiguous | 1 | 0.7595 | 0.7191 | 1.06x |
| color.gray2rgb | >f4 | endian-reversed | 1 | 0.4336 | 0.2687 | 1.61x |
| color.gray2rgba | >f4 | endian-reversed | 1 | 0.6426 | 0.4504 | 1.43x |
| color.gray2rgb | >f8 | endian-contiguous | 1 | 2.4595 | 1.5548 | 1.58x |
| color.gray2rgba | >f8 | endian-contiguous | 1 | 4.2238 | 3.2177 | 1.31x |
| color.gray2rgb | >f8 | endian-reversed | 1 | 1.2059 | 0.6072 | 1.99x |
| color.gray2rgba | >f8 | endian-reversed | 1 | 5.7328 | 3.5275 | 1.63x |
| color.gray2rgb | [('x', '>i2'), ('y', '>f4')] | record-6-contiguous | 1 | 3.1369 | 0.3478 | 9.02x |
| color.gray2rgba | [('x', '>i2'), ('y', '>f4')] | record-6-contiguous | 1 | 5.8190 | 3.2693 | 1.78x |
| color.gray2rgb | [('x', '>i2'), ('y', '>f4')] | record-6-reversed | 1 | 2.3748 | 0.3916 | 6.06x |
| color.gray2rgba | [('x', '>i2'), ('y', '>f4')] | record-6-reversed | 1 | 6.7629 | 3.2313 | 2.09x |
| color.gray2rgb | [('x', '>i2'), ('y', '>f4')] | record-6-strided | 1 | 5.9787 | 1.2576 | 4.75x |
| color.gray2rgba | [('x', '>i2'), ('y', '>f4')] | record-6-strided | 1 | 10.9992 | 3.1324 | 3.51x |
| color.gray2rgb | [('x', '>i2'), ('y', '>f4')] | record-6-fortran | 1 | 2.5786 | 0.3924 | 6.57x |
| color.gray2rgba | [('x', '>i2'), ('y', '>f4')] | record-6-fortran | 1 | 7.5165 | 2.6087 | 2.88x |
| color.gray2rgb | [('record', [('a', '>i2'), ('b', '>u4')]), ('vec', '>f4', (3,))] | record-18-contiguous | 1 | 16.8212 | 10.8437 | 1.55x |
| color.gray2rgba | [('record', [('a', '>i2'), ('b', '>u4')]), ('vec', '>f4', (3,))] | record-18-contiguous | 1 | 43.1662 | 23.6149 | 1.83x |
| color.gray2rgb | [('record', [('a', '>i2'), ('b', '>u4')]), ('vec', '>f4', (3,))] | record-18-reversed | 1 | 14.4872 | 9.9023 | 1.46x |
| color.gray2rgba | [('record', [('a', '>i2'), ('b', '>u4')]), ('vec', '>f4', (3,))] | record-18-reversed | 1 | 40.3782 | 22.5634 | 1.79x |
| color.gray2rgb | [('record', [('a', '>i2'), ('b', '>u4')]), ('vec', '>f4', (3,))] | record-18-strided | 1 | 16.9348 | 9.4971 | 1.78x |
| color.gray2rgba | [('record', [('a', '>i2'), ('b', '>u4')]), ('vec', '>f4', (3,))] | record-18-strided | 1 | 38.4189 | 19.7393 | 1.95x |
| color.gray2rgb | [('record', [('a', '>i2'), ('b', '>u4')]), ('vec', '>f4', (3,))] | record-18-fortran | 1 | 14.3442 | 8.8256 | 1.63x |
| color.gray2rgba | [('record', [('a', '>i2'), ('b', '>u4')]), ('vec', '>f4', (3,))] | record-18-fortran | 1 | 32.6669 | 20.1684 | 1.62x |
| color.gray2rgb | [('z', '>c16'), ('when', '>M8[ns]'), ('dt', '>m8[ms]')] | record-32-contiguous | 1 | 18.4786 | 9.8368 | 1.88x |
| color.gray2rgba | [('z', '>c16'), ('when', '>M8[ns]'), ('dt', '>m8[ms]')] | record-32-contiguous | 1 | 46.4099 | 23.5507 | 1.97x |
| color.gray2rgb | [('z', '>c16'), ('when', '>M8[ns]'), ('dt', '>m8[ms]')] | record-32-reversed | 1 | 24.3958 | 10.7292 | 2.27x |
| color.gray2rgba | [('z', '>c16'), ('when', '>M8[ns]'), ('dt', '>m8[ms]')] | record-32-reversed | 1 | 57.1743 | 26.5522 | 2.15x |
| color.gray2rgb | [('z', '>c16'), ('when', '>M8[ns]'), ('dt', '>m8[ms]')] | record-32-strided | 1 | 32.2684 | 14.1660 | 2.28x |
| color.gray2rgba | [('z', '>c16'), ('when', '>M8[ns]'), ('dt', '>m8[ms]')] | record-32-strided | 1 | 59.2261 | 29.9576 | 1.98x |
| color.gray2rgb | [('z', '>c16'), ('when', '>M8[ns]'), ('dt', '>m8[ms]')] | record-32-fortran | 1 | 23.5667 | 10.5503 | 2.23x |
| color.gray2rgba | [('z', '>c16'), ('when', '>M8[ns]'), ('dt', '>m8[ms]')] | record-32-fortran | 1 | 40.7556 | 23.3269 | 1.75x |
