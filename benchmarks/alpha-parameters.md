# Alpha and grayscale performance

Original scikit-image 0.26.0. Reference thread pools use one worker; native workers are explicit. Timings include dispatch, integer normalization, validation, layout handling and output allocation. Input construction is excluded. All samples and slower cases are retained in the JSON.

| API | Input | Layout / workload | Workers | Original ms | C++ ms | Speedup |
|---|---|---|---:|---:|---:|---:|
| color.rgba2rgb | bool | contiguous | 1 | 15.5717 | 0.5673 | 27.45x |
| color.rgba2rgb | bool | reversed | 1 | 14.9060 | 2.0457 | 7.29x |
| color.rgba2rgb | bool | strided | 1 | 14.5157 | 2.1273 | 6.82x |
| color.rgba2rgb | bool | fortran | 1 | 11.8127 | 2.0630 | 5.73x |
| color.rgba2rgb | bool | planar | 1 | 8.0416 | 2.1765 | 3.69x |
| color.rgba2rgb | uint8 | contiguous | 1 | 14.0673 | 0.5778 | 24.35x |
| color.rgba2rgb | uint8 | reversed | 1 | 15.1272 | 2.0380 | 7.42x |
| color.rgba2rgb | uint8 | strided | 1 | 15.3685 | 2.1478 | 7.16x |
| color.rgba2rgb | uint8 | fortran | 1 | 12.2645 | 2.1212 | 5.78x |
| color.rgba2rgb | uint8 | planar | 1 | 8.1528 | 2.0826 | 3.91x |
| color.rgba2rgb | int8 | contiguous | 1 | 15.2759 | 0.7333 | 20.83x |
| color.rgba2rgb | int8 | reversed | 1 | 15.8302 | 3.2290 | 4.90x |
| color.rgba2rgb | int8 | strided | 1 | 15.9456 | 3.4525 | 4.62x |
| color.rgba2rgb | int8 | fortran | 1 | 13.3251 | 2.6265 | 5.07x |
| color.rgba2rgb | int8 | planar | 1 | 10.8929 | 3.3135 | 3.29x |
| color.rgba2rgb | uint16 | contiguous | 1 | 15.2110 | 0.7067 | 21.52x |
| color.rgba2rgb | uint16 | reversed | 1 | 14.5103 | 2.0730 | 7.00x |
| color.rgba2rgb | uint16 | strided | 1 | 14.8077 | 2.0991 | 7.05x |
| color.rgba2rgb | uint16 | fortran | 1 | 12.2244 | 2.0619 | 5.93x |
| color.rgba2rgb | uint16 | planar | 1 | 11.1138 | 2.2335 | 4.98x |
| color.rgba2rgb | int16 | contiguous | 1 | 15.5638 | 0.7433 | 20.94x |
| color.rgba2rgb | int16 | reversed | 1 | 15.3775 | 2.3336 | 6.59x |
| color.rgba2rgb | int16 | strided | 1 | 15.2470 | 2.4176 | 6.31x |
| color.rgba2rgb | int16 | fortran | 1 | 13.9338 | 2.3849 | 5.84x |
| color.rgba2rgb | int16 | planar | 1 | 9.6242 | 2.2863 | 4.21x |
| color.rgba2rgb | uint32 | contiguous | 1 | 14.8033 | 2.1460 | 6.90x |
| color.rgba2rgb | uint32 | reversed | 1 | 15.2282 | 1.8513 | 8.23x |
| color.rgba2rgb | uint32 | strided | 1 | 14.7731 | 2.2600 | 6.54x |
| color.rgba2rgb | uint32 | fortran | 1 | 12.4341 | 1.9566 | 6.36x |
| color.rgba2rgb | uint32 | planar | 1 | 8.1617 | 1.9893 | 4.10x |
| color.rgba2rgb | int32 | contiguous | 1 | 14.6868 | 2.2429 | 6.55x |
| color.rgba2rgb | int32 | reversed | 1 | 15.5028 | 2.2604 | 6.86x |
| color.rgba2rgb | int32 | strided | 1 | 15.1085 | 2.4275 | 6.22x |
| color.rgba2rgb | int32 | fortran | 1 | 12.0139 | 2.2549 | 5.33x |
| color.rgba2rgb | int32 | planar | 1 | 8.5479 | 2.2444 | 3.81x |
| color.rgba2rgb | uint64 | contiguous | 1 | 17.8858 | 4.6366 | 3.86x |
| color.rgba2rgb | uint64 | reversed | 1 | 20.0946 | 4.6887 | 4.29x |
| color.rgba2rgb | uint64 | strided | 1 | 20.5072 | 4.3999 | 4.66x |
| color.rgba2rgb | uint64 | fortran | 1 | 16.9541 | 4.7573 | 3.56x |
| color.rgba2rgb | uint64 | planar | 1 | 15.7406 | 4.8059 | 3.28x |
| color.rgba2rgb | int64 | contiguous | 1 | 15.7323 | 2.4351 | 6.46x |
| color.rgba2rgb | int64 | reversed | 1 | 16.5271 | 2.4623 | 6.71x |
| color.rgba2rgb | int64 | strided | 1 | 16.7276 | 2.8466 | 5.88x |
| color.rgba2rgb | int64 | fortran | 1 | 13.1847 | 2.4561 | 5.37x |
| color.rgba2rgb | int64 | planar | 1 | 10.5914 | 2.4069 | 4.40x |
| color.rgba2rgb | float32 | contiguous | 1 | 9.6661 | 0.5766 | 16.76x |
| color.rgba2rgb | float32 | reversed | 1 | 7.5458 | 1.7424 | 4.33x |
| color.rgba2rgb | float32 | strided | 1 | 9.6958 | 2.7157 | 3.57x |
| color.rgba2rgb | float32 | fortran | 1 | 4.8540 | 0.6466 | 7.51x |
| color.rgba2rgb | float32 | planar | 1 | 3.6572 | 0.6036 | 6.06x |
| color.rgba2rgb | float64 | contiguous | 1 | 12.1357 | 1.1605 | 10.46x |
| color.rgba2rgb | float64 | reversed | 1 | 12.2049 | 4.4939 | 2.72x |
| color.rgba2rgb | float64 | strided | 1 | 12.9294 | 6.0766 | 2.13x |
| color.rgba2rgb | float64 | fortran | 1 | 10.3847 | 0.9936 | 10.45x |
| color.rgba2rgb | float64 | planar | 1 | 6.7239 | 0.9909 | 6.79x |
| color.gray2rgb | bool | contiguous | 1 | 0.3323 | 0.2190 | 1.52x |
| color.gray2rgba | bool | contiguous | 1 | 0.2583 | 0.0774 | 3.34x |
| color.gray2rgb | bool | reversed | 1 | 0.3557 | 0.2525 | 1.41x |
| color.gray2rgba | bool | reversed | 1 | 0.2500 | 0.0691 | 3.62x |
| color.gray2rgb | bool | strided | 1 | 0.3425 | 0.2430 | 1.41x |
| color.gray2rgba | bool | strided | 1 | 0.2898 | 0.0865 | 3.35x |
| color.gray2rgb | bool | fortran | 1 | 0.3392 | 0.2188 | 1.55x |
| color.gray2rgba | bool | fortran | 1 | 0.4883 | 0.0644 | 7.58x |
| color.gray2rgb | uint8 | contiguous | 1 | 0.3280 | 0.2096 | 1.56x |
| color.gray2rgba | uint8 | contiguous | 1 | 0.2610 | 0.0632 | 4.13x |
| color.gray2rgb | uint8 | reversed | 1 | 0.3119 | 0.2142 | 1.46x |
| color.gray2rgba | uint8 | reversed | 1 | 0.2558 | 0.0649 | 3.94x |
| color.gray2rgb | uint8 | strided | 1 | 0.3289 | 0.2408 | 1.37x |
| color.gray2rgba | uint8 | strided | 1 | 0.2822 | 0.0924 | 3.05x |
| color.gray2rgb | uint8 | fortran | 1 | 0.1922 | 0.0719 | 2.67x |
| color.gray2rgba | uint8 | fortran | 1 | 0.4863 | 0.0686 | 7.08x |
| color.gray2rgb | int8 | contiguous | 1 | 0.2037 | 0.0751 | 2.71x |
| color.gray2rgba | int8 | contiguous | 1 | 0.2713 | 0.0615 | 4.41x |
| color.gray2rgb | int8 | reversed | 1 | 0.3154 | 0.2116 | 1.49x |
| color.gray2rgba | int8 | reversed | 1 | 0.2537 | 0.0648 | 3.91x |
| color.gray2rgb | int8 | strided | 1 | 0.3399 | 0.2307 | 1.47x |
| color.gray2rgba | int8 | strided | 1 | 0.2875 | 0.0873 | 3.29x |
| color.gray2rgb | int8 | fortran | 1 | 0.3199 | 0.2065 | 1.55x |
| color.gray2rgba | int8 | fortran | 1 | 0.4750 | 0.0701 | 6.77x |
| color.gray2rgb | uint16 | contiguous | 1 | 0.2007 | 0.1389 | 1.45x |
| color.gray2rgba | uint16 | contiguous | 1 | 0.3013 | 0.1538 | 1.96x |
| color.gray2rgb | uint16 | reversed | 1 | 0.5132 | 0.4593 | 1.12x |
| color.gray2rgba | uint16 | reversed | 1 | 0.2734 | 0.1195 | 2.29x |
| color.gray2rgb | uint16 | strided | 1 | 0.6179 | 0.5359 | 1.15x |
| color.gray2rgba | uint16 | strided | 1 | 0.4053 | 0.1602 | 2.53x |
| color.gray2rgb | uint16 | fortran | 1 | 0.6329 | 0.5508 | 1.15x |
| color.gray2rgba | uint16 | fortran | 1 | 0.6665 | 0.1673 | 3.98x |
| color.gray2rgb | int16 | contiguous | 1 | 0.2077 | 0.1488 | 1.40x |
| color.gray2rgba | int16 | contiguous | 1 | 0.3004 | 0.1262 | 2.38x |
| color.gray2rgb | int16 | reversed | 1 | 0.1856 | 0.1392 | 1.33x |
| color.gray2rgba | int16 | reversed | 1 | 0.2732 | 0.1140 | 2.40x |
| color.gray2rgb | int16 | strided | 1 | 0.2263 | 0.1653 | 1.37x |
| color.gray2rgba | int16 | strided | 1 | 0.3279 | 0.1427 | 2.30x |
| color.gray2rgb | int16 | fortran | 1 | 0.1862 | 0.1362 | 1.37x |
| color.gray2rgba | int16 | fortran | 1 | 0.5953 | 0.1164 | 5.12x |
| color.gray2rgb | uint32 | contiguous | 1 | 0.3025 | 0.1531 | 1.98x |
| color.gray2rgba | uint32 | contiguous | 1 | 0.4918 | 0.2190 | 2.25x |
| color.gray2rgb | uint32 | reversed | 1 | 0.3441 | 0.1846 | 1.86x |
| color.gray2rgba | uint32 | reversed | 1 | 0.4969 | 0.2248 | 2.21x |
| color.gray2rgb | uint32 | strided | 1 | 1.0409 | 0.8327 | 1.25x |
| color.gray2rgba | uint32 | strided | 1 | 0.6522 | 0.2935 | 2.22x |
| color.gray2rgb | uint32 | fortran | 1 | 1.0341 | 0.8136 | 1.27x |
| color.gray2rgba | uint32 | fortran | 1 | 0.8894 | 0.2352 | 3.78x |
| color.gray2rgb | int32 | contiguous | 1 | 0.3104 | 0.1535 | 2.02x |
| color.gray2rgba | int32 | contiguous | 1 | 0.4866 | 0.2120 | 2.30x |
| color.gray2rgb | int32 | reversed | 1 | 0.2988 | 0.1648 | 1.81x |
| color.gray2rgba | int32 | reversed | 1 | 0.4939 | 0.2297 | 2.15x |
| color.gray2rgb | int32 | strided | 1 | 0.3613 | 0.1917 | 1.88x |
| color.gray2rgba | int32 | strided | 1 | 0.5517 | 0.2903 | 1.90x |
| color.gray2rgb | int32 | fortran | 1 | 0.2920 | 0.1561 | 1.87x |
| color.gray2rgba | int32 | fortran | 1 | 0.8657 | 0.2215 | 3.91x |
| color.gray2rgb | uint64 | contiguous | 1 | 1.9427 | 1.4935 | 1.30x |
| color.gray2rgba | uint64 | contiguous | 1 | 3.0417 | 2.0401 | 1.49x |
| color.gray2rgb | uint64 | reversed | 1 | 2.0327 | 1.6077 | 1.26x |
| color.gray2rgba | uint64 | reversed | 1 | 3.2493 | 2.1601 | 1.50x |
| color.gray2rgb | uint64 | strided | 1 | 2.3130 | 1.6509 | 1.40x |
| color.gray2rgba | uint64 | strided | 1 | 3.6169 | 2.3621 | 1.53x |
| color.gray2rgb | uint64 | fortran | 1 | 2.0569 | 1.5396 | 1.34x |
| color.gray2rgba | uint64 | fortran | 1 | 4.9549 | 2.1231 | 2.33x |
| color.gray2rgb | int64 | contiguous | 1 | 2.0097 | 1.4317 | 1.40x |
| color.gray2rgba | int64 | contiguous | 1 | 3.4013 | 2.3245 | 1.46x |
| color.gray2rgb | int64 | reversed | 1 | 2.0620 | 0.4647 | 4.44x |
| color.gray2rgba | int64 | reversed | 1 | 3.1608 | 2.2618 | 1.40x |
| color.gray2rgb | int64 | strided | 1 | 2.1442 | 0.8711 | 2.46x |
| color.gray2rgba | int64 | strided | 1 | 3.9609 | 2.3300 | 1.70x |
| color.gray2rgb | int64 | fortran | 1 | 0.6942 | 0.3251 | 2.14x |
| color.gray2rgba | int64 | fortran | 1 | 5.8278 | 2.4739 | 2.36x |
| color.gray2rgb | float32 | contiguous | 1 | 0.3404 | 0.2015 | 1.69x |
| color.gray2rgba | float32 | contiguous | 1 | 1.0304 | 0.2882 | 3.58x |
| color.gray2rgb | float32 | reversed | 1 | 0.3670 | 0.1727 | 2.12x |
| color.gray2rgba | float32 | reversed | 1 | 0.6105 | 0.2609 | 2.34x |
| color.gray2rgb | float32 | strided | 1 | 0.4644 | 0.2503 | 1.86x |
| color.gray2rgba | float32 | strided | 1 | 0.9030 | 0.3828 | 2.36x |
| color.gray2rgb | float32 | fortran | 1 | 0.3020 | 0.1708 | 1.77x |
| color.gray2rgba | float32 | fortran | 1 | 0.9929 | 0.2693 | 3.69x |
| color.gray2rgb | float64 | contiguous | 1 | 2.8320 | 1.9137 | 1.48x |
| color.gray2rgba | float64 | contiguous | 1 | 3.9992 | 2.3719 | 1.69x |
| color.gray2rgb | float64 | reversed | 1 | 2.1613 | 1.5568 | 1.39x |
| color.gray2rgba | float64 | reversed | 1 | 3.3674 | 2.3450 | 1.44x |
| color.gray2rgb | float64 | strided | 1 | 2.1266 | 1.2418 | 1.71x |
| color.gray2rgba | float64 | strided | 1 | 3.5906 | 2.2692 | 1.58x |
| color.gray2rgb | float64 | fortran | 1 | 0.8753 | 0.4805 | 1.82x |
| color.gray2rgba | float64 | fortran | 1 | 4.9622 | 2.3487 | 2.11x |
| color.gray2rgb | float16 | contiguous | 1 | 0.1905 | 0.1396 | 1.37x |
| color.gray2rgba | float16 | contiguous | 1 | 0.9532 | 0.6841 | 1.39x |
| color.gray2rgb | float16 | reversed | 1 | 0.2128 | 0.1640 | 1.30x |
| color.gray2rgba | float16 | reversed | 1 | 0.8415 | 0.6404 | 1.31x |
| color.gray2rgb | float16 | strided | 1 | 0.3816 | 0.2047 | 1.86x |
| color.gray2rgba | float16 | strided | 1 | 0.9239 | 0.5825 | 1.59x |
| color.gray2rgb | float16 | fortran | 1 | 0.2122 | 0.1438 | 1.48x |
| color.gray2rgba | float16 | fortran | 1 | 1.0459 | 0.5784 | 1.81x |
| color.rgba2rgb | uint8 | background-black | 1 | 15.7069 | 0.6341 | 24.77x |
| color.rgba2rgb | uint8 | background-color | 1 | 15.7628 | 0.6726 | 23.44x |
| color.rgba2rgb | uint8 | background-nan | 1 | 15.8950 | 0.6914 | 22.99x |
| color.gray2rgba | uint8 | alpha-scalar | 1 | 0.2795 | 0.0798 | 3.50x |
| color.gray2rgba | uint8 | alpha-typed-scalar | 1 | 0.2610 | 0.0603 | 4.33x |
| color.gray2rgba | uint8 | alpha-array | 1 | 0.2939 | 0.0598 | 4.91x |
| color.gray2rgba | uint8 | alpha-broadcast | 1 | 0.2761 | 0.0649 | 4.25x |
| color.rgba2rgb | float32 | background-black | 1 | 9.7072 | 0.6567 | 14.78x |
| color.rgba2rgb | float32 | background-color | 1 | 9.2751 | 0.6178 | 15.01x |
| color.rgba2rgb | float32 | background-nan | 1 | 8.5767 | 0.6138 | 13.97x |
| color.gray2rgba | float32 | alpha-scalar | 1 | 0.5047 | 0.2094 | 2.41x |
| color.gray2rgba | float32 | alpha-typed-scalar | 1 | 0.4922 | 0.2124 | 2.32x |
| color.gray2rgba | float32 | alpha-array | 1 | 0.6907 | 0.5750 | 1.20x |
| color.gray2rgba | float32 | alpha-broadcast | 1 | 0.5175 | 0.2279 | 2.27x |
| color.rgba2rgb | float64 | background-black | 1 | 12.0606 | 0.8945 | 13.48x |
| color.rgba2rgb | float64 | background-color | 1 | 13.1151 | 1.5585 | 8.42x |
| color.rgba2rgb | float64 | background-nan | 1 | 12.6310 | 1.0767 | 11.73x |
| color.gray2rgba | float64 | alpha-scalar | 1 | 3.1198 | 2.1141 | 1.48x |
| color.gray2rgba | float64 | alpha-typed-scalar | 1 | 2.7960 | 2.1711 | 1.29x |
| color.gray2rgba | float64 | alpha-array | 1 | 3.7052 | 2.4042 | 1.54x |
| color.gray2rgba | float64 | alpha-broadcast | 1 | 2.8562 | 2.0164 | 1.42x |
| color.gray2rgb | >u2 | endian-contiguous | 1 | 0.2219 | 0.1411 | 1.57x |
| color.gray2rgba | >u2 | endian-contiguous | 1 | 0.3079 | 0.1213 | 2.54x |
| color.gray2rgb | >u2 | endian-reversed | 1 | 0.5227 | 0.4544 | 1.15x |
| color.gray2rgba | >u2 | endian-reversed | 1 | 0.3000 | 0.1173 | 2.56x |
| color.gray2rgb | >f4 | endian-contiguous | 1 | 0.2919 | 0.2016 | 1.45x |
| color.gray2rgba | >f4 | endian-contiguous | 1 | 0.4951 | 0.3624 | 1.37x |
| color.gray2rgb | >f4 | endian-reversed | 1 | 0.2914 | 0.1941 | 1.50x |
| color.gray2rgba | >f4 | endian-reversed | 1 | 0.4976 | 0.3717 | 1.34x |
| color.gray2rgb | >f8 | endian-contiguous | 1 | 2.0299 | 1.6010 | 1.27x |
| color.gray2rgba | >f8 | endian-contiguous | 1 | 2.9604 | 2.3399 | 1.27x |
| color.gray2rgb | >f8 | endian-reversed | 1 | 1.9844 | 1.6614 | 1.19x |
| color.gray2rgba | >f8 | endian-reversed | 1 | 3.6189 | 2.4303 | 1.49x |
| color.gray2rgb | [('x', '>i2'), ('y', '>f4')] | record-6-contiguous | 1 | 1.0465 | 0.1894 | 5.53x |
| color.gray2rgba | [('x', '>i2'), ('y', '>f4')] | record-6-contiguous | 1 | 3.5790 | 1.6186 | 2.21x |
| color.gray2rgb | [('x', '>i2'), ('y', '>f4')] | record-6-reversed | 1 | 1.0449 | 0.1865 | 5.60x |
| color.gray2rgba | [('x', '>i2'), ('y', '>f4')] | record-6-reversed | 1 | 3.4010 | 1.6419 | 2.07x |
| color.gray2rgb | [('x', '>i2'), ('y', '>f4')] | record-6-strided | 1 | 1.2310 | 0.4539 | 2.71x |
| color.gray2rgba | [('x', '>i2'), ('y', '>f4')] | record-6-strided | 1 | 3.5691 | 1.6929 | 2.11x |
| color.gray2rgb | [('x', '>i2'), ('y', '>f4')] | record-6-fortran | 1 | 1.0308 | 0.2188 | 4.71x |
| color.gray2rgba | [('x', '>i2'), ('y', '>f4')] | record-6-fortran | 1 | 2.7923 | 1.4836 | 1.88x |
| color.gray2rgb | [('record', [('a', '>i2'), ('b', '>u4')]), ('vec', '>f4', (3,))] | record-18-contiguous | 1 | 8.6194 | 6.1292 | 1.41x |
| color.gray2rgba | [('record', [('a', '>i2'), ('b', '>u4')]), ('vec', '>f4', (3,))] | record-18-contiguous | 1 | 17.4332 | 14.3436 | 1.22x |
| color.gray2rgb | [('record', [('a', '>i2'), ('b', '>u4')]), ('vec', '>f4', (3,))] | record-18-reversed | 1 | 8.3659 | 6.1566 | 1.36x |
| color.gray2rgba | [('record', [('a', '>i2'), ('b', '>u4')]), ('vec', '>f4', (3,))] | record-18-reversed | 1 | 18.2790 | 14.6565 | 1.25x |
| color.gray2rgb | [('record', [('a', '>i2'), ('b', '>u4')]), ('vec', '>f4', (3,))] | record-18-strided | 1 | 8.6473 | 6.7049 | 1.29x |
| color.gray2rgba | [('record', [('a', '>i2'), ('b', '>u4')]), ('vec', '>f4', (3,))] | record-18-strided | 1 | 19.4198 | 14.7934 | 1.31x |
| color.gray2rgb | [('record', [('a', '>i2'), ('b', '>u4')]), ('vec', '>f4', (3,))] | record-18-fortran | 1 | 7.8219 | 6.0255 | 1.30x |
| color.gray2rgba | [('record', [('a', '>i2'), ('b', '>u4')]), ('vec', '>f4', (3,))] | record-18-fortran | 1 | 18.7926 | 14.5350 | 1.29x |
| color.gray2rgb | [('z', '>c16'), ('when', '>M8[ns]'), ('dt', '>m8[ms]')] | record-32-contiguous | 1 | 12.2777 | 6.5261 | 1.88x |
| color.gray2rgba | [('z', '>c16'), ('when', '>M8[ns]'), ('dt', '>m8[ms]')] | record-32-contiguous | 1 | 25.2495 | 13.8666 | 1.82x |
| color.gray2rgb | [('z', '>c16'), ('when', '>M8[ns]'), ('dt', '>m8[ms]')] | record-32-reversed | 1 | 12.4362 | 6.6711 | 1.86x |
| color.gray2rgba | [('z', '>c16'), ('when', '>M8[ns]'), ('dt', '>m8[ms]')] | record-32-reversed | 1 | 29.3266 | 15.1163 | 1.94x |
| color.gray2rgb | [('z', '>c16'), ('when', '>M8[ns]'), ('dt', '>m8[ms]')] | record-32-strided | 1 | 15.6118 | 7.6693 | 2.04x |
| color.gray2rgba | [('z', '>c16'), ('when', '>M8[ns]'), ('dt', '>m8[ms]')] | record-32-strided | 1 | 28.2980 | 15.8954 | 1.78x |
| color.gray2rgb | [('z', '>c16'), ('when', '>M8[ns]'), ('dt', '>m8[ms]')] | record-32-fortran | 1 | 11.6343 | 6.3897 | 1.82x |
| color.gray2rgba | [('z', '>c16'), ('when', '>M8[ns]'), ('dt', '>m8[ms]')] | record-32-fortran | 1 | 25.3515 | 13.7133 | 1.85x |
