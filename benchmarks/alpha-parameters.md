# Alpha and grayscale performance

Original scikit-image 0.26.0. Reference thread pools use one worker; native workers are explicit. Timings include dispatch, integer normalization, validation, layout handling and output allocation. Input construction is excluded. All samples and slower cases are retained in the JSON.

| API | Input | Layout / workload | Workers | Original ms | C++ ms | Speedup |
|---|---|---|---:|---:|---:|---:|
| color.rgba2rgb | bool | contiguous | 1 | 14.3089 | 0.5491 | 26.06x |
| color.rgba2rgb | bool | reversed | 1 | 15.1842 | 2.0612 | 7.37x |
| color.rgba2rgb | bool | strided | 1 | 14.5832 | 2.1232 | 6.87x |
| color.rgba2rgb | bool | fortran | 1 | 12.2051 | 2.0558 | 5.94x |
| color.rgba2rgb | bool | planar | 1 | 8.5086 | 2.0561 | 4.14x |
| color.rgba2rgb | uint8 | contiguous | 1 | 14.5427 | 0.5894 | 24.67x |
| color.rgba2rgb | uint8 | reversed | 1 | 15.1795 | 2.0498 | 7.41x |
| color.rgba2rgb | uint8 | strided | 1 | 15.2253 | 2.0584 | 7.40x |
| color.rgba2rgb | uint8 | fortran | 1 | 11.5524 | 2.0732 | 5.57x |
| color.rgba2rgb | uint8 | planar | 1 | 11.7638 | 2.0669 | 5.69x |
| color.rgba2rgb | int8 | contiguous | 1 | 17.5158 | 1.9186 | 9.13x |
| color.rgba2rgb | int8 | reversed | 1 | 16.6573 | 2.5939 | 6.42x |
| color.rgba2rgb | int8 | strided | 1 | 17.6414 | 3.4177 | 5.16x |
| color.rgba2rgb | int8 | fortran | 1 | 12.9366 | 2.4743 | 5.23x |
| color.rgba2rgb | int8 | planar | 1 | 9.3619 | 3.2758 | 2.86x |
| color.rgba2rgb | uint16 | contiguous | 1 | 15.3792 | 0.7015 | 21.92x |
| color.rgba2rgb | uint16 | reversed | 1 | 17.9278 | 2.1937 | 8.17x |
| color.rgba2rgb | uint16 | strided | 1 | 16.4948 | 2.2536 | 7.32x |
| color.rgba2rgb | uint16 | fortran | 1 | 13.1984 | 2.1824 | 6.05x |
| color.rgba2rgb | uint16 | planar | 1 | 9.3974 | 2.0879 | 4.50x |
| color.rgba2rgb | int16 | contiguous | 1 | 15.2909 | 0.7412 | 20.63x |
| color.rgba2rgb | int16 | reversed | 1 | 16.1655 | 2.4626 | 6.56x |
| color.rgba2rgb | int16 | strided | 1 | 15.5906 | 2.4532 | 6.36x |
| color.rgba2rgb | int16 | fortran | 1 | 15.6787 | 2.5254 | 6.21x |
| color.rgba2rgb | int16 | planar | 1 | 9.9412 | 2.4177 | 4.11x |
| color.rgba2rgb | uint32 | contiguous | 1 | 14.6333 | 2.1174 | 6.91x |
| color.rgba2rgb | uint32 | reversed | 1 | 16.2216 | 1.9054 | 8.51x |
| color.rgba2rgb | uint32 | strided | 1 | 16.3078 | 2.1896 | 7.45x |
| color.rgba2rgb | uint32 | fortran | 1 | 11.7275 | 1.9352 | 6.06x |
| color.rgba2rgb | uint32 | planar | 1 | 8.5783 | 1.8236 | 4.70x |
| color.rgba2rgb | int32 | contiguous | 1 | 14.2473 | 2.3886 | 5.96x |
| color.rgba2rgb | int32 | reversed | 1 | 14.6819 | 2.4053 | 6.10x |
| color.rgba2rgb | int32 | strided | 1 | 14.2113 | 2.5124 | 5.66x |
| color.rgba2rgb | int32 | fortran | 1 | 11.3797 | 2.4520 | 4.64x |
| color.rgba2rgb | int32 | planar | 1 | 9.6106 | 2.4673 | 3.90x |
| color.rgba2rgb | uint64 | contiguous | 1 | 19.3550 | 4.7207 | 4.10x |
| color.rgba2rgb | uint64 | reversed | 1 | 19.1218 | 4.6093 | 4.15x |
| color.rgba2rgb | uint64 | strided | 1 | 18.0646 | 4.3033 | 4.20x |
| color.rgba2rgb | uint64 | fortran | 1 | 15.8176 | 4.6197 | 3.42x |
| color.rgba2rgb | uint64 | planar | 1 | 12.1570 | 4.6063 | 2.64x |
| color.rgba2rgb | int64 | contiguous | 1 | 15.3418 | 2.6801 | 5.72x |
| color.rgba2rgb | int64 | reversed | 1 | 15.9488 | 2.6684 | 5.98x |
| color.rgba2rgb | int64 | strided | 1 | 16.2582 | 2.8487 | 5.71x |
| color.rgba2rgb | int64 | fortran | 1 | 13.9032 | 2.5863 | 5.38x |
| color.rgba2rgb | int64 | planar | 1 | 10.6122 | 2.5961 | 4.09x |
| color.rgba2rgb | float32 | contiguous | 1 | 9.4226 | 0.5915 | 15.93x |
| color.rgba2rgb | float32 | reversed | 1 | 7.0162 | 1.6527 | 4.25x |
| color.rgba2rgb | float32 | strided | 1 | 9.9629 | 2.4221 | 4.11x |
| color.rgba2rgb | float32 | fortran | 1 | 7.2835 | 0.5624 | 12.95x |
| color.rgba2rgb | float32 | planar | 1 | 3.8601 | 0.5888 | 6.56x |
| color.rgba2rgb | float64 | contiguous | 1 | 12.2623 | 1.1001 | 11.15x |
| color.rgba2rgb | float64 | reversed | 1 | 12.8674 | 4.6298 | 2.78x |
| color.rgba2rgb | float64 | strided | 1 | 12.9438 | 6.0564 | 2.14x |
| color.rgba2rgb | float64 | fortran | 1 | 12.0099 | 1.5303 | 7.85x |
| color.rgba2rgb | float64 | planar | 1 | 8.1580 | 1.0681 | 7.64x |
| color.gray2rgb | bool | contiguous | 1 | 0.2913 | 0.0843 | 3.45x |
| color.gray2rgba | bool | contiguous | 1 | 0.2670 | 0.0681 | 3.92x |
| color.gray2rgb | bool | reversed | 1 | 0.3499 | 0.2506 | 1.40x |
| color.gray2rgba | bool | reversed | 1 | 0.2518 | 0.0725 | 3.47x |
| color.gray2rgb | bool | strided | 1 | 0.3614 | 0.2582 | 1.40x |
| color.gray2rgba | bool | strided | 1 | 0.2861 | 0.0854 | 3.35x |
| color.gray2rgb | bool | fortran | 1 | 0.3366 | 0.2364 | 1.42x |
| color.gray2rgba | bool | fortran | 1 | 0.4731 | 0.0683 | 6.93x |
| color.gray2rgb | uint8 | contiguous | 1 | 0.3424 | 0.2307 | 1.48x |
| color.gray2rgba | uint8 | contiguous | 1 | 0.2583 | 0.0735 | 3.52x |
| color.gray2rgb | uint8 | reversed | 1 | 0.3359 | 0.2324 | 1.45x |
| color.gray2rgba | uint8 | reversed | 1 | 0.2467 | 0.0723 | 3.41x |
| color.gray2rgb | uint8 | strided | 1 | 0.3638 | 0.2497 | 1.46x |
| color.gray2rgba | uint8 | strided | 1 | 0.3199 | 0.0919 | 3.48x |
| color.gray2rgb | uint8 | fortran | 1 | 0.3703 | 0.2623 | 1.41x |
| color.gray2rgba | uint8 | fortran | 1 | 0.5058 | 0.0788 | 6.42x |
| color.gray2rgb | int8 | contiguous | 1 | 0.3823 | 0.2569 | 1.49x |
| color.gray2rgba | int8 | contiguous | 1 | 0.2705 | 0.0770 | 3.51x |
| color.gray2rgb | int8 | reversed | 1 | 0.3925 | 0.2754 | 1.43x |
| color.gray2rgba | int8 | reversed | 1 | 0.2850 | 0.0819 | 3.48x |
| color.gray2rgb | int8 | strided | 1 | 0.3806 | 0.2513 | 1.51x |
| color.gray2rgba | int8 | strided | 1 | 0.4090 | 0.2259 | 1.81x |
| color.gray2rgb | int8 | fortran | 1 | 0.4669 | 0.2622 | 1.78x |
| color.gray2rgba | int8 | fortran | 1 | 0.5000 | 0.0971 | 5.15x |
| color.gray2rgb | uint16 | contiguous | 1 | 0.1933 | 0.1430 | 1.35x |
| color.gray2rgba | uint16 | contiguous | 1 | 1.0256 | 0.6521 | 1.57x |
| color.gray2rgb | uint16 | reversed | 1 | 1.1094 | 0.7456 | 1.49x |
| color.gray2rgba | uint16 | reversed | 1 | 0.4259 | 0.2640 | 1.61x |
| color.gray2rgb | uint16 | strided | 1 | 0.9431 | 0.5942 | 1.59x |
| color.gray2rgba | uint16 | strided | 1 | 0.4377 | 0.1925 | 2.27x |
| color.gray2rgb | uint16 | fortran | 1 | 0.6079 | 0.4940 | 1.23x |
| color.gray2rgba | uint16 | fortran | 1 | 0.6116 | 0.1345 | 4.55x |
| color.gray2rgb | int16 | contiguous | 1 | 0.1999 | 0.1923 | 1.04x |
| color.gray2rgba | int16 | contiguous | 1 | 0.3416 | 0.1422 | 2.40x |
| color.gray2rgb | int16 | reversed | 1 | 0.6575 | 0.5167 | 1.27x |
| color.gray2rgba | int16 | reversed | 1 | 0.6471 | 0.3099 | 2.09x |
| color.gray2rgb | int16 | strided | 1 | 0.4546 | 0.3833 | 1.19x |
| color.gray2rgba | int16 | strided | 1 | 0.4909 | 0.3030 | 1.62x |
| color.gray2rgb | int16 | fortran | 1 | 0.1978 | 0.1458 | 1.36x |
| color.gray2rgba | int16 | fortran | 1 | 0.6522 | 0.1416 | 4.60x |
| color.gray2rgb | uint32 | contiguous | 1 | 0.3731 | 0.2434 | 1.53x |
| color.gray2rgba | uint32 | contiguous | 1 | 0.5299 | 0.2288 | 2.32x |
| color.gray2rgb | uint32 | reversed | 1 | 1.0619 | 0.7347 | 1.45x |
| color.gray2rgba | uint32 | reversed | 1 | 0.6063 | 0.2657 | 2.28x |
| color.gray2rgb | uint32 | strided | 1 | 1.5776 | 0.8875 | 1.78x |
| color.gray2rgba | uint32 | strided | 1 | 0.6908 | 0.3272 | 2.11x |
| color.gray2rgb | uint32 | fortran | 1 | 1.2216 | 0.9557 | 1.28x |
| color.gray2rgba | uint32 | fortran | 1 | 1.0269 | 0.2512 | 4.09x |
| color.gray2rgb | int32 | contiguous | 1 | 0.3018 | 0.1516 | 1.99x |
| color.gray2rgba | int32 | contiguous | 1 | 0.5542 | 0.2408 | 2.30x |
| color.gray2rgb | int32 | reversed | 1 | 0.3154 | 0.1491 | 2.12x |
| color.gray2rgba | int32 | reversed | 1 | 0.4964 | 0.2185 | 2.27x |
| color.gray2rgb | int32 | strided | 1 | 0.3658 | 0.1931 | 1.89x |
| color.gray2rgba | int32 | strided | 1 | 0.6235 | 0.2838 | 2.20x |
| color.gray2rgb | int32 | fortran | 1 | 0.3016 | 0.1579 | 1.91x |
| color.gray2rgba | int32 | fortran | 1 | 0.8982 | 0.2377 | 3.78x |
| color.gray2rgb | uint64 | contiguous | 1 | 2.2887 | 1.6405 | 1.40x |
| color.gray2rgba | uint64 | contiguous | 1 | 3.8963 | 2.1897 | 1.78x |
| color.gray2rgb | uint64 | reversed | 1 | 2.0225 | 1.4687 | 1.38x |
| color.gray2rgba | uint64 | reversed | 1 | 3.4422 | 2.1147 | 1.63x |
| color.gray2rgb | uint64 | strided | 1 | 2.6021 | 1.6188 | 1.61x |
| color.gray2rgba | uint64 | strided | 1 | 4.2439 | 2.3983 | 1.77x |
| color.gray2rgb | uint64 | fortran | 1 | 2.4866 | 1.8510 | 1.34x |
| color.gray2rgba | uint64 | fortran | 1 | 8.4668 | 2.8136 | 3.01x |
| color.gray2rgb | int64 | contiguous | 1 | 2.8028 | 1.9010 | 1.47x |
| color.gray2rgba | int64 | contiguous | 1 | 3.8099 | 2.8035 | 1.36x |
| color.gray2rgb | int64 | reversed | 1 | 2.1671 | 0.4780 | 4.53x |
| color.gray2rgba | int64 | reversed | 1 | 3.5589 | 2.2447 | 1.59x |
| color.gray2rgb | int64 | strided | 1 | 2.4705 | 1.7775 | 1.39x |
| color.gray2rgba | int64 | strided | 1 | 9.4363 | 5.2624 | 1.79x |
| color.gray2rgb | int64 | fortran | 1 | 6.7562 | 4.5335 | 1.49x |
| color.gray2rgba | int64 | fortran | 1 | 12.4533 | 3.7477 | 3.32x |
| color.gray2rgb | float32 | contiguous | 1 | 0.3926 | 0.1815 | 2.16x |
| color.gray2rgba | float32 | contiguous | 1 | 0.5840 | 0.2663 | 2.19x |
| color.gray2rgb | float32 | reversed | 1 | 0.3164 | 0.1659 | 1.91x |
| color.gray2rgba | float32 | reversed | 1 | 0.5505 | 0.3105 | 1.77x |
| color.gray2rgb | float32 | strided | 1 | 0.4339 | 0.2025 | 2.14x |
| color.gray2rgba | float32 | strided | 1 | 0.6904 | 0.3111 | 2.22x |
| color.gray2rgb | float32 | fortran | 1 | 0.3167 | 0.1615 | 1.96x |
| color.gray2rgba | float32 | fortran | 1 | 0.9036 | 0.2600 | 3.48x |
| color.gray2rgb | float64 | contiguous | 1 | 2.3074 | 1.7576 | 1.31x |
| color.gray2rgba | float64 | contiguous | 1 | 3.7042 | 2.2449 | 1.65x |
| color.gray2rgb | float64 | reversed | 1 | 2.9240 | 1.7933 | 1.63x |
| color.gray2rgba | float64 | reversed | 1 | 3.9733 | 2.5585 | 1.55x |
| color.gray2rgb | float64 | strided | 1 | 3.5673 | 2.2012 | 1.62x |
| color.gray2rgba | float64 | strided | 1 | 3.3835 | 2.2571 | 1.50x |
| color.gray2rgb | float64 | fortran | 1 | 2.1425 | 0.4492 | 4.77x |
| color.gray2rgba | float64 | fortran | 1 | 5.1573 | 2.0977 | 2.46x |
| color.gray2rgb | float16 | contiguous | 1 | 0.1969 | 0.1391 | 1.42x |
| color.gray2rgba | float16 | contiguous | 1 | 0.7025 | 0.5273 | 1.33x |
| color.gray2rgb | float16 | reversed | 1 | 0.4587 | 0.1500 | 3.06x |
| color.gray2rgba | float16 | reversed | 1 | 0.8039 | 0.6089 | 1.32x |
| color.gray2rgb | float16 | strided | 1 | 0.5533 | 0.4910 | 1.13x |
| color.gray2rgba | float16 | strided | 1 | 0.9604 | 0.7135 | 1.35x |
| color.gray2rgb | float16 | fortran | 1 | 0.5397 | 0.4221 | 1.28x |
| color.gray2rgba | float16 | fortran | 1 | 1.0428 | 0.5524 | 1.89x |
| color.rgba2rgb | uint8 | background-black | 1 | 14.4853 | 0.6241 | 23.21x |
| color.rgba2rgb | uint8 | background-color | 1 | 14.2866 | 0.6165 | 23.17x |
| color.rgba2rgb | uint8 | background-nan | 1 | 14.8547 | 0.6141 | 24.19x |
| color.gray2rgba | uint8 | alpha-scalar | 1 | 0.2722 | 0.0736 | 3.70x |
| color.gray2rgba | uint8 | alpha-typed-scalar | 1 | 0.2591 | 0.0662 | 3.92x |
| color.gray2rgba | uint8 | alpha-array | 1 | 0.2956 | 0.0608 | 4.86x |
| color.gray2rgba | uint8 | alpha-broadcast | 1 | 0.2779 | 0.0619 | 4.49x |
| color.rgba2rgb | float32 | background-black | 1 | 8.9147 | 0.6365 | 14.01x |
| color.rgba2rgb | float32 | background-color | 1 | 8.5757 | 0.6248 | 13.73x |
| color.rgba2rgb | float32 | background-nan | 1 | 9.4272 | 0.7406 | 12.73x |
| color.gray2rgba | float32 | alpha-scalar | 1 | 0.5262 | 0.2288 | 2.30x |
| color.gray2rgba | float32 | alpha-typed-scalar | 1 | 0.5425 | 0.2271 | 2.39x |
| color.gray2rgba | float32 | alpha-array | 1 | 0.6746 | 0.6777 | 1.00x |
| color.gray2rgba | float32 | alpha-broadcast | 1 | 0.6477 | 0.2397 | 2.70x |
| color.rgba2rgb | float64 | background-black | 1 | 12.4038 | 1.1317 | 10.96x |
| color.rgba2rgb | float64 | background-color | 1 | 13.5681 | 1.4525 | 9.34x |
| color.rgba2rgb | float64 | background-nan | 1 | 12.8691 | 1.2467 | 10.32x |
| color.gray2rgba | float64 | alpha-scalar | 1 | 3.7982 | 2.1967 | 1.73x |
| color.gray2rgba | float64 | alpha-typed-scalar | 1 | 3.2383 | 2.2361 | 1.45x |
| color.gray2rgba | float64 | alpha-array | 1 | 4.6556 | 2.6939 | 1.73x |
| color.gray2rgba | float64 | alpha-broadcast | 1 | 3.8229 | 2.2565 | 1.69x |
| color.gray2rgb | >u2 | endian-contiguous | 1 | 0.2213 | 0.1412 | 1.57x |
| color.gray2rgba | >u2 | endian-contiguous | 1 | 0.3039 | 0.1243 | 2.45x |
| color.gray2rgb | >u2 | endian-reversed | 1 | 0.5544 | 0.5652 | 0.98x |
| color.gray2rgba | >u2 | endian-reversed | 1 | 0.3036 | 0.1222 | 2.48x |
| color.gray2rgb | >f4 | endian-contiguous | 1 | 0.2987 | 0.2018 | 1.48x |
| color.gray2rgba | >f4 | endian-contiguous | 1 | 0.5208 | 0.3699 | 1.41x |
| color.gray2rgb | >f4 | endian-reversed | 1 | 0.2950 | 0.1977 | 1.49x |
| color.gray2rgba | >f4 | endian-reversed | 1 | 0.5167 | 0.3659 | 1.41x |
| color.gray2rgb | >f8 | endian-contiguous | 1 | 0.8350 | 0.6407 | 1.30x |
| color.gray2rgba | >f8 | endian-contiguous | 1 | 4.4502 | 2.8121 | 1.58x |
| color.gray2rgb | >f8 | endian-reversed | 1 | 0.9473 | 0.5551 | 1.71x |
| color.gray2rgba | >f8 | endian-reversed | 1 | 5.1572 | 2.8833 | 1.79x |
| color.gray2rgb | [('x', '>i2'), ('y', '>f4')] | record-6-contiguous | 1 | 1.1854 | 0.2456 | 4.83x |
| color.gray2rgba | [('x', '>i2'), ('y', '>f4')] | record-6-contiguous | 1 | 4.4057 | 2.8301 | 1.56x |
| color.gray2rgb | [('x', '>i2'), ('y', '>f4')] | record-6-reversed | 1 | 1.1069 | 0.2238 | 4.95x |
| color.gray2rgba | [('x', '>i2'), ('y', '>f4')] | record-6-reversed | 1 | 8.6465 | 3.6618 | 2.36x |
| color.gray2rgb | [('x', '>i2'), ('y', '>f4')] | record-6-strided | 1 | 1.9668 | 0.9136 | 2.15x |
| color.gray2rgba | [('x', '>i2'), ('y', '>f4')] | record-6-strided | 1 | 9.0453 | 2.9009 | 3.12x |
| color.gray2rgb | [('x', '>i2'), ('y', '>f4')] | record-6-fortran | 1 | 1.1343 | 0.2085 | 5.44x |
| color.gray2rgba | [('x', '>i2'), ('y', '>f4')] | record-6-fortran | 1 | 5.6156 | 2.3934 | 2.35x |
| color.gray2rgb | [('record', [('a', '>i2'), ('b', '>u4')]), ('vec', '>f4', (3,))] | record-18-contiguous | 1 | 13.0989 | 7.8370 | 1.67x |
| color.gray2rgba | [('record', [('a', '>i2'), ('b', '>u4')]), ('vec', '>f4', (3,))] | record-18-contiguous | 1 | 26.0455 | 16.9726 | 1.53x |
| color.gray2rgb | [('record', [('a', '>i2'), ('b', '>u4')]), ('vec', '>f4', (3,))] | record-18-reversed | 1 | 13.6832 | 7.2551 | 1.89x |
| color.gray2rgba | [('record', [('a', '>i2'), ('b', '>u4')]), ('vec', '>f4', (3,))] | record-18-reversed | 1 | 26.3664 | 17.2508 | 1.53x |
| color.gray2rgb | [('record', [('a', '>i2'), ('b', '>u4')]), ('vec', '>f4', (3,))] | record-18-strided | 1 | 12.8274 | 7.5227 | 1.71x |
| color.gray2rgba | [('record', [('a', '>i2'), ('b', '>u4')]), ('vec', '>f4', (3,))] | record-18-strided | 1 | 34.0746 | 16.5207 | 2.06x |
| color.gray2rgb | [('record', [('a', '>i2'), ('b', '>u4')]), ('vec', '>f4', (3,))] | record-18-fortran | 1 | 11.6183 | 6.6587 | 1.74x |
| color.gray2rgba | [('record', [('a', '>i2'), ('b', '>u4')]), ('vec', '>f4', (3,))] | record-18-fortran | 1 | 25.3053 | 16.1584 | 1.57x |
| color.gray2rgb | [('z', '>c16'), ('when', '>M8[ns]'), ('dt', '>m8[ms]')] | record-32-contiguous | 1 | 19.7773 | 9.2947 | 2.13x |
| color.gray2rgba | [('z', '>c16'), ('when', '>M8[ns]'), ('dt', '>m8[ms]')] | record-32-contiguous | 1 | 35.4384 | 19.1956 | 1.85x |
| color.gray2rgb | [('z', '>c16'), ('when', '>M8[ns]'), ('dt', '>m8[ms]')] | record-32-reversed | 1 | 17.2701 | 7.3448 | 2.35x |
| color.gray2rgba | [('z', '>c16'), ('when', '>M8[ns]'), ('dt', '>m8[ms]')] | record-32-reversed | 1 | 34.0244 | 18.0591 | 1.88x |
| color.gray2rgb | [('z', '>c16'), ('when', '>M8[ns]'), ('dt', '>m8[ms]')] | record-32-strided | 1 | 17.4680 | 8.7613 | 1.99x |
| color.gray2rgba | [('z', '>c16'), ('when', '>M8[ns]'), ('dt', '>m8[ms]')] | record-32-strided | 1 | 43.8515 | 20.3525 | 2.15x |
| color.gray2rgb | [('z', '>c16'), ('when', '>M8[ns]'), ('dt', '>m8[ms]')] | record-32-fortran | 1 | 16.7537 | 8.0395 | 2.08x |
| color.gray2rgba | [('z', '>c16'), ('when', '>M8[ns]'), ('dt', '>m8[ms]')] | record-32-fortran | 1 | 35.2596 | 17.5816 | 2.01x |
