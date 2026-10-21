# Alpha and grayscale performance

Original scikit-image 0.26.0. Reference thread pools use one worker; native workers are explicit. Timings include dispatch, integer normalization, validation, layout handling and output allocation. Input construction is excluded. All samples and slower cases are retained in the JSON.

| API | Input | Layout / workload | Workers | Original ms | C++ ms | Speedup |
|---|---|---|---:|---:|---:|---:|
| color.rgba2rgb | bool | contiguous | 1 | 27.9356 | 2.3508 | 11.88x |
| color.rgba2rgb | bool | reversed | 1 | 29.3551 | 3.2209 | 9.11x |
| color.rgba2rgb | bool | strided | 1 | 30.2639 | 3.0592 | 9.89x |
| color.rgba2rgb | bool | fortran | 1 | 25.9893 | 3.3478 | 7.76x |
| color.rgba2rgb | bool | planar | 1 | 21.0408 | 3.3776 | 6.23x |
| color.rgba2rgb | uint8 | contiguous | 1 | 28.1171 | 2.2926 | 12.26x |
| color.rgba2rgb | uint8 | reversed | 1 | 31.5764 | 3.1391 | 10.06x |
| color.rgba2rgb | uint8 | strided | 1 | 30.1986 | 3.2541 | 9.28x |
| color.rgba2rgb | uint8 | fortran | 1 | 25.0118 | 3.2756 | 7.64x |
| color.rgba2rgb | uint8 | planar | 1 | 21.3066 | 3.3973 | 6.27x |
| color.rgba2rgb | int8 | contiguous | 1 | 31.1621 | 3.9722 | 7.85x |
| color.rgba2rgb | int8 | reversed | 1 | 34.3058 | 4.6998 | 7.30x |
| color.rgba2rgb | int8 | strided | 1 | 31.0917 | 4.1686 | 7.46x |
| color.rgba2rgb | int8 | fortran | 1 | 30.5755 | 4.1102 | 7.44x |
| color.rgba2rgb | int8 | planar | 1 | 29.3346 | 3.9857 | 7.36x |
| color.rgba2rgb | uint16 | contiguous | 1 | 32.8022 | 1.6109 | 20.36x |
| color.rgba2rgb | uint16 | reversed | 1 | 34.3920 | 4.4118 | 7.80x |
| color.rgba2rgb | uint16 | strided | 1 | 33.7025 | 4.0396 | 8.34x |
| color.rgba2rgb | uint16 | fortran | 1 | 29.1704 | 4.1337 | 7.06x |
| color.rgba2rgb | uint16 | planar | 1 | 21.2930 | 3.8944 | 5.47x |
| color.rgba2rgb | int16 | contiguous | 1 | 37.0142 | 2.3797 | 15.55x |
| color.rgba2rgb | int16 | reversed | 1 | 33.4632 | 3.5756 | 9.36x |
| color.rgba2rgb | int16 | strided | 1 | 39.9503 | 3.9375 | 10.15x |
| color.rgba2rgb | int16 | fortran | 1 | 31.1844 | 3.1901 | 9.78x |
| color.rgba2rgb | int16 | planar | 1 | 29.0788 | 3.2580 | 8.93x |
| color.rgba2rgb | uint32 | contiguous | 1 | 25.7619 | 3.6677 | 7.02x |
| color.rgba2rgb | uint32 | reversed | 1 | 20.5147 | 2.5227 | 8.13x |
| color.rgba2rgb | uint32 | strided | 1 | 21.1690 | 3.0305 | 6.99x |
| color.rgba2rgb | uint32 | fortran | 1 | 16.0407 | 3.1567 | 5.08x |
| color.rgba2rgb | uint32 | planar | 1 | 12.8920 | 2.4819 | 5.19x |
| color.rgba2rgb | int32 | contiguous | 1 | 19.1825 | 2.6886 | 7.13x |
| color.rgba2rgb | int32 | reversed | 1 | 19.4049 | 2.7819 | 6.98x |
| color.rgba2rgb | int32 | strided | 1 | 18.6493 | 3.0755 | 6.06x |
| color.rgba2rgb | int32 | fortran | 1 | 18.3219 | 2.9311 | 6.25x |
| color.rgba2rgb | int32 | planar | 1 | 13.7426 | 2.6258 | 5.23x |
| color.rgba2rgb | uint64 | contiguous | 1 | 24.5171 | 6.3872 | 3.84x |
| color.rgba2rgb | uint64 | reversed | 1 | 24.2527 | 5.2347 | 4.63x |
| color.rgba2rgb | uint64 | strided | 1 | 23.9113 | 5.3209 | 4.49x |
| color.rgba2rgb | uint64 | fortran | 1 | 21.1930 | 6.1921 | 3.42x |
| color.rgba2rgb | uint64 | planar | 1 | 16.7698 | 6.3224 | 2.65x |
| color.rgba2rgb | int64 | contiguous | 1 | 17.9036 | 2.8625 | 6.25x |
| color.rgba2rgb | int64 | reversed | 1 | 18.5687 | 2.8432 | 6.53x |
| color.rgba2rgb | int64 | strided | 1 | 19.6715 | 3.9690 | 4.96x |
| color.rgba2rgb | int64 | fortran | 1 | 16.5321 | 3.0723 | 5.38x |
| color.rgba2rgb | int64 | planar | 1 | 13.7359 | 2.8392 | 4.84x |
| color.rgba2rgb | float32 | contiguous | 1 | 10.4500 | 0.8317 | 12.56x |
| color.rgba2rgb | float32 | reversed | 1 | 10.9027 | 2.4238 | 4.50x |
| color.rgba2rgb | float32 | strided | 1 | 10.3942 | 3.4642 | 3.00x |
| color.rgba2rgb | float32 | fortran | 1 | 8.4974 | 0.8966 | 9.48x |
| color.rgba2rgb | float32 | planar | 1 | 3.5607 | 0.8377 | 4.25x |
| color.rgba2rgb | float64 | contiguous | 1 | 16.9493 | 1.6628 | 10.19x |
| color.rgba2rgb | float64 | reversed | 1 | 20.1733 | 7.4524 | 2.71x |
| color.rgba2rgb | float64 | strided | 1 | 23.1266 | 12.6054 | 1.83x |
| color.rgba2rgb | float64 | fortran | 1 | 14.7806 | 3.0996 | 4.77x |
| color.rgba2rgb | float64 | planar | 1 | 12.2742 | 3.0296 | 4.05x |
| color.gray2rgb | bool | contiguous | 1 | 0.5874 | 0.3891 | 1.51x |
| color.gray2rgba | bool | contiguous | 1 | 0.4482 | 0.1676 | 2.67x |
| color.gray2rgb | bool | reversed | 1 | 0.6440 | 0.3997 | 1.61x |
| color.gray2rgba | bool | reversed | 1 | 0.4237 | 0.1885 | 2.25x |
| color.gray2rgb | bool | strided | 1 | 0.7661 | 0.4946 | 1.55x |
| color.gray2rgba | bool | strided | 1 | 0.5478 | 0.1945 | 2.82x |
| color.gray2rgb | bool | fortran | 1 | 0.5735 | 0.3512 | 1.63x |
| color.gray2rgba | bool | fortran | 1 | 0.5663 | 0.1309 | 4.33x |
| color.gray2rgb | uint8 | contiguous | 1 | 0.4993 | 0.2976 | 1.68x |
| color.gray2rgba | uint8 | contiguous | 1 | 0.2940 | 0.0922 | 3.19x |
| color.gray2rgb | uint8 | reversed | 1 | 0.4474 | 0.3249 | 1.38x |
| color.gray2rgba | uint8 | reversed | 1 | 0.3749 | 0.1286 | 2.92x |
| color.gray2rgb | uint8 | strided | 1 | 0.4281 | 0.3363 | 1.27x |
| color.gray2rgba | uint8 | strided | 1 | 0.4259 | 0.1738 | 2.45x |
| color.gray2rgb | uint8 | fortran | 1 | 0.4229 | 0.2605 | 1.62x |
| color.gray2rgba | uint8 | fortran | 1 | 0.5815 | 0.0885 | 6.57x |
| color.gray2rgb | int8 | contiguous | 1 | 0.3904 | 0.2487 | 1.57x |
| color.gray2rgba | int8 | contiguous | 1 | 0.3006 | 0.0764 | 3.93x |
| color.gray2rgb | int8 | reversed | 1 | 0.3954 | 0.3025 | 1.31x |
| color.gray2rgba | int8 | reversed | 1 | 0.3323 | 0.0972 | 3.42x |
| color.gray2rgb | int8 | strided | 1 | 0.5347 | 0.3851 | 1.39x |
| color.gray2rgba | int8 | strided | 1 | 0.5590 | 0.1736 | 3.22x |
| color.gray2rgb | int8 | fortran | 1 | 0.4366 | 0.3139 | 1.39x |
| color.gray2rgba | int8 | fortran | 1 | 0.5590 | 0.1090 | 5.13x |
| color.gray2rgb | uint16 | contiguous | 1 | 0.2278 | 0.1780 | 1.28x |
| color.gray2rgba | uint16 | contiguous | 1 | 0.9033 | 0.7602 | 1.19x |
| color.gray2rgb | uint16 | reversed | 1 | 0.2942 | 0.2056 | 1.43x |
| color.gray2rgba | uint16 | reversed | 1 | 0.3485 | 0.2163 | 1.61x |
| color.gray2rgb | uint16 | strided | 1 | 0.3538 | 0.1950 | 1.81x |
| color.gray2rgba | uint16 | strided | 1 | 0.6336 | 0.3063 | 2.07x |
| color.gray2rgb | uint16 | fortran | 1 | 0.2373 | 0.1999 | 1.19x |
| color.gray2rgba | uint16 | fortran | 1 | 0.8392 | 0.2241 | 3.75x |
| color.gray2rgb | int16 | contiguous | 1 | 0.2960 | 0.1786 | 1.66x |
| color.gray2rgba | int16 | contiguous | 1 | 0.3878 | 0.1946 | 1.99x |
| color.gray2rgb | int16 | reversed | 1 | 0.2849 | 0.2062 | 1.38x |
| color.gray2rgba | int16 | reversed | 1 | 0.3924 | 0.2166 | 1.81x |
| color.gray2rgb | int16 | strided | 1 | 0.3184 | 0.2656 | 1.20x |
| color.gray2rgba | int16 | strided | 1 | 0.4613 | 0.2759 | 1.67x |
| color.gray2rgb | int16 | fortran | 1 | 0.2676 | 0.2071 | 1.29x |
| color.gray2rgba | int16 | fortran | 1 | 0.6796 | 0.1880 | 3.61x |
| color.gray2rgb | uint32 | contiguous | 1 | 1.3019 | 1.0940 | 1.19x |
| color.gray2rgba | uint32 | contiguous | 1 | 0.8374 | 0.3665 | 2.28x |
| color.gray2rgb | uint32 | reversed | 1 | 1.0225 | 1.0410 | 0.98x |
| color.gray2rgba | uint32 | reversed | 1 | 0.7183 | 0.3709 | 1.94x |
| color.gray2rgb | uint32 | strided | 1 | 1.4729 | 1.1215 | 1.31x |
| color.gray2rgba | uint32 | strided | 1 | 0.8839 | 0.4375 | 2.02x |
| color.gray2rgb | uint32 | fortran | 1 | 1.3249 | 1.0418 | 1.27x |
| color.gray2rgba | uint32 | fortran | 1 | 1.2518 | 0.3725 | 3.36x |
| color.gray2rgb | int32 | contiguous | 1 | 0.4598 | 0.2708 | 1.70x |
| color.gray2rgba | int32 | contiguous | 1 | 0.8510 | 0.5126 | 1.66x |
| color.gray2rgb | int32 | reversed | 1 | 0.5704 | 0.2278 | 2.50x |
| color.gray2rgba | int32 | reversed | 1 | 0.7769 | 0.4105 | 1.89x |
| color.gray2rgb | int32 | strided | 1 | 0.5583 | 0.2979 | 1.87x |
| color.gray2rgba | int32 | strided | 1 | 0.8658 | 0.4366 | 1.98x |
| color.gray2rgb | int32 | fortran | 1 | 0.3761 | 0.2125 | 1.77x |
| color.gray2rgba | int32 | fortran | 1 | 1.1216 | 0.3688 | 3.04x |
| color.gray2rgb | uint64 | contiguous | 1 | 3.2274 | 2.2881 | 1.41x |
| color.gray2rgba | uint64 | contiguous | 1 | 6.4824 | 3.6712 | 1.77x |
| color.gray2rgb | uint64 | reversed | 1 | 3.1364 | 0.8168 | 3.84x |
| color.gray2rgba | uint64 | reversed | 1 | 6.7564 | 3.6507 | 1.85x |
| color.gray2rgb | uint64 | strided | 1 | 1.8795 | 1.4261 | 1.32x |
| color.gray2rgba | uint64 | strided | 1 | 9.8000 | 3.8257 | 2.56x |
| color.gray2rgb | uint64 | fortran | 1 | 2.0361 | 1.4007 | 1.45x |
| color.gray2rgba | uint64 | fortran | 1 | 12.7517 | 4.0767 | 3.13x |
| color.gray2rgb | int64 | contiguous | 1 | 3.8659 | 2.4379 | 1.59x |
| color.gray2rgba | int64 | contiguous | 1 | 6.3196 | 3.2614 | 1.94x |
| color.gray2rgb | int64 | reversed | 1 | 2.9835 | 2.0337 | 1.47x |
| color.gray2rgba | int64 | reversed | 1 | 4.6264 | 2.6803 | 1.73x |
| color.gray2rgb | int64 | strided | 1 | 3.4660 | 2.3922 | 1.45x |
| color.gray2rgba | int64 | strided | 1 | 6.9010 | 3.6594 | 1.89x |
| color.gray2rgb | int64 | fortran | 1 | 1.9945 | 0.7777 | 2.56x |
| color.gray2rgba | int64 | fortran | 1 | 13.8703 | 3.8557 | 3.60x |
| color.gray2rgb | float32 | contiguous | 1 | 0.4938 | 0.2269 | 2.18x |
| color.gray2rgba | float32 | contiguous | 1 | 0.6442 | 0.3374 | 1.91x |
| color.gray2rgb | float32 | reversed | 1 | 0.5185 | 0.3528 | 1.47x |
| color.gray2rgba | float32 | reversed | 1 | 0.8815 | 0.3952 | 2.23x |
| color.gray2rgb | float32 | strided | 1 | 0.5379 | 0.2947 | 1.83x |
| color.gray2rgba | float32 | strided | 1 | 5.7714 | 2.0596 | 2.80x |
| color.gray2rgb | float32 | fortran | 1 | 0.7259 | 0.3965 | 1.83x |
| color.gray2rgba | float32 | fortran | 1 | 2.9879 | 0.8980 | 3.33x |
| color.gray2rgb | float64 | contiguous | 1 | 1.8318 | 0.9175 | 2.00x |
| color.gray2rgba | float64 | contiguous | 1 | 10.7794 | 4.3173 | 2.50x |
| color.gray2rgb | float64 | reversed | 1 | 5.6280 | 3.0393 | 1.85x |
| color.gray2rgba | float64 | reversed | 1 | 5.6189 | 3.3871 | 1.66x |
| color.gray2rgb | float64 | strided | 1 | 5.9460 | 3.0830 | 1.93x |
| color.gray2rgba | float64 | strided | 1 | 10.4128 | 3.7915 | 2.75x |
| color.gray2rgb | float64 | fortran | 1 | 3.4556 | 2.3176 | 1.49x |
| color.gray2rgba | float64 | fortran | 1 | 10.6131 | 4.3936 | 2.42x |
| color.gray2rgb | float16 | contiguous | 1 | 0.2090 | 0.2078 | 1.01x |
| color.gray2rgba | float16 | contiguous | 1 | 1.0331 | 0.8360 | 1.24x |
| color.gray2rgb | float16 | reversed | 1 | 0.2537 | 0.1542 | 1.64x |
| color.gray2rgba | float16 | reversed | 1 | 1.0188 | 0.7422 | 1.37x |
| color.gray2rgb | float16 | strided | 1 | 0.3428 | 0.1891 | 1.81x |
| color.gray2rgba | float16 | strided | 1 | 1.4957 | 1.0509 | 1.42x |
| color.gray2rgb | float16 | fortran | 1 | 0.2165 | 0.1647 | 1.31x |
| color.gray2rgba | float16 | fortran | 1 | 1.6337 | 0.9253 | 1.77x |
| color.rgba2rgb | uint8 | background-black | 1 | 19.8658 | 1.2685 | 15.66x |
| color.rgba2rgb | uint8 | background-color | 1 | 25.6005 | 1.7591 | 14.55x |
| color.rgba2rgb | uint8 | background-nan | 1 | 28.6551 | 2.5176 | 11.38x |
| color.gray2rgba | uint8 | alpha-scalar | 1 | 0.4659 | 0.2166 | 2.15x |
| color.gray2rgba | uint8 | alpha-typed-scalar | 1 | 0.2812 | 0.1056 | 2.66x |
| color.gray2rgba | uint8 | alpha-array | 1 | 0.3562 | 0.0673 | 5.29x |
| color.gray2rgba | uint8 | alpha-broadcast | 1 | 0.2991 | 0.0753 | 3.97x |
| color.rgba2rgb | float32 | background-black | 1 | 10.5290 | 0.7559 | 13.93x |
| color.rgba2rgb | float32 | background-color | 1 | 10.0175 | 1.0313 | 9.71x |
| color.rgba2rgb | float32 | background-nan | 1 | 10.1773 | 1.1479 | 8.87x |
| color.gray2rgba | float32 | alpha-scalar | 1 | 0.6707 | 0.3579 | 1.87x |
| color.gray2rgba | float32 | alpha-typed-scalar | 1 | 0.6853 | 0.2996 | 2.29x |
| color.gray2rgba | float32 | alpha-array | 1 | 0.8732 | 0.7176 | 1.22x |
| color.gray2rgba | float32 | alpha-broadcast | 1 | 0.6997 | 0.2499 | 2.80x |
| color.rgba2rgb | float64 | background-black | 1 | 13.9186 | 1.2506 | 11.13x |
| color.rgba2rgb | float64 | background-color | 1 | 20.5118 | 4.1014 | 5.00x |
| color.rgba2rgb | float64 | background-nan | 1 | 20.3854 | 4.1641 | 4.90x |
| color.gray2rgba | float64 | alpha-scalar | 1 | 4.5232 | 2.9545 | 1.53x |
| color.gray2rgba | float64 | alpha-typed-scalar | 1 | 4.0735 | 2.4930 | 1.63x |
| color.gray2rgba | float64 | alpha-array | 1 | 4.3260 | 2.6598 | 1.63x |
| color.gray2rgba | float64 | alpha-broadcast | 1 | 4.1336 | 2.5708 | 1.61x |
| color.gray2rgb | >u2 | endian-contiguous | 1 | 0.2209 | 0.1394 | 1.59x |
| color.gray2rgba | >u2 | endian-contiguous | 1 | 0.3051 | 0.1250 | 2.44x |
| color.gray2rgb | >u2 | endian-reversed | 1 | 0.2314 | 0.1463 | 1.58x |
| color.gray2rgba | >u2 | endian-reversed | 1 | 0.3211 | 0.1299 | 2.47x |
| color.gray2rgb | >f4 | endian-contiguous | 1 | 0.3321 | 0.2692 | 1.23x |
| color.gray2rgba | >f4 | endian-contiguous | 1 | 0.5112 | 0.3705 | 1.38x |
| color.gray2rgb | >f4 | endian-reversed | 1 | 0.3040 | 0.2120 | 1.43x |
| color.gray2rgba | >f4 | endian-reversed | 1 | 0.5318 | 0.3610 | 1.47x |
| color.gray2rgb | >f8 | endian-contiguous | 1 | 2.7768 | 1.8886 | 1.47x |
| color.gray2rgba | >f8 | endian-contiguous | 1 | 3.5350 | 2.5556 | 1.38x |
| color.gray2rgb | >f8 | endian-reversed | 1 | 2.2416 | 1.7518 | 1.28x |
| color.gray2rgba | >f8 | endian-reversed | 1 | 3.3031 | 2.4362 | 1.36x |
| color.gray2rgb | [('x', '>i2'), ('y', '>f4')] | record-6-contiguous | 1 | 1.0188 | 0.1887 | 5.40x |
| color.gray2rgba | [('x', '>i2'), ('y', '>f4')] | record-6-contiguous | 1 | 3.8897 | 2.6346 | 1.48x |
| color.gray2rgb | [('x', '>i2'), ('y', '>f4')] | record-6-reversed | 1 | 0.9868 | 0.2952 | 3.34x |
| color.gray2rgba | [('x', '>i2'), ('y', '>f4')] | record-6-reversed | 1 | 5.4691 | 3.1181 | 1.75x |
| color.gray2rgb | [('x', '>i2'), ('y', '>f4')] | record-6-strided | 1 | 2.3984 | 0.7257 | 3.30x |
| color.gray2rgba | [('x', '>i2'), ('y', '>f4')] | record-6-strided | 1 | 5.9479 | 2.8121 | 2.12x |
| color.gray2rgb | [('x', '>i2'), ('y', '>f4')] | record-6-fortran | 1 | 1.0512 | 0.1876 | 5.60x |
| color.gray2rgba | [('x', '>i2'), ('y', '>f4')] | record-6-fortran | 1 | 4.4695 | 2.5702 | 1.74x |
| color.gray2rgb | [('record', [('a', '>i2'), ('b', '>u4')]), ('vec', '>f4', (3,))] | record-18-contiguous | 1 | 9.3095 | 6.2048 | 1.50x |
| color.gray2rgba | [('record', [('a', '>i2'), ('b', '>u4')]), ('vec', '>f4', (3,))] | record-18-contiguous | 1 | 18.4079 | 14.9384 | 1.23x |
| color.gray2rgb | [('record', [('a', '>i2'), ('b', '>u4')]), ('vec', '>f4', (3,))] | record-18-reversed | 1 | 8.2599 | 6.0261 | 1.37x |
| color.gray2rgba | [('record', [('a', '>i2'), ('b', '>u4')]), ('vec', '>f4', (3,))] | record-18-reversed | 1 | 23.9449 | 16.2788 | 1.47x |
| color.gray2rgb | [('record', [('a', '>i2'), ('b', '>u4')]), ('vec', '>f4', (3,))] | record-18-strided | 1 | 11.1920 | 6.9944 | 1.60x |
| color.gray2rgba | [('record', [('a', '>i2'), ('b', '>u4')]), ('vec', '>f4', (3,))] | record-18-strided | 1 | 27.1900 | 16.3363 | 1.66x |
| color.gray2rgb | [('record', [('a', '>i2'), ('b', '>u4')]), ('vec', '>f4', (3,))] | record-18-fortran | 1 | 8.4363 | 6.1707 | 1.37x |
| color.gray2rgba | [('record', [('a', '>i2'), ('b', '>u4')]), ('vec', '>f4', (3,))] | record-18-fortran | 1 | 18.1114 | 14.5432 | 1.25x |
| color.gray2rgb | [('z', '>c16'), ('when', '>M8[ns]'), ('dt', '>m8[ms]')] | record-32-contiguous | 1 | 12.9999 | 6.8051 | 1.91x |
| color.gray2rgba | [('z', '>c16'), ('when', '>M8[ns]'), ('dt', '>m8[ms]')] | record-32-contiguous | 1 | 29.8929 | 15.6873 | 1.91x |
| color.gray2rgb | [('z', '>c16'), ('when', '>M8[ns]'), ('dt', '>m8[ms]')] | record-32-reversed | 1 | 12.7140 | 6.6533 | 1.91x |
| color.gray2rgba | [('z', '>c16'), ('when', '>M8[ns]'), ('dt', '>m8[ms]')] | record-32-reversed | 1 | 31.4344 | 15.2661 | 2.06x |
| color.gray2rgb | [('z', '>c16'), ('when', '>M8[ns]'), ('dt', '>m8[ms]')] | record-32-strided | 1 | 22.4516 | 10.1284 | 2.22x |
| color.gray2rgba | [('z', '>c16'), ('when', '>M8[ns]'), ('dt', '>m8[ms]')] | record-32-strided | 1 | 42.3923 | 19.4109 | 2.18x |
| color.gray2rgb | [('z', '>c16'), ('when', '>M8[ns]'), ('dt', '>m8[ms]')] | record-32-fortran | 1 | 16.0382 | 8.3165 | 1.93x |
| color.gray2rgba | [('z', '>c16'), ('when', '>M8[ns]'), ('dt', '>m8[ms]')] | record-32-fortran | 1 | 34.1070 | 15.8281 | 2.15x |
