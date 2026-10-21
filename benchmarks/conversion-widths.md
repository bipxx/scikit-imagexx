# Conversion performance across integer widths

Each case requires exact values, dtype, and identity behavior against scikit-image. Seven warmed, allocation-inclusive samples; strict dispatch, single-thread reference. Identity/copy rows measure wrapper or copying costs, not a conversion-kernel speedup. All ratios below one remain visible.

| API | Input | Output | Workers | Execution | Original ms | Current ms | Speedup |
|---|---|---|---:|---|---:|---:|---:|
| util.img_as_float | bool | float64 | 1 | native | 1.8600 | 1.8542 | 1.00x |
| util.img_as_float32 | bool | float32 | 1 | native | 0.3934 | 0.1806 | 2.18x |
| util.img_as_float64 | bool | float64 | 1 | native | 1.9116 | 1.8893 | 1.01x |
| util.img_as_ubyte | bool | uint8 | 1 | native | 0.4815 | 0.0423 | 11.38x |
| util.img_as_uint | bool | uint16 | 1 | native | 0.5347 | 0.2473 | 2.16x |
| util.img_as_int | bool | int16 | 1 | native | 0.5533 | 0.2590 | 2.14x |
| util.img_as_bool | bool | bool | 1 | identity | 0.0010 | 0.0005 | 2.08x |
| util.img_as_float | int8 | float64 | 1 | native | 3.9464 | 2.2426 | 1.76x |
| util.img_as_float32 | int8 | float32 | 1 | native | 0.8488 | 0.2657 | 3.19x |
| util.img_as_float64 | int8 | float64 | 1 | native | 3.8859 | 2.3244 | 1.67x |
| util.img_as_ubyte | int8 | uint8 | 1 | native | 2.0185 | 0.2597 | 7.77x |
| util.img_as_uint | int8 | uint16 | 1 | native | 3.6427 | 0.3467 | 10.51x |
| util.img_as_int | int8 | int16 | 1 | native | 2.3956 | 0.2549 | 9.40x |
| util.img_as_bool | int8 | bool | 1 | native | 0.0545 | 0.0524 | 1.04x |
| util.img_as_float | uint16 | float64 | 1 | native | 2.5863 | 2.1144 | 1.22x |
| util.img_as_float32 | uint16 | float32 | 1 | native | 1.4868 | 1.1457 | 1.30x |
| util.img_as_float64 | uint16 | float64 | 1 | native | 2.4850 | 2.0013 | 1.24x |
| util.img_as_ubyte | uint16 | uint8 | 1 | native | 0.4046 | 0.1811 | 2.23x |
| util.img_as_uint | uint16 | uint16 | 1 | identity | 0.0005 | 0.0003 | 1.66x |
| util.img_as_int | uint16 | int16 | 1 | native | 0.1205 | 0.1808 | 0.67x |
| util.img_as_bool | uint16 | bool | 1 | native | 0.0602 | 0.0560 | 1.07x |
| util.img_as_float | int16 | float64 | 1 | native | 3.1298 | 1.9433 | 1.61x |
| util.img_as_float32 | int16 | float32 | 1 | native | 0.6714 | 0.1822 | 3.68x |
| util.img_as_float64 | int16 | float64 | 1 | native | 3.1311 | 1.9754 | 1.59x |
| util.img_as_ubyte | int16 | uint8 | 1 | native | 0.7707 | 0.1894 | 4.07x |
| util.img_as_uint | int16 | uint16 | 1 | native | 2.6445 | 0.2084 | 12.69x |
| util.img_as_int | int16 | int16 | 1 | identity | 0.0005 | 0.0003 | 1.98x |
| util.img_as_bool | int16 | bool | 1 | native | 0.0660 | 0.0684 | 0.96x |
| util.img_as_float | uint32 | float64 | 1 | native | 2.4081 | 1.9656 | 1.23x |
| util.img_as_float32 | uint32 | float32 | 1 | native | 1.5916 | 1.3235 | 1.20x |
| util.img_as_float64 | uint32 | float64 | 1 | native | 2.5582 | 2.2132 | 1.16x |
| util.img_as_ubyte | uint32 | uint8 | 1 | native | 0.7131 | 0.4815 | 1.48x |
| util.img_as_uint | uint32 | uint16 | 1 | native | 0.5074 | 0.2438 | 2.08x |
| util.img_as_int | uint32 | int16 | 1 | native | 0.5718 | 0.2329 | 2.46x |
| util.img_as_bool | uint32 | bool | 1 | native | 0.2946 | 0.2719 | 1.08x |
| util.img_as_float | int32 | float64 | 1 | native | 3.8914 | 2.4352 | 1.60x |
| util.img_as_float32 | int32 | float32 | 1 | native | 1.6712 | 1.1197 | 1.49x |
| util.img_as_float64 | int32 | float64 | 1 | native | 3.1880 | 2.0785 | 1.53x |
| util.img_as_ubyte | int32 | uint8 | 1 | native | 1.6543 | 0.4087 | 4.05x |
| util.img_as_uint | int32 | uint16 | 1 | native | 1.6341 | 0.2690 | 6.07x |
| util.img_as_int | int32 | int16 | 1 | native | 0.5479 | 0.2025 | 2.71x |
| util.img_as_bool | int32 | bool | 1 | native | 0.1029 | 0.0993 | 1.04x |
| util.img_as_float | uint64 | float64 | 1 | native | 5.6954 | 4.8915 | 1.16x |
| util.img_as_float32 | uint64 | float32 | 1 | native | 8.2710 | 4.1775 | 1.98x |
| util.img_as_float64 | uint64 | float64 | 1 | native | 5.9611 | 5.1372 | 1.16x |
| util.img_as_ubyte | uint64 | uint8 | 1 | native | 1.1076 | 1.0372 | 1.07x |
| util.img_as_uint | uint64 | uint16 | 1 | native | 1.0498 | 0.7728 | 1.36x |
| util.img_as_int | uint64 | int16 | 1 | native | 1.0382 | 0.9112 | 1.14x |
| util.img_as_bool | uint64 | bool | 1 | native | 0.2775 | 0.2573 | 1.08x |
| util.img_as_float | int64 | float64 | 1 | native | 3.5586 | 2.3218 | 1.53x |
| util.img_as_float32 | int64 | float32 | 1 | native | 4.2254 | 0.5416 | 7.80x |
| util.img_as_float64 | int64 | float64 | 1 | native | 3.5757 | 2.2269 | 1.61x |
| util.img_as_ubyte | int64 | uint8 | 1 | native | 2.5236 | 0.9554 | 2.64x |
| util.img_as_uint | int64 | uint16 | 1 | native | 5.2655 | 1.6958 | 3.11x |
| util.img_as_int | int64 | int16 | 1 | native | 2.6098 | 0.9704 | 2.69x |
| util.img_as_bool | int64 | bool | 1 | native | 0.3302 | 0.3704 | 0.89x |
