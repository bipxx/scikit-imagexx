# Conversion performance across integer widths

Each case requires exact values, dtype, and identity behavior against scikit-image. Seven warmed, allocation-inclusive samples; strict dispatch, single-thread reference. Identity/copy rows measure wrapper or copying costs, not a conversion-kernel speedup. All ratios below one remain visible.

| API | Input | Output | Workers | Execution | Original ms | Current ms | Speedup |
|---|---|---|---:|---|---:|---:|---:|
| util.img_as_float | bool | float64 | 1 | native | 1.9395 | 1.9836 | 0.98x |
| util.img_as_float32 | bool | float32 | 1 | native | 0.3971 | 0.1854 | 2.14x |
| util.img_as_float64 | bool | float64 | 1 | native | 1.8848 | 1.8688 | 1.01x |
| util.img_as_ubyte | bool | uint8 | 1 | native | 0.4771 | 0.0434 | 10.99x |
| util.img_as_uint | bool | uint16 | 1 | native | 0.5256 | 0.2488 | 2.11x |
| util.img_as_int | bool | int16 | 1 | native | 0.5433 | 0.2580 | 2.11x |
| util.img_as_bool | bool | bool | 1 | identity | 0.0005 | 0.0003 | 1.81x |
| util.img_as_float | int8 | float64 | 1 | native | 3.0091 | 1.8903 | 1.59x |
| util.img_as_float32 | int8 | float32 | 1 | native | 0.6841 | 0.2473 | 2.77x |
| util.img_as_float64 | int8 | float64 | 1 | native | 3.9549 | 2.3593 | 1.68x |
| util.img_as_ubyte | int8 | uint8 | 1 | native | 1.3588 | 0.2409 | 5.64x |
| util.img_as_uint | int8 | uint16 | 1 | native | 3.3209 | 0.2984 | 11.13x |
| util.img_as_int | int8 | int16 | 1 | native | 2.0272 | 0.2352 | 8.62x |
| util.img_as_bool | int8 | bool | 1 | native | 0.0464 | 0.0531 | 0.87x |
| util.img_as_float | uint16 | float64 | 1 | native | 3.1020 | 2.4283 | 1.28x |
| util.img_as_float32 | uint16 | float32 | 1 | native | 1.7831 | 1.2459 | 1.43x |
| util.img_as_float64 | uint16 | float64 | 1 | native | 2.9092 | 2.4651 | 1.18x |
| util.img_as_ubyte | uint16 | uint8 | 1 | native | 0.6679 | 0.4311 | 1.55x |
| util.img_as_uint | uint16 | uint16 | 1 | identity | 0.0005 | 0.0003 | 1.76x |
| util.img_as_int | uint16 | int16 | 1 | native | 0.1466 | 0.1803 | 0.81x |
| util.img_as_bool | uint16 | bool | 1 | native | 0.0677 | 0.0617 | 1.10x |
| util.img_as_float | int16 | float64 | 1 | native | 3.2331 | 2.0687 | 1.56x |
| util.img_as_float32 | int16 | float32 | 1 | native | 1.4594 | 0.2674 | 5.46x |
| util.img_as_float64 | int16 | float64 | 1 | native | 4.1675 | 2.4574 | 1.70x |
| util.img_as_ubyte | int16 | uint8 | 1 | native | 1.6032 | 0.3546 | 4.52x |
| util.img_as_uint | int16 | uint16 | 1 | native | 4.0942 | 0.3118 | 13.13x |
| util.img_as_int | int16 | int16 | 1 | identity | 0.0005 | 0.0003 | 1.78x |
| util.img_as_bool | int16 | bool | 1 | native | 0.0812 | 0.0918 | 0.88x |
| util.img_as_float | uint32 | float64 | 1 | native | 3.5754 | 2.8534 | 1.25x |
| util.img_as_float32 | uint32 | float32 | 1 | native | 0.9497 | 0.6861 | 1.38x |
| util.img_as_float64 | uint32 | float64 | 1 | native | 3.4017 | 2.6836 | 1.27x |
| util.img_as_ubyte | uint32 | uint8 | 1 | native | 0.5465 | 0.2534 | 2.16x |
| util.img_as_uint | uint32 | uint16 | 1 | native | 1.3169 | 0.6717 | 1.96x |
| util.img_as_int | uint32 | int16 | 1 | native | 0.6880 | 0.2945 | 2.34x |
| util.img_as_bool | uint32 | bool | 1 | native | 0.2066 | 0.1861 | 1.11x |
| util.img_as_float | int32 | float64 | 1 | native | 4.0872 | 2.1982 | 1.86x |
| util.img_as_float32 | int32 | float32 | 1 | native | 2.6061 | 1.3813 | 1.89x |
| util.img_as_float64 | int32 | float64 | 1 | native | 6.4896 | 3.9915 | 1.63x |
| util.img_as_ubyte | int32 | uint8 | 1 | native | 3.9657 | 0.8990 | 4.41x |
| util.img_as_uint | int32 | uint16 | 1 | native | 5.1408 | 0.9763 | 5.27x |
| util.img_as_int | int32 | int16 | 1 | native | 2.0431 | 0.8327 | 2.45x |
| util.img_as_bool | int32 | bool | 1 | native | 0.4529 | 0.4466 | 1.01x |
| util.img_as_float | uint64 | float64 | 1 | native | 6.7175 | 5.7799 | 1.16x |
| util.img_as_float32 | uint64 | float32 | 1 | native | 7.6023 | 4.2669 | 1.78x |
| util.img_as_float64 | uint64 | float64 | 1 | native | 5.7495 | 4.9756 | 1.16x |
| util.img_as_ubyte | uint64 | uint8 | 1 | native | 1.1649 | 0.8246 | 1.41x |
| util.img_as_uint | uint64 | uint16 | 1 | native | 1.0180 | 0.8976 | 1.13x |
| util.img_as_int | uint64 | int16 | 1 | native | 1.1331 | 0.7849 | 1.44x |
| util.img_as_bool | uint64 | bool | 1 | native | 0.2879 | 0.2436 | 1.18x |
| util.img_as_float | int64 | float64 | 1 | native | 3.1038 | 2.0285 | 1.53x |
| util.img_as_float32 | int64 | float32 | 1 | native | 3.6331 | 0.6029 | 6.03x |
| util.img_as_float64 | int64 | float64 | 1 | native | 3.2483 | 2.1463 | 1.51x |
| util.img_as_ubyte | int64 | uint8 | 1 | native | 2.0542 | 0.7611 | 2.70x |
| util.img_as_uint | int64 | uint16 | 1 | native | 3.8745 | 0.9539 | 4.06x |
| util.img_as_int | int64 | int16 | 1 | native | 1.3548 | 0.7914 | 1.71x |
| util.img_as_bool | int64 | bool | 1 | native | 0.4068 | 0.2984 | 1.36x |
