# Conversion performance across integer widths

Each case requires exact values, dtype, and identity behavior against scikit-image. Seven warmed, allocation-inclusive samples; strict dispatch, single-thread reference. Identity/copy rows measure wrapper or copying costs, not a conversion-kernel speedup. All ratios below one remain visible.

| API | Input | Output | Workers | Execution | Original ms | Current ms | Speedup |
|---|---|---|---:|---|---:|---:|---:|
| util.img_as_float | bool | float64 | 1 | native | 2.4512 | 2.3912 | 1.03x |
| util.img_as_float32 | bool | float32 | 1 | native | 0.4073 | 0.1931 | 2.11x |
| util.img_as_float64 | bool | float64 | 1 | native | 2.0529 | 2.0182 | 1.02x |
| util.img_as_ubyte | bool | uint8 | 1 | native | 0.4874 | 0.0465 | 10.48x |
| util.img_as_uint | bool | uint16 | 1 | native | 0.5465 | 0.2523 | 2.17x |
| util.img_as_int | bool | int16 | 1 | native | 0.5693 | 0.2430 | 2.34x |
| util.img_as_bool | bool | bool | 1 | identity | 0.0005 | 0.0003 | 1.90x |
| util.img_as_float | int8 | float64 | 1 | native | 3.1723 | 1.9957 | 1.59x |
| util.img_as_float32 | int8 | float32 | 1 | native | 0.7236 | 0.2084 | 3.47x |
| util.img_as_float64 | int8 | float64 | 1 | native | 3.7201 | 2.3823 | 1.56x |
| util.img_as_ubyte | int8 | uint8 | 1 | native | 1.3416 | 0.2063 | 6.50x |
| util.img_as_uint | int8 | uint16 | 1 | native | 2.7685 | 0.2504 | 11.06x |
| util.img_as_int | int8 | int16 | 1 | native | 1.8664 | 0.2229 | 8.37x |
| util.img_as_bool | int8 | bool | 1 | native | 0.0498 | 0.0447 | 1.12x |
| util.img_as_float | uint16 | float64 | 1 | native | 2.7899 | 2.1485 | 1.30x |
| util.img_as_float32 | uint16 | float32 | 1 | native | 1.3914 | 1.0621 | 1.31x |
| util.img_as_float64 | uint16 | float64 | 1 | native | 2.8583 | 2.4989 | 1.14x |
| util.img_as_ubyte | uint16 | uint8 | 1 | native | 0.3912 | 0.1813 | 2.16x |
| util.img_as_uint | uint16 | uint16 | 1 | identity | 0.0005 | 0.0003 | 1.91x |
| util.img_as_int | uint16 | int16 | 1 | native | 0.1511 | 0.1899 | 0.80x |
| util.img_as_bool | uint16 | bool | 1 | native | 0.0781 | 0.0663 | 1.18x |
| util.img_as_float | int16 | float64 | 1 | native | 3.6316 | 2.2154 | 1.64x |
| util.img_as_float32 | int16 | float32 | 1 | native | 0.6857 | 0.1866 | 3.68x |
| util.img_as_float64 | int16 | float64 | 1 | native | 4.0005 | 2.5458 | 1.57x |
| util.img_as_ubyte | int16 | uint8 | 1 | native | 0.8845 | 0.1958 | 4.52x |
| util.img_as_uint | int16 | uint16 | 1 | native | 3.0491 | 0.1956 | 15.59x |
| util.img_as_int | int16 | int16 | 1 | identity | 0.0005 | 0.0003 | 1.90x |
| util.img_as_bool | int16 | bool | 1 | native | 0.0684 | 0.0670 | 1.02x |
| util.img_as_float | uint32 | float64 | 1 | native | 2.6114 | 2.1491 | 1.22x |
| util.img_as_float32 | uint32 | float32 | 1 | native | 0.8579 | 0.2318 | 3.70x |
| util.img_as_float64 | uint32 | float64 | 1 | native | 2.8820 | 2.3812 | 1.21x |
| util.img_as_ubyte | uint32 | uint8 | 1 | native | 0.6544 | 0.4641 | 1.41x |
| util.img_as_uint | uint32 | uint16 | 1 | native | 0.5463 | 0.2924 | 1.87x |
| util.img_as_int | uint32 | int16 | 1 | native | 0.5306 | 0.2222 | 2.39x |
| util.img_as_bool | uint32 | bool | 1 | native | 0.3669 | 0.3111 | 1.18x |
| util.img_as_float | int32 | float64 | 1 | native | 3.8685 | 2.5331 | 1.53x |
| util.img_as_float32 | int32 | float32 | 1 | native | 0.9153 | 0.5042 | 1.82x |
| util.img_as_float64 | int32 | float64 | 1 | native | 3.3580 | 2.3760 | 1.41x |
| util.img_as_ubyte | int32 | uint8 | 1 | native | 1.9098 | 0.3001 | 6.36x |
| util.img_as_uint | int32 | uint16 | 1 | native | 1.8946 | 0.2389 | 7.93x |
| util.img_as_int | int32 | int16 | 1 | native | 0.5778 | 0.2544 | 2.27x |
| util.img_as_bool | int32 | bool | 1 | native | 0.1489 | 0.1413 | 1.05x |
| util.img_as_float | uint64 | float64 | 1 | native | 6.3033 | 5.3985 | 1.17x |
| util.img_as_float32 | uint64 | float32 | 1 | native | 6.8803 | 3.7386 | 1.84x |
| util.img_as_float64 | uint64 | float64 | 1 | native | 5.8365 | 5.0560 | 1.15x |
| util.img_as_ubyte | uint64 | uint8 | 1 | native | 1.1646 | 0.9216 | 1.26x |
| util.img_as_uint | uint64 | uint16 | 1 | native | 1.2260 | 0.9705 | 1.26x |
| util.img_as_int | uint64 | int16 | 1 | native | 1.2307 | 0.9871 | 1.25x |
| util.img_as_bool | uint64 | bool | 1 | native | 0.3801 | 0.3723 | 1.02x |
| util.img_as_float | int64 | float64 | 1 | native | 4.4676 | 3.0656 | 1.46x |
| util.img_as_float32 | int64 | float32 | 1 | native | 4.9223 | 0.8491 | 5.80x |
| util.img_as_float64 | int64 | float64 | 1 | native | 3.5856 | 2.5846 | 1.39x |
| util.img_as_ubyte | int64 | uint8 | 1 | native | 2.4405 | 0.8617 | 2.83x |
| util.img_as_uint | int64 | uint16 | 1 | native | 3.4286 | 1.1688 | 2.93x |
| util.img_as_int | int64 | int16 | 1 | native | 1.5792 | 0.9551 | 1.65x |
| util.img_as_bool | int64 | bool | 1 | native | 0.2968 | 0.3467 | 0.86x |
