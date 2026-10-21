# Conversion performance across integer widths

Each case requires exact values, dtype, and identity behavior against scikit-image. Seven warmed, allocation-inclusive samples; strict dispatch, single-thread reference. Identity/copy rows measure wrapper or copying costs, not a conversion-kernel speedup. All ratios below one remain visible.

| API | Input | Output | Workers | Execution | Original ms | Current ms | Speedup |
|---|---|---|---:|---|---:|---:|---:|
| util.img_as_float | bool | float64 | 1 | native | 1.9309 | 1.9792 | 0.98x |
| util.img_as_float32 | bool | float32 | 1 | native | 0.4190 | 0.1545 | 2.71x |
| util.img_as_float64 | bool | float64 | 1 | native | 2.0007 | 1.9720 | 1.01x |
| util.img_as_ubyte | bool | uint8 | 1 | native | 0.4809 | 0.0393 | 12.24x |
| util.img_as_uint | bool | uint16 | 1 | native | 0.5300 | 0.2984 | 1.78x |
| util.img_as_int | bool | int16 | 1 | native | 0.6247 | 0.2470 | 2.53x |
| util.img_as_bool | bool | bool | 1 | identity | 0.0010 | 0.0010 | 1.07x |
| util.img_as_float | int8 | float64 | 1 | native | 4.0153 | 2.5369 | 1.58x |
| util.img_as_float32 | int8 | float32 | 1 | native | 0.8726 | 0.2926 | 2.98x |
| util.img_as_float64 | int8 | float64 | 1 | native | 3.1615 | 2.0442 | 1.55x |
| util.img_as_ubyte | int8 | uint8 | 1 | native | 1.3640 | 0.1736 | 7.86x |
| util.img_as_uint | int8 | uint16 | 1 | native | 3.1514 | 0.2452 | 12.85x |
| util.img_as_int | int8 | int16 | 1 | native | 1.8476 | 0.2160 | 8.55x |
| util.img_as_bool | int8 | bool | 1 | native | 0.0992 | 0.0921 | 1.08x |
| util.img_as_float | uint16 | float64 | 1 | native | 2.8938 | 2.2670 | 1.28x |
| util.img_as_float32 | uint16 | float32 | 1 | native | 1.4951 | 1.2201 | 1.23x |
| util.img_as_float64 | uint16 | float64 | 1 | native | 2.5368 | 2.0243 | 1.25x |
| util.img_as_ubyte | uint16 | uint8 | 1 | native | 0.4476 | 0.1650 | 2.71x |
| util.img_as_uint | uint16 | uint16 | 1 | identity | 0.0007 | 0.0003 | 2.75x |
| util.img_as_int | uint16 | int16 | 1 | native | 0.1406 | 0.2034 | 0.69x |
| util.img_as_bool | uint16 | bool | 1 | native | 0.0883 | 0.0779 | 1.13x |
| util.img_as_float | int16 | float64 | 1 | native | 3.2687 | 2.1410 | 1.53x |
| util.img_as_float32 | int16 | float32 | 1 | native | 0.6923 | 0.1948 | 3.55x |
| util.img_as_float64 | int16 | float64 | 1 | native | 3.2445 | 2.1330 | 1.52x |
| util.img_as_ubyte | int16 | uint8 | 1 | native | 0.7967 | 0.2083 | 3.82x |
| util.img_as_uint | int16 | uint16 | 1 | native | 2.5513 | 0.2034 | 12.54x |
| util.img_as_int | int16 | int16 | 1 | identity | 0.0014 | 0.0006 | 2.46x |
| util.img_as_bool | int16 | bool | 1 | native | 0.0701 | 0.0773 | 0.91x |
| util.img_as_float | uint32 | float64 | 1 | native | 2.8456 | 2.4263 | 1.17x |
| util.img_as_float32 | uint32 | float32 | 1 | native | 1.5756 | 1.2052 | 1.31x |
| util.img_as_float64 | uint32 | float64 | 1 | native | 2.7618 | 2.3664 | 1.17x |
| util.img_as_ubyte | uint32 | uint8 | 1 | native | 0.6660 | 0.4864 | 1.37x |
| util.img_as_uint | uint32 | uint16 | 1 | native | 0.8935 | 0.8521 | 1.05x |
| util.img_as_int | uint32 | int16 | 1 | native | 0.5300 | 0.2237 | 2.37x |
| util.img_as_bool | uint32 | bool | 1 | native | 0.1505 | 0.1429 | 1.05x |
| util.img_as_float | int32 | float64 | 1 | native | 2.9655 | 1.9910 | 1.49x |
| util.img_as_float32 | int32 | float32 | 1 | native | 0.5952 | 0.4867 | 1.22x |
| util.img_as_float64 | int32 | float64 | 1 | native | 3.6873 | 2.3065 | 1.60x |
| util.img_as_ubyte | int32 | uint8 | 1 | native | 1.8761 | 0.3389 | 5.54x |
| util.img_as_uint | int32 | uint16 | 1 | native | 1.6595 | 0.2312 | 7.18x |
| util.img_as_int | int32 | int16 | 1 | native | 0.6380 | 0.2244 | 2.84x |
| util.img_as_bool | int32 | bool | 1 | native | 0.1698 | 0.1041 | 1.63x |
| util.img_as_float | uint64 | float64 | 1 | native | 6.7956 | 5.3119 | 1.28x |
| util.img_as_float32 | uint64 | float32 | 1 | native | 6.6067 | 3.6022 | 1.83x |
| util.img_as_float64 | uint64 | float64 | 1 | native | 6.7861 | 5.3247 | 1.27x |
| util.img_as_ubyte | uint64 | uint8 | 1 | native | 1.3094 | 0.8972 | 1.46x |
| util.img_as_uint | uint64 | uint16 | 1 | native | 1.0913 | 0.8778 | 1.24x |
| util.img_as_int | uint64 | int16 | 1 | native | 1.0972 | 1.1356 | 0.97x |
| util.img_as_bool | uint64 | bool | 1 | native | 0.2862 | 0.2570 | 1.11x |
| util.img_as_float | int64 | float64 | 1 | native | 3.2176 | 2.2176 | 1.45x |
| util.img_as_float32 | int64 | float32 | 1 | native | 3.7685 | 0.5366 | 7.02x |
| util.img_as_float64 | int64 | float64 | 1 | native | 4.3992 | 2.1560 | 2.04x |
| util.img_as_ubyte | int64 | uint8 | 1 | native | 2.4548 | 0.7879 | 3.12x |
| util.img_as_uint | int64 | uint16 | 1 | native | 2.6727 | 0.9050 | 2.95x |
| util.img_as_int | int64 | int16 | 1 | native | 1.6112 | 0.8252 | 1.95x |
| util.img_as_bool | int64 | bool | 1 | native | 0.3580 | 0.3270 | 1.09x |
