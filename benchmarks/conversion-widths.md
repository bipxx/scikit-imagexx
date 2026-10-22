# Conversion performance across integer widths

Each case requires exact values, dtype, and identity behavior against scikit-image. Seven warmed, allocation-inclusive samples; strict dispatch, single-thread reference. Identity/copy rows measure wrapper or copying costs, not a conversion-kernel speedup. All ratios below one remain visible.

| API | Input | Output | Workers | Execution | Original ms | Current ms | Speedup |
|---|---|---|---:|---|---:|---:|---:|
| util.img_as_float | bool | float64 | 1 | native | 1.7755 | 1.9562 | 0.91x |
| util.img_as_float32 | bool | float32 | 1 | native | 0.3930 | 0.1826 | 2.15x |
| util.img_as_float64 | bool | float64 | 1 | native | 1.9242 | 2.0080 | 0.96x |
| util.img_as_ubyte | bool | uint8 | 1 | native | 0.5111 | 0.0417 | 12.26x |
| util.img_as_uint | bool | uint16 | 1 | native | 0.5543 | 0.2594 | 2.14x |
| util.img_as_int | bool | int16 | 1 | native | 0.5481 | 0.2666 | 2.06x |
| util.img_as_bool | bool | bool | 1 | identity | 0.0005 | 0.0003 | 2.00x |
| util.img_as_float | int8 | float64 | 1 | native | 3.0528 | 1.9732 | 1.55x |
| util.img_as_float32 | int8 | float32 | 1 | native | 0.6339 | 0.2284 | 2.78x |
| util.img_as_float64 | int8 | float64 | 1 | native | 3.0663 | 1.8898 | 1.62x |
| util.img_as_ubyte | int8 | uint8 | 1 | native | 1.2937 | 0.2225 | 5.81x |
| util.img_as_uint | int8 | uint16 | 1 | native | 3.4900 | 0.2668 | 13.08x |
| util.img_as_int | int8 | int16 | 1 | native | 1.8678 | 0.2200 | 8.49x |
| util.img_as_bool | int8 | bool | 1 | native | 0.0528 | 0.0405 | 1.30x |
| util.img_as_float | uint16 | float64 | 1 | native | 2.4881 | 1.9952 | 1.25x |
| util.img_as_float32 | uint16 | float32 | 1 | native | 1.4041 | 1.0138 | 1.38x |
| util.img_as_float64 | uint16 | float64 | 1 | native | 2.7150 | 1.9709 | 1.38x |
| util.img_as_ubyte | uint16 | uint8 | 1 | native | 0.4324 | 0.1709 | 2.53x |
| util.img_as_uint | uint16 | uint16 | 1 | identity | 0.0006 | 0.0005 | 1.19x |
| util.img_as_int | uint16 | int16 | 1 | native | 0.2199 | 0.3906 | 0.56x |
| util.img_as_bool | uint16 | bool | 1 | native | 0.1042 | 0.0974 | 1.07x |
| util.img_as_float | int16 | float64 | 1 | native | 5.3346 | 3.1459 | 1.70x |
| util.img_as_float32 | int16 | float32 | 1 | native | 1.2366 | 0.3351 | 3.69x |
| util.img_as_float64 | int16 | float64 | 1 | native | 4.2294 | 2.7222 | 1.55x |
| util.img_as_ubyte | int16 | uint8 | 1 | native | 1.2123 | 0.3335 | 3.63x |
| util.img_as_uint | int16 | uint16 | 1 | native | 2.8925 | 0.2210 | 13.09x |
| util.img_as_int | int16 | int16 | 1 | identity | 0.0005 | 0.0004 | 1.19x |
| util.img_as_bool | int16 | bool | 1 | native | 0.1015 | 0.1090 | 0.93x |
| util.img_as_float | uint32 | float64 | 1 | native | 4.5423 | 3.5886 | 1.27x |
| util.img_as_float32 | uint32 | float32 | 1 | native | 1.1714 | 0.3994 | 2.93x |
| util.img_as_float64 | uint32 | float64 | 1 | native | 3.3612 | 2.6766 | 1.26x |
| util.img_as_ubyte | uint32 | uint8 | 1 | native | 1.0999 | 0.5814 | 1.89x |
| util.img_as_uint | uint32 | uint16 | 1 | native | 0.8016 | 0.3137 | 2.56x |
| util.img_as_int | uint32 | int16 | 1 | native | 0.6475 | 0.3314 | 1.95x |
| util.img_as_bool | uint32 | bool | 1 | native | 0.1794 | 0.2103 | 0.85x |
| util.img_as_float | int32 | float64 | 1 | native | 4.9727 | 2.9957 | 1.66x |
| util.img_as_float32 | int32 | float32 | 1 | native | 1.4703 | 0.5610 | 2.62x |
| util.img_as_float64 | int32 | float64 | 1 | native | 5.7271 | 3.3252 | 1.72x |
| util.img_as_ubyte | int32 | uint8 | 1 | native | 2.8418 | 0.4711 | 6.03x |
| util.img_as_uint | int32 | uint16 | 1 | native | 2.5745 | 0.3511 | 7.33x |
| util.img_as_int | int32 | int16 | 1 | native | 1.2932 | 0.3909 | 3.31x |
| util.img_as_bool | int32 | bool | 1 | native | 0.1883 | 0.1501 | 1.25x |
| util.img_as_float | uint64 | float64 | 1 | native | 7.1524 | 5.3117 | 1.35x |
| util.img_as_float32 | uint64 | float32 | 1 | native | 8.1053 | 4.3611 | 1.86x |
| util.img_as_float64 | uint64 | float64 | 1 | native | 7.5986 | 5.8466 | 1.30x |
| util.img_as_ubyte | uint64 | uint8 | 1 | native | 2.0819 | 1.6609 | 1.25x |
| util.img_as_uint | uint64 | uint16 | 1 | native | 1.3851 | 1.1023 | 1.26x |
| util.img_as_int | uint64 | int16 | 1 | native | 1.3151 | 0.9767 | 1.35x |
| util.img_as_bool | uint64 | bool | 1 | native | 0.5969 | 0.7724 | 0.77x |
| util.img_as_float | int64 | float64 | 1 | native | 5.2837 | 4.1153 | 1.28x |
| util.img_as_float32 | int64 | float32 | 1 | native | 5.9878 | 1.1937 | 5.02x |
| util.img_as_float64 | int64 | float64 | 1 | native | 3.3466 | 2.3910 | 1.40x |
| util.img_as_ubyte | int64 | uint8 | 1 | native | 2.1057 | 0.9567 | 2.20x |
| util.img_as_uint | int64 | uint16 | 1 | native | 2.8665 | 1.2075 | 2.37x |
| util.img_as_int | int64 | int16 | 1 | native | 2.0501 | 1.2008 | 1.71x |
| util.img_as_bool | int64 | bool | 1 | native | 0.3083 | 0.3371 | 0.91x |
