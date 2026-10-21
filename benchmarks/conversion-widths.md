# Conversion performance across integer widths

Each case requires exact values, dtype, and identity behavior against scikit-image. Seven warmed, allocation-inclusive samples; strict dispatch, single-thread reference. Identity/copy rows measure wrapper or copying costs, not a conversion-kernel speedup. All ratios below one remain visible.

| API | Input | Output | Workers | Execution | Original ms | Current ms | Speedup |
|---|---|---|---:|---|---:|---:|---:|
| util.img_as_float | bool | float64 | 1 | native | 2.8060 | 2.6339 | 1.07x |
| util.img_as_float32 | bool | float32 | 1 | native | 0.3993 | 0.1853 | 2.15x |
| util.img_as_float64 | bool | float64 | 1 | native | 2.3570 | 2.3376 | 1.01x |
| util.img_as_ubyte | bool | uint8 | 1 | native | 0.4948 | 0.0466 | 10.62x |
| util.img_as_uint | bool | uint16 | 1 | native | 0.5440 | 0.2883 | 1.89x |
| util.img_as_int | bool | int16 | 1 | native | 0.5616 | 0.2861 | 1.96x |
| util.img_as_bool | bool | bool | 1 | identity | 0.0005 | 0.0003 | 1.81x |
| util.img_as_float | int8 | float64 | 1 | native | 3.7636 | 2.5214 | 1.49x |
| util.img_as_float32 | int8 | float32 | 1 | native | 0.6434 | 0.2045 | 3.15x |
| util.img_as_float64 | int8 | float64 | 1 | native | 3.8933 | 2.4631 | 1.58x |
| util.img_as_ubyte | int8 | uint8 | 1 | native | 1.3401 | 0.1792 | 7.48x |
| util.img_as_uint | int8 | uint16 | 1 | native | 3.3754 | 0.2645 | 12.76x |
| util.img_as_int | int8 | int16 | 1 | native | 2.1776 | 0.2302 | 9.46x |
| util.img_as_bool | int8 | bool | 1 | native | 0.0402 | 0.0410 | 0.98x |
| util.img_as_float | uint16 | float64 | 1 | native | 3.2199 | 2.8783 | 1.12x |
| util.img_as_float32 | uint16 | float32 | 1 | native | 2.0456 | 1.4501 | 1.41x |
| util.img_as_float64 | uint16 | float64 | 1 | native | 3.6010 | 3.0916 | 1.16x |
| util.img_as_ubyte | uint16 | uint8 | 1 | native | 0.4010 | 0.1870 | 2.14x |
| util.img_as_uint | uint16 | uint16 | 1 | identity | 0.0005 | 0.0003 | 1.74x |
| util.img_as_int | uint16 | int16 | 1 | native | 0.1465 | 0.1979 | 0.74x |
| util.img_as_bool | uint16 | bool | 1 | native | 0.0735 | 0.0687 | 1.07x |
| util.img_as_float | int16 | float64 | 1 | native | 4.2842 | 2.9004 | 1.48x |
| util.img_as_float32 | int16 | float32 | 1 | native | 0.7240 | 0.1834 | 3.95x |
| util.img_as_float64 | int16 | float64 | 1 | native | 4.1158 | 2.3693 | 1.74x |
| util.img_as_ubyte | int16 | uint8 | 1 | native | 0.8643 | 0.2040 | 4.24x |
| util.img_as_uint | int16 | uint16 | 1 | native | 2.9041 | 0.1670 | 17.39x |
| util.img_as_int | int16 | int16 | 1 | identity | 0.0005 | 0.0003 | 1.58x |
| util.img_as_bool | int16 | bool | 1 | native | 0.0675 | 0.0606 | 1.11x |
| util.img_as_float | uint32 | float64 | 1 | native | 2.6863 | 2.1031 | 1.28x |
| util.img_as_float32 | uint32 | float32 | 1 | native | 1.5787 | 1.2202 | 1.29x |
| util.img_as_float64 | uint32 | float64 | 1 | native | 2.6030 | 2.1470 | 1.21x |
| util.img_as_ubyte | uint32 | uint8 | 1 | native | 0.7261 | 0.4587 | 1.58x |
| util.img_as_uint | uint32 | uint16 | 1 | native | 0.5681 | 0.1984 | 2.86x |
| util.img_as_int | uint32 | int16 | 1 | native | 0.4927 | 0.2137 | 2.31x |
| util.img_as_bool | uint32 | bool | 1 | native | 0.1871 | 0.1330 | 1.41x |
| util.img_as_float | int32 | float64 | 1 | native | 3.8707 | 2.3563 | 1.64x |
| util.img_as_float32 | int32 | float32 | 1 | native | 1.6864 | 0.5122 | 3.29x |
| util.img_as_float64 | int32 | float64 | 1 | native | 3.1239 | 2.3071 | 1.35x |
| util.img_as_ubyte | int32 | uint8 | 1 | native | 2.1122 | 0.4447 | 4.75x |
| util.img_as_uint | int32 | uint16 | 1 | native | 1.9840 | 0.2770 | 7.16x |
| util.img_as_int | int32 | int16 | 1 | native | 0.6010 | 0.2441 | 2.46x |
| util.img_as_bool | int32 | bool | 1 | native | 0.1207 | 0.1041 | 1.16x |
| util.img_as_float | uint64 | float64 | 1 | native | 5.9864 | 5.0855 | 1.18x |
| util.img_as_float32 | uint64 | float32 | 1 | native | 7.0052 | 3.7273 | 1.88x |
| util.img_as_float64 | uint64 | float64 | 1 | native | 6.1325 | 5.4310 | 1.13x |
| util.img_as_ubyte | uint64 | uint8 | 1 | native | 1.5587 | 1.0952 | 1.42x |
| util.img_as_uint | uint64 | uint16 | 1 | native | 1.1192 | 0.9920 | 1.13x |
| util.img_as_int | uint64 | int16 | 1 | native | 1.3895 | 1.0605 | 1.31x |
| util.img_as_bool | uint64 | bool | 1 | native | 0.4639 | 0.4519 | 1.03x |
| util.img_as_float | int64 | float64 | 1 | native | 3.7492 | 2.4627 | 1.52x |
| util.img_as_float32 | int64 | float32 | 1 | native | 3.8678 | 0.5999 | 6.45x |
| util.img_as_float64 | int64 | float64 | 1 | native | 3.4758 | 2.0779 | 1.67x |
| util.img_as_ubyte | int64 | uint8 | 1 | native | 2.0843 | 0.7922 | 2.63x |
| util.img_as_uint | int64 | uint16 | 1 | native | 2.4719 | 0.7773 | 3.18x |
| util.img_as_int | int64 | int16 | 1 | native | 1.3282 | 0.8307 | 1.60x |
| util.img_as_bool | int64 | bool | 1 | native | 0.2522 | 0.2218 | 1.14x |
