# Conversion performance across integer widths

Each case requires exact values, dtype, and identity behavior against scikit-image. Seven warmed, allocation-inclusive samples; strict dispatch, single-thread reference. Identity/copy rows measure wrapper or copying costs, not a conversion-kernel speedup. All ratios below one remain visible.

| API | Input | Output | Workers | Execution | Original ms | Current ms | Speedup |
|---|---|---|---:|---|---:|---:|---:|
| util.img_as_float | bool | float64 | 1 | native | 2.1170 | 2.2510 | 0.94x |
| util.img_as_float32 | bool | float32 | 1 | native | 0.4113 | 0.1925 | 2.14x |
| util.img_as_float64 | bool | float64 | 1 | native | 2.3905 | 2.3213 | 1.03x |
| util.img_as_ubyte | bool | uint8 | 1 | native | 0.5151 | 0.0554 | 9.30x |
| util.img_as_uint | bool | uint16 | 1 | native | 0.7207 | 0.2696 | 2.67x |
| util.img_as_int | bool | int16 | 1 | native | 0.7777 | 0.2907 | 2.68x |
| util.img_as_bool | bool | bool | 1 | identity | 0.0006 | 0.0004 | 1.48x |
| util.img_as_float | int8 | float64 | 1 | native | 6.8076 | 3.7312 | 1.82x |
| util.img_as_float32 | int8 | float32 | 1 | native | 1.9747 | 0.5206 | 3.79x |
| util.img_as_float64 | int8 | float64 | 1 | native | 5.4528 | 2.6902 | 2.03x |
| util.img_as_ubyte | int8 | uint8 | 1 | native | 2.2568 | 0.2292 | 9.85x |
| util.img_as_uint | int8 | uint16 | 1 | native | 3.7872 | 0.2880 | 13.15x |
| util.img_as_int | int8 | int16 | 1 | native | 3.1428 | 0.2599 | 12.09x |
| util.img_as_bool | int8 | bool | 1 | native | 0.0634 | 0.0586 | 1.08x |
| util.img_as_float | uint16 | float64 | 1 | native | 3.3261 | 2.7471 | 1.21x |
| util.img_as_float32 | uint16 | float32 | 1 | native | 1.6747 | 1.2772 | 1.31x |
| util.img_as_float64 | uint16 | float64 | 1 | native | 3.0646 | 2.6196 | 1.17x |
| util.img_as_ubyte | uint16 | uint8 | 1 | native | 0.6123 | 0.4477 | 1.37x |
| util.img_as_uint | uint16 | uint16 | 1 | identity | 0.0008 | 0.0005 | 1.59x |
| util.img_as_int | uint16 | int16 | 1 | native | 0.2013 | 0.3495 | 0.58x |
| util.img_as_bool | uint16 | bool | 1 | native | 0.0925 | 0.1294 | 0.72x |
| util.img_as_float | int16 | float64 | 1 | native | 4.7810 | 2.6829 | 1.78x |
| util.img_as_float32 | int16 | float32 | 1 | native | 1.3243 | 0.2425 | 5.46x |
| util.img_as_float64 | int16 | float64 | 1 | native | 5.3139 | 2.9222 | 1.82x |
| util.img_as_ubyte | int16 | uint8 | 1 | native | 1.9716 | 0.2594 | 7.60x |
| util.img_as_uint | int16 | uint16 | 1 | native | 4.5016 | 0.2267 | 19.86x |
| util.img_as_int | int16 | int16 | 1 | identity | 0.0010 | 0.0006 | 1.71x |
| util.img_as_bool | int16 | bool | 1 | native | 0.1094 | 0.1030 | 1.06x |
| util.img_as_float | uint32 | float64 | 1 | native | 3.6883 | 2.9470 | 1.25x |
| util.img_as_float32 | uint32 | float32 | 1 | native | 1.5293 | 0.4688 | 3.26x |
| util.img_as_float64 | uint32 | float64 | 1 | native | 4.2998 | 3.4706 | 1.24x |
| util.img_as_ubyte | uint32 | uint8 | 1 | native | 0.9081 | 0.3531 | 2.57x |
| util.img_as_uint | uint32 | uint16 | 1 | native | 2.2015 | 1.6279 | 1.35x |
| util.img_as_int | uint32 | int16 | 1 | native | 1.4262 | 0.5665 | 2.52x |
| util.img_as_bool | uint32 | bool | 1 | native | 0.3765 | 0.1978 | 1.90x |
| util.img_as_float | int32 | float64 | 1 | native | 5.1711 | 3.2325 | 1.60x |
| util.img_as_float32 | int32 | float32 | 1 | native | 2.1137 | 0.6941 | 3.05x |
| util.img_as_float64 | int32 | float64 | 1 | native | 5.4656 | 3.0200 | 1.81x |
| util.img_as_ubyte | int32 | uint8 | 1 | native | 2.6367 | 0.3766 | 7.00x |
| util.img_as_uint | int32 | uint16 | 1 | native | 3.3235 | 0.4537 | 7.33x |
| util.img_as_int | int32 | int16 | 1 | native | 1.1577 | 0.3920 | 2.95x |
| util.img_as_bool | int32 | bool | 1 | native | 0.1804 | 0.1746 | 1.03x |
| util.img_as_float | uint64 | float64 | 1 | native | 7.1765 | 6.7130 | 1.07x |
| util.img_as_float32 | uint64 | float32 | 1 | native | 9.1765 | 5.3132 | 1.73x |
| util.img_as_float64 | uint64 | float64 | 1 | native | 7.8181 | 6.2901 | 1.24x |
| util.img_as_ubyte | uint64 | uint8 | 1 | native | 2.6886 | 2.7264 | 0.99x |
| util.img_as_uint | uint64 | uint16 | 1 | native | 5.7109 | 4.9774 | 1.15x |
| util.img_as_int | uint64 | int16 | 1 | native | 6.5001 | 4.8540 | 1.34x |
| util.img_as_bool | uint64 | bool | 1 | native | 1.4719 | 2.2203 | 0.66x |
| util.img_as_float | int64 | float64 | 1 | native | 5.7516 | 3.2391 | 1.78x |
| util.img_as_float32 | int64 | float32 | 1 | native | 8.0853 | 1.7352 | 4.66x |
| util.img_as_float64 | int64 | float64 | 1 | native | 5.5503 | 3.3044 | 1.68x |
| util.img_as_ubyte | int64 | uint8 | 1 | native | 3.3843 | 1.5638 | 2.16x |
| util.img_as_uint | int64 | uint16 | 1 | native | 4.7160 | 2.2742 | 2.07x |
| util.img_as_int | int64 | int16 | 1 | native | 3.0657 | 2.0229 | 1.52x |
| util.img_as_bool | int64 | bool | 1 | native | 0.9235 | 0.4948 | 1.87x |
