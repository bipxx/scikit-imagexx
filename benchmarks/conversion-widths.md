# Conversion performance across integer widths

Each case requires exact values, dtype, and identity behavior against scikit-image. Seven warmed, allocation-inclusive samples; strict dispatch, single-thread reference. Identity/copy rows measure wrapper or copying costs, not a conversion-kernel speedup. All ratios below one remain visible.

| API | Input | Output | Workers | Execution | Original ms | Current ms | Speedup |
|---|---|---|---:|---|---:|---:|---:|
| util.img_as_float | bool | float64 | 1 | native | 2.0827 | 2.0073 | 1.04x |
| util.img_as_float32 | bool | float32 | 1 | native | 0.3907 | 0.1413 | 2.76x |
| util.img_as_float64 | bool | float64 | 1 | native | 2.1258 | 2.2571 | 0.94x |
| util.img_as_ubyte | bool | uint8 | 1 | native | 0.4889 | 0.0433 | 11.30x |
| util.img_as_uint | bool | uint16 | 1 | native | 0.5510 | 0.2503 | 2.20x |
| util.img_as_int | bool | int16 | 1 | native | 0.5311 | 0.2228 | 2.38x |
| util.img_as_bool | bool | bool | 1 | identity | 0.0005 | 0.0003 | 1.78x |
| util.img_as_float | int8 | float64 | 1 | native | 3.2916 | 1.9267 | 1.71x |
| util.img_as_float32 | int8 | float32 | 1 | native | 0.5877 | 0.1882 | 3.12x |
| util.img_as_float64 | int8 | float64 | 1 | native | 3.4489 | 2.1410 | 1.61x |
| util.img_as_ubyte | int8 | uint8 | 1 | native | 1.3175 | 0.1748 | 7.54x |
| util.img_as_uint | int8 | uint16 | 1 | native | 2.9574 | 0.2526 | 11.71x |
| util.img_as_int | int8 | int16 | 1 | native | 2.0355 | 0.2272 | 8.96x |
| util.img_as_bool | int8 | bool | 1 | native | 0.0495 | 0.0485 | 1.02x |
| util.img_as_float | uint16 | float64 | 1 | native | 2.9808 | 2.4880 | 1.20x |
| util.img_as_float32 | uint16 | float32 | 1 | native | 1.9001 | 1.7939 | 1.06x |
| util.img_as_float64 | uint16 | float64 | 1 | native | 3.5600 | 2.6768 | 1.33x |
| util.img_as_ubyte | uint16 | uint8 | 1 | native | 0.4061 | 0.1795 | 2.26x |
| util.img_as_uint | uint16 | uint16 | 1 | identity | 0.0005 | 0.0003 | 1.85x |
| util.img_as_int | uint16 | int16 | 1 | native | 0.1413 | 0.1857 | 0.76x |
| util.img_as_bool | uint16 | bool | 1 | native | 0.0757 | 0.0667 | 1.14x |
| util.img_as_float | int16 | float64 | 1 | native | 3.5556 | 2.2109 | 1.61x |
| util.img_as_float32 | int16 | float32 | 1 | native | 0.6345 | 0.1650 | 3.85x |
| util.img_as_float64 | int16 | float64 | 1 | native | 3.2132 | 2.1257 | 1.51x |
| util.img_as_ubyte | int16 | uint8 | 1 | native | 0.8123 | 0.1908 | 4.26x |
| util.img_as_uint | int16 | uint16 | 1 | native | 2.8430 | 0.1798 | 15.81x |
| util.img_as_int | int16 | int16 | 1 | identity | 0.0005 | 0.0003 | 1.98x |
| util.img_as_bool | int16 | bool | 1 | native | 0.0656 | 0.0610 | 1.08x |
| util.img_as_float | uint32 | float64 | 1 | native | 2.5509 | 2.1188 | 1.20x |
| util.img_as_float32 | uint32 | float32 | 1 | native | 1.7138 | 1.3184 | 1.30x |
| util.img_as_float64 | uint32 | float64 | 1 | native | 2.4311 | 1.9389 | 1.25x |
| util.img_as_ubyte | uint32 | uint8 | 1 | native | 0.6891 | 0.4141 | 1.66x |
| util.img_as_uint | uint32 | uint16 | 1 | native | 0.5356 | 0.2700 | 1.98x |
| util.img_as_int | uint32 | int16 | 1 | native | 0.4653 | 0.2114 | 2.20x |
| util.img_as_bool | uint32 | bool | 1 | native | 0.2927 | 0.3147 | 0.93x |
| util.img_as_float | int32 | float64 | 1 | native | 3.7632 | 2.5599 | 1.47x |
| util.img_as_float32 | int32 | float32 | 1 | native | 1.8098 | 1.2332 | 1.47x |
| util.img_as_float64 | int32 | float64 | 1 | native | 3.3495 | 2.3338 | 1.44x |
| util.img_as_ubyte | int32 | uint8 | 1 | native | 2.3783 | 0.6902 | 3.45x |
| util.img_as_uint | int32 | uint16 | 1 | native | 1.9731 | 0.2852 | 6.92x |
| util.img_as_int | int32 | int16 | 1 | native | 0.5469 | 0.2349 | 2.33x |
| util.img_as_bool | int32 | bool | 1 | native | 0.1107 | 0.1093 | 1.01x |
| util.img_as_float | uint64 | float64 | 1 | native | 5.8281 | 5.0946 | 1.14x |
| util.img_as_float32 | uint64 | float32 | 1 | native | 6.7538 | 3.7799 | 1.79x |
| util.img_as_float64 | uint64 | float64 | 1 | native | 6.0389 | 5.2402 | 1.15x |
| util.img_as_ubyte | uint64 | uint8 | 1 | native | 1.0721 | 0.8760 | 1.22x |
| util.img_as_uint | uint64 | uint16 | 1 | native | 0.9879 | 0.8663 | 1.14x |
| util.img_as_int | uint64 | int16 | 1 | native | 1.1403 | 1.0301 | 1.11x |
| util.img_as_bool | uint64 | bool | 1 | native | 0.2702 | 0.3897 | 0.69x |
| util.img_as_float | int64 | float64 | 1 | native | 3.7443 | 2.4271 | 1.54x |
| util.img_as_float32 | int64 | float32 | 1 | native | 4.6296 | 0.6308 | 7.34x |
| util.img_as_float64 | int64 | float64 | 1 | native | 3.6207 | 2.3148 | 1.56x |
| util.img_as_ubyte | int64 | uint8 | 1 | native | 2.2857 | 0.8362 | 2.73x |
| util.img_as_uint | int64 | uint16 | 1 | native | 2.9058 | 1.0805 | 2.69x |
| util.img_as_int | int64 | int16 | 1 | native | 1.3890 | 0.7676 | 1.81x |
| util.img_as_bool | int64 | bool | 1 | native | 0.3133 | 0.2804 | 1.12x |
