# Conversion performance across integer widths

Each case requires exact values, dtype, and identity behavior against scikit-image. Seven warmed, allocation-inclusive samples; strict dispatch, single-thread reference. Identity/copy rows measure wrapper or copying costs, not a conversion-kernel speedup. All ratios below one remain visible.

| API | Input | Output | Workers | Execution | Original ms | Current ms | Speedup |
|---|---|---|---:|---|---:|---:|---:|
| util.img_as_float | bool | float64 | 1 | native | 2.1072 | 2.3226 | 0.91x |
| util.img_as_float32 | bool | float32 | 1 | native | 0.4071 | 0.1565 | 2.60x |
| util.img_as_float64 | bool | float64 | 1 | native | 2.3169 | 2.3086 | 1.00x |
| util.img_as_ubyte | bool | uint8 | 1 | native | 0.8429 | 0.0781 | 10.79x |
| util.img_as_uint | bool | uint16 | 1 | native | 0.5647 | 0.2549 | 2.22x |
| util.img_as_int | bool | int16 | 1 | native | 0.6862 | 0.2464 | 2.79x |
| util.img_as_bool | bool | bool | 1 | identity | 0.0006 | 0.0003 | 1.70x |
| util.img_as_float | int8 | float64 | 1 | native | 3.6665 | 2.0231 | 1.81x |
| util.img_as_float32 | int8 | float32 | 1 | native | 0.8271 | 0.2227 | 3.71x |
| util.img_as_float64 | int8 | float64 | 1 | native | 3.9108 | 2.3154 | 1.69x |
| util.img_as_ubyte | int8 | uint8 | 1 | native | 1.3692 | 0.1764 | 7.76x |
| util.img_as_uint | int8 | uint16 | 1 | native | 3.2212 | 0.2645 | 12.18x |
| util.img_as_int | int8 | int16 | 1 | native | 2.1813 | 0.2318 | 9.41x |
| util.img_as_bool | int8 | bool | 1 | native | 0.0496 | 0.0496 | 1.00x |
| util.img_as_float | uint16 | float64 | 1 | native | 2.8279 | 2.3339 | 1.21x |
| util.img_as_float32 | uint16 | float32 | 1 | native | 1.6072 | 1.0961 | 1.47x |
| util.img_as_float64 | uint16 | float64 | 1 | native | 2.7894 | 2.1953 | 1.27x |
| util.img_as_ubyte | uint16 | uint8 | 1 | native | 0.4650 | 0.1939 | 2.40x |
| util.img_as_uint | uint16 | uint16 | 1 | identity | 0.0007 | 0.0003 | 2.03x |
| util.img_as_int | uint16 | int16 | 1 | native | 0.1800 | 0.2363 | 0.76x |
| util.img_as_bool | uint16 | bool | 1 | native | 0.0828 | 0.0752 | 1.10x |
| util.img_as_float | int16 | float64 | 1 | native | 3.7567 | 2.1788 | 1.72x |
| util.img_as_float32 | int16 | float32 | 1 | native | 1.1058 | 0.1931 | 5.73x |
| util.img_as_float64 | int16 | float64 | 1 | native | 4.3397 | 2.6493 | 1.64x |
| util.img_as_ubyte | int16 | uint8 | 1 | native | 1.1153 | 0.2306 | 4.84x |
| util.img_as_uint | int16 | uint16 | 1 | native | 4.0270 | 0.2194 | 18.36x |
| util.img_as_int | int16 | int16 | 1 | identity | 0.0006 | 0.0003 | 1.70x |
| util.img_as_bool | int16 | bool | 1 | native | 0.1059 | 0.0842 | 1.26x |
| util.img_as_float | uint32 | float64 | 1 | native | 3.0925 | 2.4518 | 1.26x |
| util.img_as_float32 | uint32 | float32 | 1 | native | 1.7852 | 1.2757 | 1.40x |
| util.img_as_float64 | uint32 | float64 | 1 | native | 2.8537 | 2.0439 | 1.40x |
| util.img_as_ubyte | uint32 | uint8 | 1 | native | 0.8778 | 0.4481 | 1.96x |
| util.img_as_uint | uint32 | uint16 | 1 | native | 0.5280 | 0.2537 | 2.08x |
| util.img_as_int | uint32 | int16 | 1 | native | 0.5441 | 0.2550 | 2.13x |
| util.img_as_bool | uint32 | bool | 1 | native | 0.1616 | 0.1127 | 1.43x |
| util.img_as_float | int32 | float64 | 1 | native | 3.2795 | 2.1598 | 1.52x |
| util.img_as_float32 | int32 | float32 | 1 | native | 1.6064 | 0.5025 | 3.20x |
| util.img_as_float64 | int32 | float64 | 1 | native | 3.7887 | 2.6134 | 1.45x |
| util.img_as_ubyte | int32 | uint8 | 1 | native | 2.0267 | 0.3833 | 5.29x |
| util.img_as_uint | int32 | uint16 | 1 | native | 2.2026 | 0.2932 | 7.51x |
| util.img_as_int | int32 | int16 | 1 | native | 0.6322 | 0.2426 | 2.61x |
| util.img_as_bool | int32 | bool | 1 | native | 0.1382 | 0.1046 | 1.32x |
| util.img_as_float | uint64 | float64 | 1 | native | 6.3580 | 5.6922 | 1.12x |
| util.img_as_float32 | uint64 | float32 | 1 | native | 7.5019 | 4.1947 | 1.79x |
| util.img_as_float64 | uint64 | float64 | 1 | native | 6.8098 | 5.9881 | 1.14x |
| util.img_as_ubyte | uint64 | uint8 | 1 | native | 1.5915 | 1.2839 | 1.24x |
| util.img_as_uint | uint64 | uint16 | 1 | native | 1.8256 | 1.6067 | 1.14x |
| util.img_as_int | uint64 | int16 | 1 | native | 2.0604 | 1.3936 | 1.48x |
| util.img_as_bool | uint64 | bool | 1 | native | 1.0189 | 0.9103 | 1.12x |
| util.img_as_float | int64 | float64 | 1 | native | 5.0282 | 3.4973 | 1.44x |
| util.img_as_float32 | int64 | float32 | 1 | native | 5.6314 | 0.9078 | 6.20x |
| util.img_as_float64 | int64 | float64 | 1 | native | 4.0823 | 2.6959 | 1.51x |
| util.img_as_ubyte | int64 | uint8 | 1 | native | 3.4078 | 1.3926 | 2.45x |
| util.img_as_uint | int64 | uint16 | 1 | native | 5.0934 | 1.7956 | 2.84x |
| util.img_as_int | int64 | int16 | 1 | native | 2.8870 | 1.5273 | 1.89x |
| util.img_as_bool | int64 | bool | 1 | native | 0.4620 | 0.5296 | 0.87x |
