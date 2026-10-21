# Conversion performance across integer widths

Each case requires exact values, dtype, and identity behavior against scikit-image. Seven warmed, allocation-inclusive samples; strict dispatch, single-thread reference. Identity/copy rows measure wrapper or copying costs, not a conversion-kernel speedup. All ratios below one remain visible.

| API | Input | Output | Workers | Execution | Original ms | Current ms | Speedup |
|---|---|---|---:|---|---:|---:|---:|
| util.img_as_float | bool | float64 | 1 | native | 2.5157 | 2.0831 | 1.21x |
| util.img_as_float32 | bool | float32 | 1 | native | 0.4355 | 0.1880 | 2.32x |
| util.img_as_float64 | bool | float64 | 1 | native | 2.4910 | 2.4097 | 1.03x |
| util.img_as_ubyte | bool | uint8 | 1 | native | 0.5338 | 0.0469 | 11.39x |
| util.img_as_uint | bool | uint16 | 1 | native | 0.5873 | 0.2414 | 2.43x |
| util.img_as_int | bool | int16 | 1 | native | 0.5463 | 0.2347 | 2.33x |
| util.img_as_bool | bool | bool | 1 | identity | 0.0006 | 0.0003 | 2.37x |
| util.img_as_float | int8 | float64 | 1 | native | 4.3264 | 2.4280 | 1.78x |
| util.img_as_float32 | int8 | float32 | 1 | native | 0.7210 | 0.2094 | 3.44x |
| util.img_as_float64 | int8 | float64 | 1 | native | 3.8166 | 2.2878 | 1.67x |
| util.img_as_ubyte | int8 | uint8 | 1 | native | 1.4074 | 0.2113 | 6.66x |
| util.img_as_uint | int8 | uint16 | 1 | native | 3.1751 | 0.2527 | 12.56x |
| util.img_as_int | int8 | int16 | 1 | native | 1.8294 | 0.2314 | 7.90x |
| util.img_as_bool | int8 | bool | 1 | native | 0.0639 | 0.0531 | 1.20x |
| util.img_as_float | uint16 | float64 | 1 | native | 3.0997 | 2.7773 | 1.12x |
| util.img_as_float32 | uint16 | float32 | 1 | native | 1.6670 | 1.2635 | 1.32x |
| util.img_as_float64 | uint16 | float64 | 1 | native | 2.8054 | 2.3872 | 1.18x |
| util.img_as_ubyte | uint16 | uint8 | 1 | native | 0.3834 | 0.1779 | 2.15x |
| util.img_as_uint | uint16 | uint16 | 1 | identity | 0.0005 | 0.0003 | 1.91x |
| util.img_as_int | uint16 | int16 | 1 | native | 0.1446 | 0.2018 | 0.72x |
| util.img_as_bool | uint16 | bool | 1 | native | 0.0733 | 0.0649 | 1.13x |
| util.img_as_float | int16 | float64 | 1 | native | 4.4446 | 2.6546 | 1.67x |
| util.img_as_float32 | int16 | float32 | 1 | native | 0.8657 | 0.2394 | 3.62x |
| util.img_as_float64 | int16 | float64 | 1 | native | 4.1513 | 2.3448 | 1.77x |
| util.img_as_ubyte | int16 | uint8 | 1 | native | 0.9819 | 0.2198 | 4.47x |
| util.img_as_uint | int16 | uint16 | 1 | native | 3.5580 | 0.2343 | 15.18x |
| util.img_as_int | int16 | int16 | 1 | identity | 0.0005 | 0.0003 | 1.50x |
| util.img_as_bool | int16 | bool | 1 | native | 0.0637 | 0.0609 | 1.05x |
| util.img_as_float | uint32 | float64 | 1 | native | 2.8102 | 2.6206 | 1.07x |
| util.img_as_float32 | uint32 | float32 | 1 | native | 0.7837 | 0.2652 | 2.96x |
| util.img_as_float64 | uint32 | float64 | 1 | native | 2.5956 | 2.2114 | 1.17x |
| util.img_as_ubyte | uint32 | uint8 | 1 | native | 0.5572 | 0.2445 | 2.28x |
| util.img_as_uint | uint32 | uint16 | 1 | native | 0.7353 | 0.3011 | 2.44x |
| util.img_as_int | uint32 | int16 | 1 | native | 0.5908 | 0.2452 | 2.41x |
| util.img_as_bool | uint32 | bool | 1 | native | 0.5440 | 0.4130 | 1.32x |
| util.img_as_float | int32 | float64 | 1 | native | 4.4761 | 2.7226 | 1.64x |
| util.img_as_float32 | int32 | float32 | 1 | native | 1.0101 | 0.5345 | 1.89x |
| util.img_as_float64 | int32 | float64 | 1 | native | 4.5352 | 2.9261 | 1.55x |
| util.img_as_ubyte | int32 | uint8 | 1 | native | 2.2929 | 0.3741 | 6.13x |
| util.img_as_uint | int32 | uint16 | 1 | native | 2.3954 | 0.2575 | 9.30x |
| util.img_as_int | int32 | int16 | 1 | native | 0.5714 | 0.2602 | 2.20x |
| util.img_as_bool | int32 | bool | 1 | native | 0.1324 | 0.1120 | 1.18x |
| util.img_as_float | uint64 | float64 | 1 | native | 6.1579 | 5.1432 | 1.20x |
| util.img_as_float32 | uint64 | float32 | 1 | native | 6.6378 | 3.7894 | 1.75x |
| util.img_as_float64 | uint64 | float64 | 1 | native | 6.3356 | 5.1848 | 1.22x |
| util.img_as_ubyte | uint64 | uint8 | 1 | native | 1.2610 | 1.1101 | 1.14x |
| util.img_as_uint | uint64 | uint16 | 1 | native | 1.5457 | 1.1009 | 1.40x |
| util.img_as_int | uint64 | int16 | 1 | native | 1.9081 | 1.3916 | 1.37x |
| util.img_as_bool | uint64 | bool | 1 | native | 0.3721 | 0.2415 | 1.54x |
| util.img_as_float | int64 | float64 | 1 | native | 4.0053 | 2.8218 | 1.42x |
| util.img_as_float32 | int64 | float32 | 1 | native | 5.9136 | 0.9312 | 6.35x |
| util.img_as_float64 | int64 | float64 | 1 | native | 4.6157 | 2.9075 | 1.59x |
| util.img_as_ubyte | int64 | uint8 | 1 | native | 2.6064 | 0.9247 | 2.82x |
| util.img_as_uint | int64 | uint16 | 1 | native | 5.4129 | 1.8055 | 3.00x |
| util.img_as_int | int64 | int16 | 1 | native | 2.5457 | 1.7517 | 1.45x |
| util.img_as_bool | int64 | bool | 1 | native | 0.4813 | 0.5042 | 0.95x |
