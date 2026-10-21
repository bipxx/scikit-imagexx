# Conversion performance across integer widths

Each case requires exact values, dtype, and identity behavior against scikit-image. Seven warmed, allocation-inclusive samples; strict dispatch, single-thread reference. Identity/copy rows measure wrapper or copying costs, not a conversion-kernel speedup. All ratios below one remain visible.

| API | Input | Output | Workers | Execution | Original ms | Current ms | Speedup |
|---|---|---|---:|---|---:|---:|---:|
| util.img_as_float | bool | float64 | 1 | native | 2.4932 | 2.5557 | 0.98x |
| util.img_as_float32 | bool | float32 | 1 | native | 0.4138 | 0.1692 | 2.45x |
| util.img_as_float64 | bool | float64 | 1 | native | 2.1651 | 2.2959 | 0.94x |
| util.img_as_ubyte | bool | uint8 | 1 | native | 0.5503 | 0.0609 | 9.04x |
| util.img_as_uint | bool | uint16 | 1 | native | 0.5653 | 0.2558 | 2.21x |
| util.img_as_int | bool | int16 | 1 | native | 0.6095 | 0.2388 | 2.55x |
| util.img_as_bool | bool | bool | 1 | identity | 0.0005 | 0.0003 | 1.67x |
| util.img_as_float | int8 | float64 | 1 | native | 4.4053 | 2.3223 | 1.90x |
| util.img_as_float32 | int8 | float32 | 1 | native | 0.9681 | 0.2434 | 3.98x |
| util.img_as_float64 | int8 | float64 | 1 | native | 4.3187 | 2.6300 | 1.64x |
| util.img_as_ubyte | int8 | uint8 | 1 | native | 1.5500 | 0.1798 | 8.62x |
| util.img_as_uint | int8 | uint16 | 1 | native | 3.1560 | 0.2422 | 13.03x |
| util.img_as_int | int8 | int16 | 1 | native | 2.2614 | 0.1961 | 11.53x |
| util.img_as_bool | int8 | bool | 1 | native | 0.0555 | 0.0419 | 1.32x |
| util.img_as_float | uint16 | float64 | 1 | native | 2.7719 | 2.3498 | 1.18x |
| util.img_as_float32 | uint16 | float32 | 1 | native | 1.5570 | 1.2128 | 1.28x |
| util.img_as_float64 | uint16 | float64 | 1 | native | 2.8015 | 2.3956 | 1.17x |
| util.img_as_ubyte | uint16 | uint8 | 1 | native | 0.6351 | 0.4563 | 1.39x |
| util.img_as_uint | uint16 | uint16 | 1 | identity | 0.0005 | 0.0003 | 1.80x |
| util.img_as_int | uint16 | int16 | 1 | native | 0.1404 | 0.1787 | 0.79x |
| util.img_as_bool | uint16 | bool | 1 | native | 0.0644 | 0.0762 | 0.84x |
| util.img_as_float | int16 | float64 | 1 | native | 3.5021 | 2.3310 | 1.50x |
| util.img_as_float32 | int16 | float32 | 1 | native | 0.8558 | 0.1616 | 5.30x |
| util.img_as_float64 | int16 | float64 | 1 | native | 4.1395 | 2.6078 | 1.59x |
| util.img_as_ubyte | int16 | uint8 | 1 | native | 0.9415 | 0.1970 | 4.78x |
| util.img_as_uint | int16 | uint16 | 1 | native | 4.4279 | 0.2347 | 18.86x |
| util.img_as_int | int16 | int16 | 1 | identity | 0.0006 | 0.0004 | 1.56x |
| util.img_as_bool | int16 | bool | 1 | native | 0.2073 | 0.1023 | 2.03x |
| util.img_as_float | uint32 | float64 | 1 | native | 4.6076 | 4.9400 | 0.93x |
| util.img_as_float32 | uint32 | float32 | 1 | native | 1.3946 | 0.9998 | 1.39x |
| util.img_as_float64 | uint32 | float64 | 1 | native | 3.9463 | 3.2526 | 1.21x |
| util.img_as_ubyte | uint32 | uint8 | 1 | native | 1.1410 | 0.4892 | 2.33x |
| util.img_as_uint | uint32 | uint16 | 1 | native | 2.3165 | 2.0740 | 1.12x |
| util.img_as_int | uint32 | int16 | 1 | native | 1.2958 | 0.4667 | 2.78x |
| util.img_as_bool | uint32 | bool | 1 | native | 0.3788 | 0.1469 | 2.58x |
| util.img_as_float | int32 | float64 | 1 | native | 5.5971 | 3.1356 | 1.79x |
| util.img_as_float32 | int32 | float32 | 1 | native | 3.2631 | 2.0871 | 1.56x |
| util.img_as_float64 | int32 | float64 | 1 | native | 5.2840 | 3.3865 | 1.56x |
| util.img_as_ubyte | int32 | uint8 | 1 | native | 2.7625 | 0.5177 | 5.34x |
| util.img_as_uint | int32 | uint16 | 1 | native | 3.5460 | 0.4199 | 8.44x |
| util.img_as_int | int32 | int16 | 1 | native | 1.2857 | 0.7065 | 1.82x |
| util.img_as_bool | int32 | bool | 1 | native | 0.8087 | 0.7132 | 1.13x |
| util.img_as_float | uint64 | float64 | 1 | native | 7.7243 | 7.0090 | 1.10x |
| util.img_as_float32 | uint64 | float32 | 1 | native | 12.6351 | 5.5510 | 2.28x |
| util.img_as_float64 | uint64 | float64 | 1 | native | 9.7893 | 7.8513 | 1.25x |
| util.img_as_ubyte | uint64 | uint8 | 1 | native | 5.9445 | 4.2735 | 1.39x |
| util.img_as_uint | uint64 | uint16 | 1 | native | 5.8922 | 5.5071 | 1.07x |
| util.img_as_int | uint64 | int16 | 1 | native | 4.8319 | 6.3659 | 0.76x |
| util.img_as_bool | uint64 | bool | 1 | native | 2.9970 | 2.8087 | 1.07x |
| util.img_as_float | int64 | float64 | 1 | native | 8.9541 | 5.7671 | 1.55x |
| util.img_as_float32 | int64 | float32 | 1 | native | 10.6730 | 2.5891 | 4.12x |
| util.img_as_float64 | int64 | float64 | 1 | native | 7.1449 | 4.6216 | 1.55x |
| util.img_as_ubyte | int64 | uint8 | 1 | native | 5.5942 | 3.5118 | 1.59x |
| util.img_as_uint | int64 | uint16 | 1 | native | 5.8996 | 3.2601 | 1.81x |
| util.img_as_int | int64 | int16 | 1 | native | 3.1075 | 1.9477 | 1.60x |
| util.img_as_bool | int64 | bool | 1 | native | 0.9840 | 0.6709 | 1.47x |
