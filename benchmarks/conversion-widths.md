# Conversion performance across integer widths

Each case requires exact values, dtype, and identity behavior against scikit-image. Seven warmed, allocation-inclusive samples; strict dispatch, single-thread reference. Identity/copy rows measure wrapper or copying costs, not a conversion-kernel speedup. All ratios below one remain visible.

| API | Input | Output | Workers | Execution | Original ms | Current ms | Speedup |
|---|---|---|---:|---|---:|---:|---:|
| util.img_as_float | bool | float64 | 1 | native | 2.0971 | 2.1012 | 1.00x |
| util.img_as_float32 | bool | float32 | 1 | native | 0.3994 | 0.1533 | 2.61x |
| util.img_as_float64 | bool | float64 | 1 | native | 2.1287 | 2.1081 | 1.01x |
| util.img_as_ubyte | bool | uint8 | 1 | native | 0.5149 | 0.0454 | 11.34x |
| util.img_as_uint | bool | uint16 | 1 | native | 0.5508 | 0.2694 | 2.04x |
| util.img_as_int | bool | int16 | 1 | native | 0.5516 | 0.2428 | 2.27x |
| util.img_as_bool | bool | bool | 1 | identity | 0.0005 | 0.0003 | 2.02x |
| util.img_as_float | int8 | float64 | 1 | native | 2.9994 | 2.1278 | 1.41x |
| util.img_as_float32 | int8 | float32 | 1 | native | 0.6362 | 0.2147 | 2.96x |
| util.img_as_float64 | int8 | float64 | 1 | native | 3.1114 | 2.0353 | 1.53x |
| util.img_as_ubyte | int8 | uint8 | 1 | native | 1.2911 | 0.1871 | 6.90x |
| util.img_as_uint | int8 | uint16 | 1 | native | 2.6817 | 0.2780 | 9.65x |
| util.img_as_int | int8 | int16 | 1 | native | 1.8001 | 0.2266 | 7.94x |
| util.img_as_bool | int8 | bool | 1 | native | 0.0481 | 0.0465 | 1.04x |
| util.img_as_float | uint16 | float64 | 1 | native | 2.8921 | 2.6034 | 1.11x |
| util.img_as_float32 | uint16 | float32 | 1 | native | 1.7881 | 1.4763 | 1.21x |
| util.img_as_float64 | uint16 | float64 | 1 | native | 3.5951 | 2.8620 | 1.26x |
| util.img_as_ubyte | uint16 | uint8 | 1 | native | 0.4531 | 0.1722 | 2.63x |
| util.img_as_uint | uint16 | uint16 | 1 | identity | 0.0005 | 0.0003 | 1.78x |
| util.img_as_int | uint16 | int16 | 1 | native | 0.1614 | 0.2170 | 0.74x |
| util.img_as_bool | uint16 | bool | 1 | native | 0.0707 | 0.0698 | 1.01x |
| util.img_as_float | int16 | float64 | 1 | native | 2.9843 | 1.9511 | 1.53x |
| util.img_as_float32 | int16 | float32 | 1 | native | 0.6424 | 0.1567 | 4.10x |
| util.img_as_float64 | int16 | float64 | 1 | native | 3.7570 | 2.1691 | 1.73x |
| util.img_as_ubyte | int16 | uint8 | 1 | native | 0.7931 | 0.1955 | 4.06x |
| util.img_as_uint | int16 | uint16 | 1 | native | 2.8757 | 0.1868 | 15.39x |
| util.img_as_int | int16 | int16 | 1 | identity | 0.0005 | 0.0003 | 1.97x |
| util.img_as_bool | int16 | bool | 1 | native | 0.0641 | 0.0631 | 1.02x |
| util.img_as_float | uint32 | float64 | 1 | native | 2.8911 | 2.5905 | 1.12x |
| util.img_as_float32 | uint32 | float32 | 1 | native | 1.7885 | 1.2230 | 1.46x |
| util.img_as_float64 | uint32 | float64 | 1 | native | 2.4976 | 1.9533 | 1.28x |
| util.img_as_ubyte | uint32 | uint8 | 1 | native | 0.6604 | 0.4359 | 1.52x |
| util.img_as_uint | uint32 | uint16 | 1 | native | 0.4668 | 0.1963 | 2.38x |
| util.img_as_int | uint32 | int16 | 1 | native | 0.4735 | 0.1932 | 2.45x |
| util.img_as_bool | uint32 | bool | 1 | native | 0.2997 | 0.1388 | 2.16x |
| util.img_as_float | int32 | float64 | 1 | native | 3.0265 | 2.0808 | 1.45x |
| util.img_as_float32 | int32 | float32 | 1 | native | 1.5589 | 1.0242 | 1.52x |
| util.img_as_float64 | int32 | float64 | 1 | native | 3.3399 | 2.2372 | 1.49x |
| util.img_as_ubyte | int32 | uint8 | 1 | native | 2.1789 | 0.4661 | 4.67x |
| util.img_as_uint | int32 | uint16 | 1 | native | 1.6241 | 0.2386 | 6.81x |
| util.img_as_int | int32 | int16 | 1 | native | 0.5208 | 0.1916 | 2.72x |
| util.img_as_bool | int32 | bool | 1 | native | 0.1060 | 0.1073 | 0.99x |
| util.img_as_float | uint64 | float64 | 1 | native | 5.6013 | 4.8626 | 1.15x |
| util.img_as_float32 | uint64 | float32 | 1 | native | 6.3573 | 3.7115 | 1.71x |
| util.img_as_float64 | uint64 | float64 | 1 | native | 5.6793 | 4.9918 | 1.14x |
| util.img_as_ubyte | uint64 | uint8 | 1 | native | 1.0377 | 1.0079 | 1.03x |
| util.img_as_uint | uint64 | uint16 | 1 | native | 0.9848 | 0.7335 | 1.34x |
| util.img_as_int | uint64 | int16 | 1 | native | 0.9342 | 0.8588 | 1.09x |
| util.img_as_bool | uint64 | bool | 1 | native | 0.2653 | 0.2191 | 1.21x |
| util.img_as_float | int64 | float64 | 1 | native | 3.1268 | 2.1160 | 1.48x |
| util.img_as_float32 | int64 | float32 | 1 | native | 3.6718 | 0.5845 | 6.28x |
| util.img_as_float64 | int64 | float64 | 1 | native | 3.1805 | 2.2631 | 1.41x |
| util.img_as_ubyte | int64 | uint8 | 1 | native | 2.1145 | 0.8144 | 2.60x |
| util.img_as_uint | int64 | uint16 | 1 | native | 2.5556 | 0.7443 | 3.43x |
| util.img_as_int | int64 | int16 | 1 | native | 1.7163 | 0.9014 | 1.90x |
| util.img_as_bool | int64 | bool | 1 | native | 0.2581 | 0.2546 | 1.01x |
