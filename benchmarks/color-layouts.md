# Color dtype and layout performance

Every call is checked against original scikit-image before timing. Fallback is disabled. Seven warmed allocation-inclusive samples include channel movement, contiguous copies and integer normalization. The reference uses one thread. Speedup = reference/current; slower cases remain visible.

| API | Input | Layout | Workers | Original ms | C++ ms | Speedup |
|---|---|---|---:|---:|---:|---:|
| color.rgb2yuv | uint16 | strided | 1 | 8.2134 | 4.8747 | 1.68x |
| color.yuv2rgb | uint16 | strided | 1 | 8.4145 | 5.0058 | 1.68x |
| color.rgb2yiq | uint16 | strided | 1 | 8.4263 | 4.4310 | 1.90x |
| color.yiq2rgb | uint16 | strided | 1 | 8.7957 | 8.4179 | 1.04x |
| color.rgb2ypbpr | uint16 | strided | 1 | 11.5371 | 6.2997 | 1.83x |
| color.ypbpr2rgb | uint16 | strided | 1 | 7.4757 | 3.5372 | 2.11x |
| color.rgb2ycbcr | uint16 | strided | 1 | 11.5970 | 3.7060 | 3.13x |
| color.ycbcr2rgb | uint16 | strided | 1 | 10.3070 | 4.9465 | 2.08x |
| color.rgb2ydbdr | uint16 | strided | 1 | 6.9180 | 3.1614 | 2.19x |
| color.ydbdr2rgb | uint16 | strided | 1 | 6.4031 | 3.3907 | 1.89x |
| color.rgb2rgbcie | uint16 | strided | 1 | 7.4588 | 2.9666 | 2.51x |
| color.rgbcie2rgb | uint16 | strided | 1 | 6.9376 | 2.9757 | 2.33x |
| color.rgb2xyz | uint16 | strided | 1 | 44.1843 | 15.5074 | 2.85x |
| color.xyz2rgb | uint16 | strided | 1 | 48.0593 | 13.6646 | 3.52x |
| color.rgb2yuv | uint16 | channel-first | 1 | 8.2626 | 5.9192 | 1.40x |
| color.yuv2rgb | uint16 | channel-first | 1 | 5.8967 | 2.9318 | 2.01x |
| color.rgb2yiq | uint16 | channel-first | 1 | 5.7950 | 2.9654 | 1.95x |
| color.yiq2rgb | uint16 | channel-first | 1 | 6.1340 | 3.3296 | 1.84x |
| color.rgb2ypbpr | uint16 | channel-first | 1 | 6.2195 | 3.0390 | 2.05x |
| color.ypbpr2rgb | uint16 | channel-first | 1 | 5.6151 | 3.4250 | 1.64x |
| color.rgb2ycbcr | uint16 | channel-first | 1 | 12.9054 | 5.1288 | 2.52x |
| color.ycbcr2rgb | uint16 | channel-first | 1 | 9.3592 | 3.1845 | 2.94x |
| color.rgb2ydbdr | uint16 | channel-first | 1 | 6.2127 | 2.8841 | 2.15x |
| color.ydbdr2rgb | uint16 | channel-first | 1 | 9.2333 | 5.6528 | 1.63x |
| color.rgb2rgbcie | uint16 | channel-first | 1 | 6.8489 | 3.0725 | 2.23x |
| color.rgbcie2rgb | uint16 | channel-first | 1 | 7.3744 | 3.2833 | 2.25x |
| color.rgb2xyz | uint16 | channel-first | 1 | 48.6380 | 13.6583 | 3.56x |
| color.xyz2rgb | uint16 | channel-first | 1 | 52.4774 | 11.6762 | 4.49x |
| color.rgb2yuv | float32 | strided | 1 | 5.6351 | 6.7406 | 0.84x |
| color.yuv2rgb | float32 | strided | 1 | 3.0782 | 3.2812 | 0.94x |
| color.rgb2yiq | float32 | strided | 1 | 5.1927 | 4.7052 | 1.10x |
| color.yiq2rgb | float32 | strided | 1 | 2.5451 | 2.7123 | 0.94x |
| color.rgb2ypbpr | float32 | strided | 1 | 4.2599 | 3.8897 | 1.10x |
| color.ypbpr2rgb | float32 | strided | 1 | 4.8352 | 4.1887 | 1.15x |
| color.rgb2ycbcr | float32 | strided | 1 | 6.3577 | 3.0915 | 2.06x |
| color.ycbcr2rgb | float32 | strided | 1 | 9.4985 | 3.6625 | 2.59x |
| color.rgb2ydbdr | float32 | strided | 1 | 2.2041 | 2.0071 | 1.10x |
| color.ydbdr2rgb | float32 | strided | 1 | 3.5472 | 2.7681 | 1.28x |
| color.rgb2rgbcie | float32 | strided | 1 | 2.1077 | 1.3239 | 1.59x |
| color.rgbcie2rgb | float32 | strided | 1 | 3.2028 | 2.9076 | 1.10x |
| color.rgb2xyz | float32 | strided | 1 | 15.5233 | 5.7535 | 2.70x |
| color.xyz2rgb | float32 | strided | 1 | 21.5939 | 6.2637 | 3.45x |
| color.rgb2yuv | float32 | channel-first | 1 | 1.9161 | 1.5416 | 1.24x |
| color.yuv2rgb | float32 | channel-first | 1 | 1.6666 | 1.3114 | 1.27x |
| color.rgb2yiq | float32 | channel-first | 1 | 1.7605 | 1.3291 | 1.32x |
| color.yiq2rgb | float32 | channel-first | 1 | 1.6399 | 1.1814 | 1.39x |
| color.rgb2ypbpr | float32 | channel-first | 1 | 1.9333 | 1.3737 | 1.41x |
| color.ypbpr2rgb | float32 | channel-first | 1 | 1.6644 | 1.1948 | 1.39x |
| color.rgb2ycbcr | float32 | channel-first | 1 | 2.3986 | 1.2703 | 1.89x |
| color.ycbcr2rgb | float32 | channel-first | 1 | 3.8070 | 0.6539 | 5.82x |
| color.rgb2ydbdr | float32 | channel-first | 1 | 2.2507 | 1.9442 | 1.16x |
| color.ydbdr2rgb | float32 | channel-first | 1 | 1.2296 | 0.6240 | 1.97x |
| color.rgb2rgbcie | float32 | channel-first | 1 | 2.3510 | 1.5624 | 1.50x |
| color.rgbcie2rgb | float32 | channel-first | 1 | 1.4129 | 0.7685 | 1.84x |
| color.rgb2xyz | float32 | channel-first | 1 | 17.4100 | 5.6176 | 3.10x |
| color.xyz2rgb | float32 | channel-first | 1 | 20.5230 | 6.4770 | 3.17x |
