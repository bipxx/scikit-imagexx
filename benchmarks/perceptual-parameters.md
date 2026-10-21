# Perceptual color parameter performance

Strict native dispatch versus original scikit-image. Values, dtype and warnings are checked before timing. Seven warmed allocation-inclusive samples; one-thread reference. UserWarning is ignored during timing, while native negative-Z counting remains included. Speedup = reference/current; slower cases remain visible.

| API | Input | Workload | Workers | Original ms | C++ ms | Speedup |
|---|---|---|---:|---:|---:|---:|
| color.rgb2hsv | float32 | grayscale | 1 | 35.8718 | 0.4566 | 78.56x |
| color.rgb2hsv | float32 | grayscale | 4 | 30.8283 | 0.1494 | 206.31x |
| color.hsv2rgb | float32 | wrapped-hue | 1 | 25.8511 | 0.5721 | 45.19x |
| color.hsv2rgb | float32 | wrapped-hue | 4 | 25.9570 | 0.5491 | 47.27x |
| color.xyz2lab | float32 | D50/10 | 1 | 16.5286 | 6.4465 | 2.56x |
| color.xyz2lab | float32 | D50/10 | 4 | 15.9585 | 2.6203 | 6.09x |
| color.lab2xyz | float32 | D50/10 | 1 | 14.4189 | 1.0508 | 13.72x |
| color.lab2xyz | float32 | D50/10 | 4 | 13.3898 | 0.2662 | 50.30x |
| color.rgb2lab | float32 | D50/10 | 1 | 24.3889 | 10.6499 | 2.29x |
| color.rgb2lab | float32 | D50/10 | 4 | 27.3882 | 3.3086 | 8.28x |
| color.lab2rgb | float32 | D50/10 | 1 | 22.9440 | 6.4314 | 3.57x |
| color.lab2rgb | float32 | D50/10 | 4 | 24.1234 | 2.5947 | 9.30x |
| color.lab2lch | float32 | four-channels | 1 | 6.6877 | 2.8923 | 2.31x |
| color.lab2lch | float32 | four-channels | 4 | 7.8711 | 1.8282 | 4.31x |
| color.lch2lab | float32 | four-channels | 1 | 1.9739 | 1.5085 | 1.31x |
| color.lch2lab | float32 | four-channels | 4 | 3.2322 | 1.4852 | 2.18x |
| color.lab2xyz | float32 | negative-Z | 1 | 10.3260 | 0.5866 | 17.60x |
| color.lab2xyz | float32 | negative-Z | 4 | 9.1605 | 0.3369 | 27.19x |
| color.lab2rgb | float32 | negative-Z | 1 | 27.6506 | 5.9963 | 4.61x |
| color.lab2rgb | float32 | negative-Z | 4 | 19.1715 | 2.3230 | 8.25x |
| color.rgb2hsv | float64 | grayscale | 1 | 42.2180 | 1.1425 | 36.95x |
| color.rgb2hsv | float64 | grayscale | 4 | 48.5261 | 1.5340 | 31.63x |
| color.hsv2rgb | float64 | wrapped-hue | 1 | 47.1862 | 1.7442 | 27.05x |
| color.hsv2rgb | float64 | wrapped-hue | 4 | 45.7585 | 1.5165 | 30.17x |
| color.xyz2lab | float64 | D50/10 | 1 | 18.9933 | 5.5433 | 3.43x |
| color.xyz2lab | float64 | D50/10 | 4 | 20.7529 | 2.5612 | 8.10x |
| color.lab2xyz | float64 | D50/10 | 1 | 20.2990 | 1.4294 | 14.20x |
| color.lab2xyz | float64 | D50/10 | 4 | 19.6568 | 1.6592 | 11.85x |
| color.rgb2lab | float64 | D50/10 | 1 | 39.9075 | 14.9983 | 2.66x |
| color.rgb2lab | float64 | D50/10 | 4 | 42.6993 | 4.4355 | 9.63x |
| color.lab2rgb | float64 | D50/10 | 1 | 39.4966 | 7.3403 | 5.38x |
| color.lab2rgb | float64 | D50/10 | 4 | 36.5674 | 2.7751 | 13.18x |
| color.lab2lch | float64 | four-channels | 1 | 10.9322 | 4.6689 | 2.34x |
| color.lab2lch | float64 | four-channels | 4 | 9.8644 | 2.1765 | 4.53x |
| color.lch2lab | float64 | four-channels | 1 | 8.5131 | 3.0799 | 2.76x |
| color.lch2lab | float64 | four-channels | 4 | 9.9494 | 2.1399 | 4.65x |
| color.lab2xyz | float64 | negative-Z | 1 | 17.5920 | 1.2380 | 14.21x |
| color.lab2xyz | float64 | negative-Z | 4 | 14.6188 | 0.6153 | 23.76x |
| color.lab2rgb | float64 | negative-Z | 1 | 25.8683 | 7.4433 | 3.48x |
| color.lab2rgb | float64 | negative-Z | 4 | 27.2023 | 2.7990 | 9.72x |
