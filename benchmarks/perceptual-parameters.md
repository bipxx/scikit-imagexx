# Perceptual color parameter performance

Strict native dispatch versus original scikit-image. Values, dtype and warnings are checked before timing. Seven warmed allocation-inclusive samples; one-thread reference. UserWarning is ignored during timing, while native negative-Z counting remains included. Float32 inverse Luv checks each rounding stage and a propagated direct RGB bound; the JSON records its contract and measured errors. Speedup = reference/current; slower cases remain visible.

| API | Input | Workload | Workers | Original ms | C++ ms | Speedup |
|---|---|---|---:|---:|---:|---:|
| color.rgb2hsv | float32 | grayscale | 1 | 32.1149 | 0.3742 | 85.83x |
| color.hsv2rgb | float32 | wrapped-hue | 1 | 24.8614 | 0.5066 | 49.07x |
| color.xyz2lab | float32 | D50/10 | 1 | 13.7804 | 5.2966 | 2.60x |
| color.lab2xyz | float32 | D50/10 | 1 | 13.6533 | 0.5679 | 24.04x |
| color.rgb2lab | float32 | D50/10 | 1 | 22.4967 | 10.2666 | 2.19x |
| color.lab2rgb | float32 | D50/10 | 1 | 21.8936 | 6.0346 | 3.63x |
| color.lab2lch | float32 | four-channels | 1 | 5.8391 | 2.6748 | 2.18x |
| color.lch2lab | float32 | four-channels | 1 | 2.4410 | 2.2042 | 1.11x |
| color.lab2xyz | float32 | negative-Z | 1 | 8.4568 | 1.0203 | 8.29x |
| color.lab2rgb | float32 | negative-Z | 1 | 16.2881 | 3.0995 | 5.26x |
| color.xyz2luv | float32 | D50/10 | 1 | 6.1551 | 2.5076 | 2.45x |
| color.luv2xyz | float32 | D50/10 | 1 | 6.3477 | 2.1625 | 2.94x |
| color.rgb2luv | float32 | dark-sRGB | 1 | 7.2878 | 0.3014 | 24.18x |
| color.luv2rgb | float32 | low-lightness | 1 | 15.3160 | 5.6152 | 2.73x |
| color.luv2rgb | float32 | Luv-midtones | 1 | 20.6129 | 8.2533 | 2.50x |
| color.rgb2hsv | float64 | grayscale | 1 | 38.3394 | 2.0572 | 18.64x |
| color.hsv2rgb | float64 | wrapped-hue | 1 | 49.9845 | 2.4828 | 20.13x |
| color.xyz2lab | float64 | D50/10 | 1 | 19.4314 | 5.6430 | 3.44x |
| color.lab2xyz | float64 | D50/10 | 1 | 19.1522 | 2.0673 | 9.26x |
| color.rgb2lab | float64 | D50/10 | 1 | 37.0178 | 11.9167 | 3.11x |
| color.lab2rgb | float64 | D50/10 | 1 | 33.6659 | 7.0721 | 4.76x |
| color.lab2lch | float64 | four-channels | 1 | 9.5580 | 4.2559 | 2.25x |
| color.lch2lab | float64 | four-channels | 1 | 8.0757 | 2.8657 | 2.82x |
| color.lab2xyz | float64 | negative-Z | 1 | 13.8373 | 1.1454 | 12.08x |
| color.lab2rgb | float64 | negative-Z | 1 | 22.5057 | 3.0389 | 7.41x |
| color.xyz2luv | float64 | D50/10 | 1 | 10.8792 | 2.5302 | 4.30x |
| color.luv2xyz | float64 | D50/10 | 1 | 13.0909 | 0.7788 | 16.81x |
| color.rgb2luv | float64 | dark-sRGB | 1 | 12.7966 | 0.8258 | 15.50x |
| color.luv2rgb | float64 | low-lightness | 1 | 26.0351 | 6.2387 | 4.17x |
| color.luv2rgb | float64 | Luv-midtones | 1 | 30.1624 | 6.2846 | 4.80x |
