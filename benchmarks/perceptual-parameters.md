# Perceptual color parameter performance

Strict native dispatch versus original scikit-image. Values, dtype and warnings are checked before timing. Seven warmed allocation-inclusive samples; one-thread reference. UserWarning is ignored during timing, while native negative-Z counting remains included. Float32 inverse Luv checks each rounding stage and a propagated direct RGB bound; the JSON records its contract and measured errors. Speedup = reference/current; slower cases remain visible.

| API | Input | Workload | Workers | Original ms | C++ ms | Speedup |
|---|---|---|---:|---:|---:|---:|
| color.rgb2hsv | float32 | grayscale | 1 | 44.5384 | 0.5594 | 79.62x |
| color.hsv2rgb | float32 | wrapped-hue | 1 | 27.2786 | 0.5654 | 48.24x |
| color.xyz2lab | float32 | D50/10 | 1 | 13.2755 | 5.0152 | 2.65x |
| color.lab2xyz | float32 | D50/10 | 1 | 13.3975 | 0.5149 | 26.02x |
| color.rgb2lab | float32 | D50/10 | 1 | 38.0965 | 20.6765 | 1.84x |
| color.lab2rgb | float32 | D50/10 | 1 | 34.2028 | 10.2332 | 3.34x |
| color.lab2lch | float32 | four-channels | 1 | 10.1090 | 3.7553 | 2.69x |
| color.lch2lab | float32 | four-channels | 1 | 4.5517 | 3.3998 | 1.34x |
| color.lab2xyz | float32 | negative-Z | 1 | 12.1494 | 1.2455 | 9.75x |
| color.lab2rgb | float32 | negative-Z | 1 | 23.6964 | 3.3853 | 7.00x |
| color.xyz2luv | float32 | D50/10 | 1 | 6.0495 | 2.1907 | 2.76x |
| color.luv2xyz | float32 | D50/10 | 1 | 7.8673 | 3.7357 | 2.11x |
| color.rgb2luv | float32 | dark-sRGB | 1 | 10.9228 | 0.3889 | 28.09x |
| color.luv2rgb | float32 | low-lightness | 1 | 23.1842 | 8.3134 | 2.79x |
| color.luv2rgb | float32 | Luv-midtones | 1 | 21.3431 | 9.2821 | 2.30x |
| color.rgb2hsv | float64 | grayscale | 1 | 38.8984 | 1.1106 | 35.03x |
| color.hsv2rgb | float64 | wrapped-hue | 1 | 50.6935 | 2.5632 | 19.78x |
| color.xyz2lab | float64 | D50/10 | 1 | 23.2098 | 7.4713 | 3.11x |
| color.lab2xyz | float64 | D50/10 | 1 | 27.8435 | 3.3472 | 8.32x |
| color.rgb2lab | float64 | D50/10 | 1 | 51.6325 | 17.0414 | 3.03x |
| color.lab2rgb | float64 | D50/10 | 1 | 41.1402 | 9.4721 | 4.34x |
| color.lab2lch | float64 | four-channels | 1 | 10.9208 | 4.8463 | 2.25x |
| color.lch2lab | float64 | four-channels | 1 | 10.0931 | 3.8215 | 2.64x |
| color.lab2xyz | float64 | negative-Z | 1 | 16.1986 | 1.2286 | 13.18x |
| color.lab2rgb | float64 | negative-Z | 1 | 30.6860 | 3.9040 | 7.86x |
| color.xyz2luv | float64 | D50/10 | 1 | 17.3718 | 4.4922 | 3.87x |
| color.luv2xyz | float64 | D50/10 | 1 | 12.1089 | 0.9438 | 12.83x |
| color.rgb2luv | float64 | dark-sRGB | 1 | 16.9935 | 2.1500 | 7.90x |
| color.luv2rgb | float64 | low-lightness | 1 | 26.8698 | 6.4290 | 4.18x |
| color.luv2rgb | float64 | Luv-midtones | 1 | 30.6289 | 6.4683 | 4.74x |
