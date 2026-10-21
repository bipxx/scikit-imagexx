# Perceptual color parameter performance

Strict native dispatch versus original scikit-image. Values, dtype and warnings are checked before timing. Seven warmed allocation-inclusive samples; one-thread reference. UserWarning is ignored during timing, while native negative-Z counting remains included. Float32 inverse Luv checks each rounding stage and a propagated direct RGB bound; the JSON records its contract and measured errors. Speedup = reference/current; slower cases remain visible.

| API | Input | Workload | Workers | Original ms | C++ ms | Speedup |
|---|---|---|---:|---:|---:|---:|
| color.rgb2hsv | float32 | grayscale | 1 | 32.1458 | 0.3777 | 85.12x |
| color.hsv2rgb | float32 | wrapped-hue | 1 | 25.0862 | 0.5331 | 47.06x |
| color.xyz2lab | float32 | D50/10 | 1 | 16.3396 | 5.3860 | 3.03x |
| color.lab2xyz | float32 | D50/10 | 1 | 13.5253 | 0.5791 | 23.36x |
| color.rgb2lab | float32 | D50/10 | 1 | 25.9024 | 11.1197 | 2.33x |
| color.lab2rgb | float32 | D50/10 | 1 | 25.2822 | 7.3501 | 3.44x |
| color.lab2lch | float32 | four-channels | 1 | 6.4029 | 3.1055 | 2.06x |
| color.lch2lab | float32 | four-channels | 1 | 3.9307 | 2.3255 | 1.69x |
| color.lab2xyz | float32 | negative-Z | 1 | 9.3876 | 1.3056 | 7.19x |
| color.lab2rgb | float32 | negative-Z | 1 | 21.0646 | 3.5176 | 5.99x |
| color.xyz2luv | float32 | D50/10 | 1 | 8.9657 | 3.0870 | 2.90x |
| color.luv2xyz | float32 | D50/10 | 1 | 6.1560 | 3.0158 | 2.04x |
| color.rgb2luv | float32 | dark-sRGB | 1 | 8.2584 | 0.3960 | 20.85x |
| color.luv2rgb | float32 | low-lightness | 1 | 17.7375 | 7.2197 | 2.46x |
| color.luv2rgb | float32 | Luv-midtones | 1 | 22.0083 | 9.3237 | 2.36x |
| color.rgb2hsv | float64 | grayscale | 1 | 40.6961 | 2.5289 | 16.09x |
| color.hsv2rgb | float64 | wrapped-hue | 1 | 47.0569 | 2.5888 | 18.18x |
| color.xyz2lab | float64 | D50/10 | 1 | 27.5042 | 8.3725 | 3.29x |
| color.lab2xyz | float64 | D50/10 | 1 | 25.4149 | 2.5340 | 10.03x |
| color.rgb2lab | float64 | D50/10 | 1 | 42.9404 | 14.2871 | 3.01x |
| color.lab2rgb | float64 | D50/10 | 1 | 37.8723 | 8.0367 | 4.71x |
| color.lab2lch | float64 | four-channels | 1 | 10.8578 | 4.8300 | 2.25x |
| color.lch2lab | float64 | four-channels | 1 | 9.2084 | 3.2027 | 2.88x |
| color.lab2xyz | float64 | negative-Z | 1 | 16.3668 | 1.1286 | 14.50x |
| color.lab2rgb | float64 | negative-Z | 1 | 32.0692 | 4.4741 | 7.17x |
| color.xyz2luv | float64 | D50/10 | 1 | 15.0035 | 3.6263 | 4.14x |
| color.luv2xyz | float64 | D50/10 | 1 | 16.8031 | 1.2680 | 13.25x |
| color.rgb2luv | float64 | dark-sRGB | 1 | 16.1412 | 2.1185 | 7.62x |
| color.luv2rgb | float64 | low-lightness | 1 | 28.8164 | 6.5314 | 4.41x |
| color.luv2rgb | float64 | Luv-midtones | 1 | 43.6539 | 7.9181 | 5.51x |
