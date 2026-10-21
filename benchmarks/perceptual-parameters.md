# Perceptual color parameter performance

Strict native dispatch versus original scikit-image. Values, dtype and warnings are checked before timing. Seven warmed allocation-inclusive samples; one-thread reference. UserWarning is ignored during timing, while native negative-Z counting remains included. Float32 inverse Luv checks each rounding stage and a propagated direct RGB bound; the JSON records its contract and measured errors. Speedup = reference/current; slower cases remain visible.

| API | Input | Workload | Workers | Original ms | C++ ms | Speedup |
|---|---|---|---:|---:|---:|---:|
| color.rgb2hsv | float32 | grayscale | 1 | 31.4880 | 0.3717 | 84.71x |
| color.hsv2rgb | float32 | wrapped-hue | 1 | 24.6120 | 0.5577 | 44.13x |
| color.xyz2lab | float32 | D50/10 | 1 | 12.9978 | 5.0533 | 2.57x |
| color.lab2xyz | float32 | D50/10 | 1 | 13.2163 | 0.5314 | 24.87x |
| color.rgb2lab | float32 | D50/10 | 1 | 24.7313 | 10.5960 | 2.33x |
| color.lab2rgb | float32 | D50/10 | 1 | 24.5353 | 6.2302 | 3.94x |
| color.lab2lch | float32 | four-channels | 1 | 6.4525 | 2.8207 | 2.29x |
| color.lch2lab | float32 | four-channels | 1 | 2.1120 | 2.0734 | 1.02x |
| color.lab2xyz | float32 | negative-Z | 1 | 8.5102 | 1.0000 | 8.51x |
| color.lab2rgb | float32 | negative-Z | 1 | 18.4277 | 3.1751 | 5.80x |
| color.xyz2luv | float32 | D50/10 | 1 | 5.9473 | 2.5597 | 2.32x |
| color.luv2xyz | float32 | D50/10 | 1 | 5.6289 | 2.1968 | 2.56x |
| color.rgb2luv | float32 | dark-sRGB | 1 | 6.4084 | 0.3026 | 21.17x |
| color.luv2rgb | float32 | low-lightness | 1 | 15.2448 | 5.5429 | 2.75x |
| color.luv2rgb | float32 | Luv-midtones | 1 | 19.4700 | 8.4946 | 2.29x |
| color.rgb2hsv | float64 | grayscale | 1 | 36.0461 | 2.0449 | 17.63x |
| color.hsv2rgb | float64 | wrapped-hue | 1 | 46.2252 | 2.2161 | 20.86x |
| color.xyz2lab | float64 | D50/10 | 1 | 21.1441 | 5.8293 | 3.63x |
| color.lab2xyz | float64 | D50/10 | 1 | 25.6939 | 2.7552 | 9.33x |
| color.rgb2lab | float64 | D50/10 | 1 | 39.7631 | 12.3471 | 3.22x |
| color.lab2rgb | float64 | D50/10 | 1 | 69.7555 | 8.2933 | 8.41x |
| color.lab2lch | float64 | four-channels | 1 | 12.1972 | 4.6149 | 2.64x |
| color.lch2lab | float64 | four-channels | 1 | 10.0908 | 3.4410 | 2.93x |
| color.lab2xyz | float64 | negative-Z | 1 | 15.4106 | 1.1901 | 12.95x |
| color.lab2rgb | float64 | negative-Z | 1 | 29.0229 | 3.2140 | 9.03x |
| color.xyz2luv | float64 | D50/10 | 1 | 13.6361 | 2.6772 | 5.09x |
| color.luv2xyz | float64 | D50/10 | 1 | 15.0607 | 1.3440 | 11.21x |
| color.rgb2luv | float64 | dark-sRGB | 1 | 17.3891 | 2.1782 | 7.98x |
| color.luv2rgb | float64 | low-lightness | 1 | 30.8346 | 6.2762 | 4.91x |
| color.luv2rgb | float64 | Luv-midtones | 1 | 40.9741 | 6.4798 | 6.32x |
