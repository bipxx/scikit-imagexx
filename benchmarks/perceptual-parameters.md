# Perceptual color parameter performance

Strict native dispatch versus original scikit-image. Values, dtype and warnings are checked before timing. Seven warmed allocation-inclusive samples; one-thread reference. UserWarning is ignored during timing, while native negative-Z counting remains included. Float32 inverse Luv checks each rounding stage and a propagated direct RGB bound; the JSON records its contract and measured errors. Speedup = reference/current; slower cases remain visible.

| API | Input | Workload | Workers | Original ms | C++ ms | Speedup |
|---|---|---|---:|---:|---:|---:|
| color.rgb2hsv | float32 | grayscale | 1 | 33.1598 | 0.3965 | 83.63x |
| color.hsv2rgb | float32 | wrapped-hue | 1 | 25.3514 | 0.5086 | 49.85x |
| color.xyz2lab | float32 | D50/10 | 1 | 12.3617 | 4.8177 | 2.57x |
| color.lab2xyz | float32 | D50/10 | 1 | 13.1181 | 0.5124 | 25.60x |
| color.rgb2lab | float32 | D50/10 | 1 | 23.2889 | 10.8253 | 2.15x |
| color.lab2rgb | float32 | D50/10 | 1 | 22.8016 | 6.4268 | 3.55x |
| color.lab2lch | float32 | four-channels | 1 | 6.3667 | 2.8970 | 2.20x |
| color.lch2lab | float32 | four-channels | 1 | 2.5733 | 2.2345 | 1.15x |
| color.lab2xyz | float32 | negative-Z | 1 | 8.2995 | 1.0201 | 8.14x |
| color.lab2rgb | float32 | negative-Z | 1 | 16.7295 | 3.1002 | 5.40x |
| color.xyz2luv | float32 | D50/10 | 1 | 6.1257 | 2.5340 | 2.42x |
| color.luv2xyz | float32 | D50/10 | 1 | 7.0368 | 2.7559 | 2.55x |
| color.rgb2luv | float32 | dark-sRGB | 1 | 7.0452 | 0.3085 | 22.84x |
| color.luv2rgb | float32 | low-lightness | 1 | 16.7587 | 5.6372 | 2.97x |
| color.luv2rgb | float32 | Luv-midtones | 1 | 22.6859 | 8.4615 | 2.68x |
| color.rgb2hsv | float64 | grayscale | 1 | 45.8297 | 3.1904 | 14.37x |
| color.hsv2rgb | float64 | wrapped-hue | 1 | 47.8891 | 2.7316 | 17.53x |
| color.xyz2lab | float64 | D50/10 | 1 | 19.5607 | 5.8404 | 3.35x |
| color.lab2xyz | float64 | D50/10 | 1 | 22.6772 | 2.7457 | 8.26x |
| color.rgb2lab | float64 | D50/10 | 1 | 36.2133 | 12.4494 | 2.91x |
| color.lab2rgb | float64 | D50/10 | 1 | 39.8624 | 7.6964 | 5.18x |
| color.lab2lch | float64 | four-channels | 1 | 11.3302 | 4.5982 | 2.46x |
| color.lch2lab | float64 | four-channels | 1 | 11.3467 | 3.4802 | 3.26x |
| color.lab2xyz | float64 | negative-Z | 1 | 16.0991 | 1.1834 | 13.60x |
| color.lab2rgb | float64 | negative-Z | 1 | 32.5077 | 3.5128 | 9.25x |
| color.xyz2luv | float64 | D50/10 | 1 | 15.1810 | 3.0107 | 5.04x |
| color.luv2xyz | float64 | D50/10 | 1 | 15.4961 | 1.0886 | 14.24x |
| color.rgb2luv | float64 | dark-sRGB | 1 | 14.1873 | 0.9525 | 14.90x |
| color.luv2rgb | float64 | low-lightness | 1 | 29.5241 | 6.2610 | 4.72x |
| color.luv2rgb | float64 | Luv-midtones | 1 | 29.3945 | 6.5062 | 4.52x |
