# Perceptual color parameter performance

Strict native dispatch versus original scikit-image. Values, dtype and warnings are checked before timing. Seven warmed allocation-inclusive samples; one-thread reference. UserWarning is ignored during timing, while native negative-Z counting remains included. Float32 inverse Luv checks each rounding stage and a propagated direct RGB bound; the JSON records its contract and measured errors. Speedup = reference/current; slower cases remain visible.

| API | Input | Workload | Workers | Original ms | C++ ms | Speedup |
|---|---|---|---:|---:|---:|---:|
| color.rgb2hsv | float32 | grayscale | 1 | 31.0193 | 0.4099 | 75.68x |
| color.rgb2hsv | float32 | grayscale | 4 | 30.7340 | 0.1557 | 197.38x |
| color.hsv2rgb | float32 | wrapped-hue | 1 | 23.7745 | 0.5079 | 46.81x |
| color.hsv2rgb | float32 | wrapped-hue | 4 | 23.9490 | 0.2483 | 96.47x |
| color.xyz2lab | float32 | D50/10 | 1 | 13.2675 | 5.1746 | 2.56x |
| color.xyz2lab | float32 | D50/10 | 4 | 13.9644 | 1.7688 | 7.90x |
| color.lab2xyz | float32 | D50/10 | 1 | 13.7407 | 1.0315 | 13.32x |
| color.lab2xyz | float32 | D50/10 | 4 | 12.7073 | 0.5285 | 24.05x |
| color.rgb2lab | float32 | D50/10 | 1 | 22.6537 | 11.1686 | 2.03x |
| color.rgb2lab | float32 | D50/10 | 4 | 25.3512 | 3.7547 | 6.75x |
| color.lab2rgb | float32 | D50/10 | 1 | 22.2365 | 5.9987 | 3.71x |
| color.lab2rgb | float32 | D50/10 | 4 | 22.2058 | 1.8655 | 11.90x |
| color.lab2lch | float32 | four-channels | 1 | 6.1357 | 2.9430 | 2.08x |
| color.lab2lch | float32 | four-channels | 4 | 6.9307 | 1.7734 | 3.91x |
| color.lch2lab | float32 | four-channels | 1 | 1.3931 | 1.4648 | 0.95x |
| color.lch2lab | float32 | four-channels | 4 | 1.5292 | 0.5192 | 2.95x |
| color.lab2xyz | float32 | negative-Z | 1 | 9.4731 | 0.5263 | 18.00x |
| color.lab2xyz | float32 | negative-Z | 4 | 9.0150 | 0.2559 | 35.23x |
| color.lab2rgb | float32 | negative-Z | 1 | 17.5713 | 3.4372 | 5.11x |
| color.lab2rgb | float32 | negative-Z | 4 | 18.2454 | 1.1674 | 15.63x |
| color.xyz2luv | float32 | D50/10 | 1 | 6.4202 | 2.1113 | 3.04x |
| color.xyz2luv | float32 | D50/10 | 4 | 7.4062 | 0.8789 | 8.43x |
| color.luv2xyz | float32 | D50/10 | 1 | 6.0018 | 2.2003 | 2.73x |
| color.luv2xyz | float32 | D50/10 | 4 | 6.9344 | 1.2208 | 5.68x |
| color.rgb2luv | float32 | dark-sRGB | 1 | 11.2154 | 0.9163 | 12.24x |
| color.rgb2luv | float32 | dark-sRGB | 4 | 6.8862 | 0.2954 | 23.31x |
| color.luv2rgb | float32 | low-lightness | 1 | 17.2656 | 6.2969 | 2.74x |
| color.luv2rgb | float32 | low-lightness | 4 | 16.9474 | 2.1346 | 7.94x |
| color.luv2rgb | float32 | Luv-midtones | 1 | 21.6107 | 8.3037 | 2.60x |
| color.luv2rgb | float32 | Luv-midtones | 4 | 19.6555 | 2.3256 | 8.45x |
| color.rgb2hsv | float64 | grayscale | 1 | 33.8883 | 1.9364 | 17.50x |
| color.rgb2hsv | float64 | grayscale | 4 | 38.9511 | 1.2343 | 31.56x |
| color.hsv2rgb | float64 | wrapped-hue | 1 | 44.1762 | 2.4264 | 18.21x |
| color.hsv2rgb | float64 | wrapped-hue | 4 | 41.4900 | 1.1857 | 34.99x |
| color.xyz2lab | float64 | D50/10 | 1 | 19.7786 | 6.1829 | 3.20x |
| color.xyz2lab | float64 | D50/10 | 4 | 19.3526 | 2.2578 | 8.57x |
| color.lab2xyz | float64 | D50/10 | 1 | 18.9009 | 2.3333 | 8.10x |
| color.lab2xyz | float64 | D50/10 | 4 | 19.6602 | 1.2939 | 15.19x |
| color.rgb2lab | float64 | D50/10 | 1 | 34.8738 | 12.0585 | 2.89x |
| color.rgb2lab | float64 | D50/10 | 4 | 33.6779 | 3.3837 | 9.95x |
| color.lab2rgb | float64 | D50/10 | 1 | 34.2143 | 8.9951 | 3.80x |
| color.lab2rgb | float64 | D50/10 | 4 | 32.1258 | 2.6051 | 12.33x |
| color.lab2lch | float64 | four-channels | 1 | 10.1768 | 4.5465 | 2.24x |
| color.lab2lch | float64 | four-channels | 4 | 8.7874 | 2.0970 | 4.19x |
| color.lch2lab | float64 | four-channels | 1 | 9.1655 | 3.2983 | 2.78x |
| color.lch2lab | float64 | four-channels | 4 | 7.9855 | 1.7052 | 4.68x |
| color.lab2xyz | float64 | negative-Z | 1 | 12.3969 | 1.0751 | 11.53x |
| color.lab2xyz | float64 | negative-Z | 4 | 14.7341 | 0.5221 | 28.22x |
| color.lab2rgb | float64 | negative-Z | 1 | 25.8486 | 3.0936 | 8.36x |
| color.lab2rgb | float64 | negative-Z | 4 | 28.7076 | 1.2459 | 23.04x |
| color.xyz2luv | float64 | D50/10 | 1 | 12.3308 | 2.8158 | 4.38x |
| color.xyz2luv | float64 | D50/10 | 4 | 11.8331 | 1.7327 | 6.83x |
| color.luv2xyz | float64 | D50/10 | 1 | 11.6923 | 1.8862 | 6.20x |
| color.luv2xyz | float64 | D50/10 | 4 | 14.1987 | 1.3033 | 10.89x |
| color.rgb2luv | float64 | dark-sRGB | 1 | 10.7426 | 1.8725 | 5.74x |
| color.rgb2luv | float64 | dark-sRGB | 4 | 14.9599 | 1.1880 | 12.59x |
| color.luv2rgb | float64 | low-lightness | 1 | 27.2564 | 6.8608 | 3.97x |
| color.luv2rgb | float64 | low-lightness | 4 | 25.1740 | 2.0505 | 12.28x |
| color.luv2rgb | float64 | Luv-midtones | 1 | 30.2510 | 7.2001 | 4.20x |
| color.luv2rgb | float64 | Luv-midtones | 4 | 29.6091 | 2.2805 | 12.98x |
