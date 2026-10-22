# Perceptual color parameter performance

Strict native dispatch versus original scikit-image. Values, dtype and warnings are checked before timing. Seven warmed allocation-inclusive samples; one-thread reference. UserWarning is ignored during timing, while native negative-Z counting remains included. Float32 inverse Luv checks each rounding stage and a propagated direct RGB bound; the JSON records its contract and measured errors. Speedup = reference/current; slower cases remain visible.

| API | Input | Workload | Workers | Original ms | C++ ms | Speedup |
|---|---|---|---:|---:|---:|---:|
| color.rgb2hsv | float32 | grayscale | 1 | 34.7492 | 0.3902 | 89.07x |
| color.hsv2rgb | float32 | wrapped-hue | 1 | 28.0468 | 0.6472 | 43.34x |
| color.xyz2lab | float32 | D50/10 | 1 | 13.8473 | 5.1380 | 2.70x |
| color.lab2xyz | float32 | D50/10 | 1 | 15.3133 | 0.5589 | 27.40x |
| color.rgb2lab | float32 | D50/10 | 1 | 24.2966 | 11.2605 | 2.16x |
| color.lab2rgb | float32 | D50/10 | 1 | 27.8042 | 6.7732 | 4.11x |
| color.lab2lch | float32 | four-channels | 1 | 7.9502 | 3.6217 | 2.20x |
| color.lch2lab | float32 | four-channels | 1 | 4.3606 | 2.8899 | 1.51x |
| color.lab2xyz | float32 | negative-Z | 1 | 10.3206 | 1.1346 | 9.10x |
| color.lab2rgb | float32 | negative-Z | 1 | 20.1700 | 3.8648 | 5.22x |
| color.xyz2luv | float32 | D50/10 | 1 | 11.0383 | 3.9579 | 2.79x |
| color.luv2xyz | float32 | D50/10 | 1 | 9.2091 | 3.7848 | 2.43x |
| color.rgb2luv | float32 | dark-sRGB | 1 | 12.9848 | 0.8433 | 15.40x |
| color.luv2rgb | float32 | low-lightness | 1 | 24.2001 | 8.4906 | 2.85x |
| color.luv2rgb | float32 | Luv-midtones | 1 | 31.2163 | 11.3739 | 2.74x |
| color.rgb2hsv | float64 | grayscale | 1 | 61.5246 | 1.9002 | 32.38x |
| color.hsv2rgb | float64 | wrapped-hue | 1 | 99.4232 | 4.5341 | 21.93x |
| color.xyz2lab | float64 | D50/10 | 1 | 37.1569 | 8.9519 | 4.15x |
| color.lab2xyz | float64 | D50/10 | 1 | 37.3029 | 4.1786 | 8.93x |
| color.rgb2lab | float64 | D50/10 | 1 | 68.1702 | 15.4888 | 4.40x |
| color.lab2rgb | float64 | D50/10 | 1 | 64.6084 | 8.6627 | 7.46x |
| color.lab2lch | float64 | four-channels | 1 | 18.3112 | 5.9176 | 3.09x |
| color.lch2lab | float64 | four-channels | 1 | 15.8854 | 5.0995 | 3.12x |
| color.lab2xyz | float64 | negative-Z | 1 | 37.0814 | 1.8426 | 20.12x |
| color.lab2rgb | float64 | negative-Z | 1 | 43.6125 | 4.3232 | 10.09x |
| color.xyz2luv | float64 | D50/10 | 1 | 28.0986 | 3.7144 | 7.56x |
| color.luv2xyz | float64 | D50/10 | 1 | 32.0157 | 2.2158 | 14.45x |
| color.rgb2luv | float64 | dark-sRGB | 1 | 41.9564 | 1.9594 | 21.41x |
| color.luv2rgb | float64 | low-lightness | 1 | 64.9604 | 8.6340 | 7.52x |
| color.luv2rgb | float64 | Luv-midtones | 1 | 92.8959 | 11.1278 | 8.35x |
