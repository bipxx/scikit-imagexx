# Perceptual color parameter performance

Strict native dispatch versus original scikit-image. Values, dtype and warnings are checked before timing. Seven warmed allocation-inclusive samples; one-thread reference. UserWarning is ignored during timing, while native negative-Z counting remains included. Float32 inverse Luv checks each rounding stage and a propagated direct RGB bound; the JSON records its contract and measured errors. Speedup = reference/current; slower cases remain visible.

| API | Input | Workload | Workers | Original ms | C++ ms | Speedup |
|---|---|---|---:|---:|---:|---:|
| color.rgb2hsv | float32 | grayscale | 1 | 30.7992 | 0.3709 | 83.05x |
| color.rgb2hsv | float32 | grayscale | 4 | 30.6706 | 0.1480 | 207.24x |
| color.hsv2rgb | float32 | wrapped-hue | 1 | 23.7761 | 0.5333 | 44.58x |
| color.hsv2rgb | float32 | wrapped-hue | 4 | 25.0588 | 0.1994 | 125.70x |
| color.xyz2lab | float32 | D50/10 | 1 | 17.3173 | 5.7433 | 3.02x |
| color.xyz2lab | float32 | D50/10 | 4 | 14.6131 | 2.5371 | 5.76x |
| color.lab2xyz | float32 | D50/10 | 1 | 12.6601 | 1.0324 | 12.26x |
| color.lab2xyz | float32 | D50/10 | 4 | 13.0397 | 0.5391 | 24.19x |
| color.rgb2lab | float32 | D50/10 | 1 | 24.9122 | 11.2028 | 2.22x |
| color.rgb2lab | float32 | D50/10 | 4 | 25.9181 | 2.8436 | 9.11x |
| color.lab2rgb | float32 | D50/10 | 1 | 22.4426 | 6.8077 | 3.30x |
| color.lab2rgb | float32 | D50/10 | 4 | 21.7625 | 1.5262 | 14.26x |
| color.lab2lch | float32 | four-channels | 1 | 6.0925 | 3.0436 | 2.00x |
| color.lab2lch | float32 | four-channels | 4 | 6.3205 | 0.9212 | 6.86x |
| color.lch2lab | float32 | four-channels | 1 | 2.0654 | 1.9455 | 1.06x |
| color.lch2lab | float32 | four-channels | 4 | 1.9500 | 0.8194 | 2.38x |
| color.lab2xyz | float32 | negative-Z | 1 | 11.5570 | 0.5583 | 20.70x |
| color.lab2xyz | float32 | negative-Z | 4 | 9.8589 | 0.1867 | 52.80x |
| color.lab2rgb | float32 | negative-Z | 1 | 17.1507 | 3.1800 | 5.39x |
| color.lab2rgb | float32 | negative-Z | 4 | 16.8456 | 1.2995 | 12.96x |
| color.xyz2luv | float32 | D50/10 | 1 | 5.4992 | 2.1254 | 2.59x |
| color.xyz2luv | float32 | D50/10 | 4 | 6.0001 | 0.8328 | 7.20x |
| color.luv2xyz | float32 | D50/10 | 1 | 5.8004 | 2.1291 | 2.72x |
| color.luv2xyz | float32 | D50/10 | 4 | 6.3749 | 1.1302 | 5.64x |
| color.rgb2luv | float32 | dark-sRGB | 1 | 9.7055 | 0.5920 | 16.39x |
| color.rgb2luv | float32 | dark-sRGB | 4 | 6.8732 | 0.3492 | 19.69x |
| color.luv2rgb | float32 | low-lightness | 1 | 14.8376 | 5.4237 | 2.74x |
| color.luv2rgb | float32 | low-lightness | 4 | 16.7456 | 1.8343 | 9.13x |
| color.luv2rgb | float32 | Luv-midtones | 1 | 21.6754 | 9.8139 | 2.21x |
| color.luv2rgb | float32 | Luv-midtones | 4 | 20.9503 | 2.7365 | 7.66x |
| color.rgb2hsv | float64 | grayscale | 1 | 33.6471 | 1.9288 | 17.44x |
| color.rgb2hsv | float64 | grayscale | 4 | 35.5549 | 1.3603 | 26.14x |
| color.hsv2rgb | float64 | wrapped-hue | 1 | 38.9272 | 2.2107 | 17.61x |
| color.hsv2rgb | float64 | wrapped-hue | 4 | 38.8704 | 1.1709 | 33.20x |
| color.xyz2lab | float64 | D50/10 | 1 | 20.6836 | 6.7773 | 3.05x |
| color.xyz2lab | float64 | D50/10 | 4 | 19.2812 | 2.8533 | 6.76x |
| color.lab2xyz | float64 | D50/10 | 1 | 18.4659 | 1.2037 | 15.34x |
| color.lab2xyz | float64 | D50/10 | 4 | 18.5769 | 1.3102 | 14.18x |
| color.rgb2lab | float64 | D50/10 | 1 | 34.8188 | 12.2190 | 2.85x |
| color.rgb2lab | float64 | D50/10 | 4 | 35.5621 | 3.7125 | 9.58x |
| color.lab2rgb | float64 | D50/10 | 1 | 32.6154 | 7.3840 | 4.42x |
| color.lab2rgb | float64 | D50/10 | 4 | 33.1045 | 2.6197 | 12.64x |
| color.lab2lch | float64 | four-channels | 1 | 12.8468 | 5.0640 | 2.54x |
| color.lab2lch | float64 | four-channels | 4 | 10.3825 | 2.4483 | 4.24x |
| color.lch2lab | float64 | four-channels | 1 | 9.0868 | 3.4835 | 2.61x |
| color.lch2lab | float64 | four-channels | 4 | 8.6035 | 1.8199 | 4.73x |
| color.lab2xyz | float64 | negative-Z | 1 | 14.5358 | 1.2006 | 12.11x |
| color.lab2xyz | float64 | negative-Z | 4 | 14.0176 | 0.6803 | 20.60x |
| color.lab2rgb | float64 | negative-Z | 1 | 24.1263 | 3.0665 | 7.87x |
| color.lab2rgb | float64 | negative-Z | 4 | 24.7989 | 1.4466 | 17.14x |
| color.xyz2luv | float64 | D50/10 | 1 | 12.9042 | 3.6388 | 3.55x |
| color.xyz2luv | float64 | D50/10 | 4 | 9.9498 | 1.8565 | 5.36x |
| color.luv2xyz | float64 | D50/10 | 1 | 11.4760 | 1.7714 | 6.48x |
| color.luv2xyz | float64 | D50/10 | 4 | 14.0178 | 0.3984 | 35.19x |
| color.rgb2luv | float64 | dark-sRGB | 1 | 11.6625 | 2.5484 | 4.58x |
| color.rgb2luv | float64 | dark-sRGB | 4 | 11.9340 | 1.1148 | 10.70x |
| color.luv2rgb | float64 | low-lightness | 1 | 25.3760 | 7.6084 | 3.34x |
| color.luv2rgb | float64 | low-lightness | 4 | 24.8163 | 2.3216 | 10.69x |
| color.luv2rgb | float64 | Luv-midtones | 1 | 29.9810 | 7.3335 | 4.09x |
| color.luv2rgb | float64 | Luv-midtones | 4 | 31.3594 | 1.9391 | 16.17x |
