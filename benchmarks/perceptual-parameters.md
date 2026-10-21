# Perceptual color parameter performance

Strict native dispatch versus original scikit-image. Values, dtype and warnings are checked before timing. Seven warmed allocation-inclusive samples; one-thread reference. UserWarning is ignored during timing, while native negative-Z counting remains included. Float32 inverse Luv checks each rounding stage and a propagated direct RGB bound; the JSON records its contract and measured errors. Speedup = reference/current; slower cases remain visible.

| API | Input | Workload | Workers | Original ms | C++ ms | Speedup |
|---|---|---|---:|---:|---:|---:|
| color.rgb2hsv | float32 | grayscale | 1 | 29.6696 | 0.3947 | 75.17x |
| color.rgb2hsv | float32 | grayscale | 4 | 28.5978 | 0.1831 | 156.17x |
| color.hsv2rgb | float32 | wrapped-hue | 1 | 22.9079 | 0.5448 | 42.05x |
| color.hsv2rgb | float32 | wrapped-hue | 4 | 22.7665 | 0.5452 | 41.76x |
| color.xyz2lab | float32 | D50/10 | 1 | 14.6804 | 5.2434 | 2.80x |
| color.xyz2lab | float32 | D50/10 | 4 | 14.3824 | 2.1943 | 6.55x |
| color.lab2xyz | float32 | D50/10 | 1 | 15.5454 | 1.2379 | 12.56x |
| color.lab2xyz | float32 | D50/10 | 4 | 25.6069 | 0.5123 | 49.98x |
| color.rgb2lab | float32 | D50/10 | 1 | 25.9554 | 10.4882 | 2.47x |
| color.rgb2lab | float32 | D50/10 | 4 | 25.6752 | 4.9140 | 5.22x |
| color.lab2rgb | float32 | D50/10 | 1 | 25.6104 | 6.6003 | 3.88x |
| color.lab2rgb | float32 | D50/10 | 4 | 23.7552 | 2.8281 | 8.40x |
| color.lab2lch | float32 | four-channels | 1 | 6.6485 | 2.7819 | 2.39x |
| color.lab2lch | float32 | four-channels | 4 | 6.7073 | 1.2882 | 5.21x |
| color.lch2lab | float32 | four-channels | 1 | 1.9013 | 1.6197 | 1.17x |
| color.lch2lab | float32 | four-channels | 4 | 1.8940 | 0.9517 | 1.99x |
| color.lab2xyz | float32 | negative-Z | 1 | 10.1695 | 0.4846 | 20.98x |
| color.lab2xyz | float32 | negative-Z | 4 | 9.7555 | 0.2662 | 36.65x |
| color.lab2rgb | float32 | negative-Z | 1 | 17.9002 | 3.4654 | 5.17x |
| color.lab2rgb | float32 | negative-Z | 4 | 17.6660 | 1.4756 | 11.97x |
| color.xyz2luv | float32 | D50/10 | 1 | 8.0672 | 3.9346 | 2.05x |
| color.xyz2luv | float32 | D50/10 | 4 | 9.8248 | 1.3670 | 7.19x |
| color.luv2xyz | float32 | D50/10 | 1 | 8.2144 | 3.4266 | 2.40x |
| color.luv2xyz | float32 | D50/10 | 4 | 11.5628 | 1.2559 | 9.21x |
| color.rgb2luv | float32 | dark-sRGB | 1 | 14.8113 | 0.4751 | 31.17x |
| color.rgb2luv | float32 | dark-sRGB | 4 | 11.0957 | 0.3627 | 30.59x |
| color.luv2rgb | float32 | low-lightness | 1 | 26.6906 | 8.5501 | 3.12x |
| color.luv2rgb | float32 | low-lightness | 4 | 32.5688 | 4.1146 | 7.92x |
| color.luv2rgb | float32 | Luv-midtones | 1 | 52.0309 | 12.1291 | 4.29x |
| color.luv2rgb | float32 | Luv-midtones | 4 | 42.4364 | 5.1352 | 8.26x |
| color.rgb2hsv | float64 | grayscale | 1 | 41.7496 | 2.3001 | 18.15x |
| color.rgb2hsv | float64 | grayscale | 4 | 42.8187 | 1.5570 | 27.50x |
| color.hsv2rgb | float64 | wrapped-hue | 1 | 76.8620 | 3.7117 | 20.71x |
| color.hsv2rgb | float64 | wrapped-hue | 4 | 86.1205 | 1.9749 | 43.61x |
| color.xyz2lab | float64 | D50/10 | 1 | 37.6636 | 8.4204 | 4.47x |
| color.xyz2lab | float64 | D50/10 | 4 | 29.6968 | 3.9400 | 7.54x |
| color.lab2xyz | float64 | D50/10 | 1 | 22.0423 | 1.2339 | 17.86x |
| color.lab2xyz | float64 | D50/10 | 4 | 27.6140 | 1.3620 | 20.27x |
| color.rgb2lab | float64 | D50/10 | 1 | 36.6465 | 11.9049 | 3.08x |
| color.rgb2lab | float64 | D50/10 | 4 | 46.3552 | 4.6414 | 9.99x |
| color.lab2rgb | float64 | D50/10 | 1 | 33.4885 | 7.4044 | 4.52x |
| color.lab2rgb | float64 | D50/10 | 4 | 32.7372 | 2.3121 | 14.16x |
| color.lab2lch | float64 | four-channels | 1 | 10.9883 | 4.1910 | 2.62x |
| color.lab2lch | float64 | four-channels | 4 | 9.9888 | 2.6256 | 3.80x |
| color.lch2lab | float64 | four-channels | 1 | 8.7173 | 3.2081 | 2.72x |
| color.lch2lab | float64 | four-channels | 4 | 9.6085 | 1.5902 | 6.04x |
| color.lab2xyz | float64 | negative-Z | 1 | 13.6994 | 1.0859 | 12.62x |
| color.lab2xyz | float64 | negative-Z | 4 | 13.0538 | 0.3299 | 39.57x |
| color.lab2rgb | float64 | negative-Z | 1 | 25.0082 | 3.0557 | 8.18x |
| color.lab2rgb | float64 | negative-Z | 4 | 24.6872 | 0.9844 | 25.08x |
| color.xyz2luv | float64 | D50/10 | 1 | 14.0414 | 3.6107 | 3.89x |
| color.xyz2luv | float64 | D50/10 | 4 | 13.4616 | 1.3919 | 9.67x |
| color.luv2xyz | float64 | D50/10 | 1 | 12.1267 | 0.8154 | 14.87x |
| color.luv2xyz | float64 | D50/10 | 4 | 11.2892 | 0.3164 | 35.68x |
| color.rgb2luv | float64 | dark-sRGB | 1 | 12.2001 | 1.7994 | 6.78x |
| color.rgb2luv | float64 | dark-sRGB | 4 | 13.0088 | 1.0322 | 12.60x |
| color.luv2rgb | float64 | low-lightness | 1 | 27.1888 | 6.2848 | 4.33x |
| color.luv2rgb | float64 | low-lightness | 4 | 31.6633 | 2.8425 | 11.14x |
| color.luv2rgb | float64 | Luv-midtones | 1 | 38.2289 | 7.9941 | 4.78x |
| color.luv2rgb | float64 | Luv-midtones | 4 | 30.4818 | 1.8217 | 16.73x |
