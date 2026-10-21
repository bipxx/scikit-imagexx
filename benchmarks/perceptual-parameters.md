# Perceptual color parameter performance

Strict native dispatch versus original scikit-image. Values, dtype and warnings are checked before timing. Seven warmed allocation-inclusive samples; one-thread reference. UserWarning is ignored during timing, while native negative-Z counting remains included. Float32 inverse Luv checks each rounding stage and a propagated direct RGB bound; the JSON records its contract and measured errors. Speedup = reference/current; slower cases remain visible.

| API | Input | Workload | Workers | Original ms | C++ ms | Speedup |
|---|---|---|---:|---:|---:|---:|
| color.rgb2hsv | float32 | grayscale | 1 | 32.6869 | 0.3933 | 83.10x |
| color.hsv2rgb | float32 | wrapped-hue | 1 | 27.0031 | 0.5551 | 48.65x |
| color.xyz2lab | float32 | D50/10 | 1 | 13.0307 | 4.9324 | 2.64x |
| color.lab2xyz | float32 | D50/10 | 1 | 13.0937 | 0.5075 | 25.80x |
| color.rgb2lab | float32 | D50/10 | 1 | 22.2188 | 10.3727 | 2.14x |
| color.lab2rgb | float32 | D50/10 | 1 | 21.9399 | 6.1067 | 3.59x |
| color.lab2lch | float32 | four-channels | 1 | 5.8799 | 2.7454 | 2.14x |
| color.lch2lab | float32 | four-channels | 1 | 2.3249 | 2.0558 | 1.13x |
| color.lab2xyz | float32 | negative-Z | 1 | 7.8703 | 1.0031 | 7.85x |
| color.lab2rgb | float32 | negative-Z | 1 | 16.8171 | 3.0621 | 5.49x |
| color.xyz2luv | float32 | D50/10 | 1 | 6.2011 | 2.4864 | 2.49x |
| color.luv2xyz | float32 | D50/10 | 1 | 6.9456 | 2.2703 | 3.06x |
| color.rgb2luv | float32 | dark-sRGB | 1 | 8.6727 | 0.3228 | 26.87x |
| color.luv2rgb | float32 | low-lightness | 1 | 16.0147 | 5.4659 | 2.93x |
| color.luv2rgb | float32 | Luv-midtones | 1 | 20.4000 | 8.3139 | 2.45x |
| color.rgb2hsv | float64 | grayscale | 1 | 35.2028 | 2.0304 | 17.34x |
| color.hsv2rgb | float64 | wrapped-hue | 1 | 43.1433 | 2.2216 | 19.42x |
| color.xyz2lab | float64 | D50/10 | 1 | 17.0643 | 6.0879 | 2.80x |
| color.lab2xyz | float64 | D50/10 | 1 | 18.4651 | 2.1206 | 8.71x |
| color.rgb2lab | float64 | D50/10 | 1 | 32.3966 | 12.0146 | 2.70x |
| color.lab2rgb | float64 | D50/10 | 1 | 35.6412 | 7.1520 | 4.98x |
| color.lab2lch | float64 | four-channels | 1 | 10.1377 | 4.2562 | 2.38x |
| color.lch2lab | float64 | four-channels | 1 | 8.3129 | 3.0983 | 2.68x |
| color.lab2xyz | float64 | negative-Z | 1 | 13.9013 | 1.1090 | 12.53x |
| color.lab2rgb | float64 | negative-Z | 1 | 23.2410 | 3.1100 | 7.47x |
| color.xyz2luv | float64 | D50/10 | 1 | 10.8704 | 2.8621 | 3.80x |
| color.luv2xyz | float64 | D50/10 | 1 | 12.6261 | 0.8342 | 15.14x |
| color.rgb2luv | float64 | dark-sRGB | 1 | 12.2522 | 0.8591 | 14.26x |
| color.luv2rgb | float64 | low-lightness | 1 | 26.4830 | 6.3219 | 4.19x |
| color.luv2rgb | float64 | Luv-midtones | 1 | 33.4146 | 6.5072 | 5.14x |
