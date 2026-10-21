# Perceptual color parameter performance

Strict native dispatch versus original scikit-image. Values, dtype and warnings are checked before timing. Seven warmed allocation-inclusive samples; one-thread reference. UserWarning is ignored during timing, while native negative-Z counting remains included. Float32 inverse Luv checks each rounding stage and a propagated direct RGB bound; the JSON records its contract and measured errors. Speedup = reference/current; slower cases remain visible.

| API | Input | Workload | Workers | Original ms | C++ ms | Speedup |
|---|---|---|---:|---:|---:|---:|
| color.rgb2hsv | float32 | grayscale | 1 | 30.5242 | 0.4152 | 73.52x |
| color.rgb2hsv | float32 | grayscale | 4 | 28.9862 | 0.1465 | 197.92x |
| color.hsv2rgb | float32 | wrapped-hue | 1 | 27.2886 | 0.5723 | 47.68x |
| color.hsv2rgb | float32 | wrapped-hue | 4 | 24.8051 | 0.6375 | 38.91x |
| color.xyz2lab | float32 | D50/10 | 1 | 14.5077 | 5.8769 | 2.47x |
| color.xyz2lab | float32 | D50/10 | 4 | 14.3576 | 2.2379 | 6.42x |
| color.lab2xyz | float32 | D50/10 | 1 | 12.4566 | 0.5204 | 23.94x |
| color.lab2xyz | float32 | D50/10 | 4 | 13.0367 | 0.5961 | 21.87x |
| color.rgb2lab | float32 | D50/10 | 1 | 23.9622 | 10.8133 | 2.22x |
| color.rgb2lab | float32 | D50/10 | 4 | 22.6372 | 3.1304 | 7.23x |
| color.lab2rgb | float32 | D50/10 | 1 | 21.6245 | 6.1159 | 3.54x |
| color.lab2rgb | float32 | D50/10 | 4 | 22.9663 | 2.1199 | 10.83x |
| color.lab2lch | float32 | four-channels | 1 | 5.9839 | 2.7562 | 2.17x |
| color.lab2lch | float32 | four-channels | 4 | 6.0362 | 0.8502 | 7.10x |
| color.lch2lab | float32 | four-channels | 1 | 1.5961 | 1.5369 | 1.04x |
| color.lch2lab | float32 | four-channels | 4 | 1.5044 | 0.4611 | 3.26x |
| color.lab2xyz | float32 | negative-Z | 1 | 8.7175 | 0.4836 | 18.03x |
| color.lab2xyz | float32 | negative-Z | 4 | 8.6160 | 0.1618 | 53.26x |
| color.lab2rgb | float32 | negative-Z | 1 | 17.0076 | 2.8753 | 5.92x |
| color.lab2rgb | float32 | negative-Z | 4 | 27.0150 | 2.1347 | 12.65x |
| color.xyz2luv | float32 | D50/10 | 1 | 9.0761 | 2.7164 | 3.34x |
| color.xyz2luv | float32 | D50/10 | 4 | 13.1822 | 1.1428 | 11.54x |
| color.luv2xyz | float32 | D50/10 | 1 | 11.5379 | 3.4090 | 3.38x |
| color.luv2xyz | float32 | D50/10 | 4 | 11.4543 | 2.1462 | 5.34x |
| color.rgb2luv | float32 | dark-sRGB | 1 | 9.2910 | 0.4235 | 21.94x |
| color.rgb2luv | float32 | dark-sRGB | 4 | 6.3146 | 0.1975 | 31.97x |
| color.luv2rgb | float32 | low-lightness | 1 | 14.5481 | 5.5114 | 2.64x |
| color.luv2rgb | float32 | low-lightness | 4 | 17.4232 | 2.0203 | 8.62x |
| color.luv2rgb | float32 | Luv-midtones | 1 | 21.5932 | 9.4875 | 2.28x |
| color.luv2rgb | float32 | Luv-midtones | 4 | 19.3899 | 2.1647 | 8.96x |
| color.rgb2hsv | float64 | grayscale | 1 | 34.1238 | 1.9757 | 17.27x |
| color.rgb2hsv | float64 | grayscale | 4 | 34.5355 | 1.1769 | 29.35x |
| color.hsv2rgb | float64 | wrapped-hue | 1 | 40.2328 | 2.3555 | 17.08x |
| color.hsv2rgb | float64 | wrapped-hue | 4 | 42.2346 | 1.1674 | 36.18x |
| color.xyz2lab | float64 | D50/10 | 1 | 24.5623 | 8.5838 | 2.86x |
| color.xyz2lab | float64 | D50/10 | 4 | 19.1050 | 2.2011 | 8.68x |
| color.lab2xyz | float64 | D50/10 | 1 | 18.3869 | 2.0929 | 8.79x |
| color.lab2xyz | float64 | D50/10 | 4 | 20.6935 | 1.3566 | 15.25x |
| color.rgb2lab | float64 | D50/10 | 1 | 38.1920 | 12.1161 | 3.15x |
| color.rgb2lab | float64 | D50/10 | 4 | 41.6263 | 4.6715 | 8.91x |
| color.lab2rgb | float64 | D50/10 | 1 | 35.0342 | 7.6387 | 4.59x |
| color.lab2rgb | float64 | D50/10 | 4 | 32.4369 | 2.3987 | 13.52x |
| color.lab2lch | float64 | four-channels | 1 | 9.5996 | 5.3431 | 1.80x |
| color.lab2lch | float64 | four-channels | 4 | 8.7577 | 2.1478 | 4.08x |
| color.lch2lab | float64 | four-channels | 1 | 8.8152 | 3.2900 | 2.68x |
| color.lch2lab | float64 | four-channels | 4 | 8.6037 | 2.0413 | 4.21x |
| color.lab2xyz | float64 | negative-Z | 1 | 14.3179 | 1.1332 | 12.63x |
| color.lab2xyz | float64 | negative-Z | 4 | 13.9914 | 0.3632 | 38.52x |
| color.lab2rgb | float64 | negative-Z | 1 | 22.8619 | 3.0309 | 7.54x |
| color.lab2rgb | float64 | negative-Z | 4 | 25.8105 | 1.1918 | 21.66x |
| color.xyz2luv | float64 | D50/10 | 1 | 11.4261 | 2.6918 | 4.24x |
| color.xyz2luv | float64 | D50/10 | 4 | 12.0105 | 1.7250 | 6.96x |
| color.luv2xyz | float64 | D50/10 | 1 | 13.1798 | 0.9204 | 14.32x |
| color.luv2xyz | float64 | D50/10 | 4 | 13.7414 | 0.3197 | 42.98x |
| color.rgb2luv | float64 | dark-sRGB | 1 | 11.5321 | 1.9943 | 5.78x |
| color.rgb2luv | float64 | dark-sRGB | 4 | 19.2059 | 1.1682 | 16.44x |
| color.luv2rgb | float64 | low-lightness | 1 | 24.4028 | 6.0786 | 4.01x |
| color.luv2rgb | float64 | low-lightness | 4 | 24.5015 | 1.7505 | 14.00x |
| color.luv2rgb | float64 | Luv-midtones | 1 | 28.7236 | 6.4717 | 4.44x |
| color.luv2rgb | float64 | Luv-midtones | 4 | 28.1495 | 1.7795 | 15.82x |
