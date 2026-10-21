# Perceptual color parameter performance

Strict native dispatch versus original scikit-image. Values, dtype and warnings are checked before timing. Seven warmed allocation-inclusive samples; one-thread reference. UserWarning is ignored during timing, while native negative-Z counting remains included. Float32 inverse Luv checks each rounding stage and a propagated direct RGB bound; the JSON records its contract and measured errors. Speedup = reference/current; slower cases remain visible.

| API | Input | Workload | Workers | Original ms | C++ ms | Speedup |
|---|---|---|---:|---:|---:|---:|
| color.rgb2hsv | float32 | grayscale | 1 | 29.2113 | 0.4109 | 71.08x |
| color.hsv2rgb | float32 | wrapped-hue | 1 | 23.3611 | 0.4799 | 48.68x |
| color.xyz2lab | float32 | D50/10 | 1 | 14.3218 | 5.5568 | 2.58x |
| color.lab2xyz | float32 | D50/10 | 1 | 13.3098 | 1.0502 | 12.67x |
| color.rgb2lab | float32 | D50/10 | 1 | 21.8157 | 12.2698 | 1.78x |
| color.lab2rgb | float32 | D50/10 | 1 | 21.3951 | 6.1822 | 3.46x |
| color.lab2lch | float32 | four-channels | 1 | 5.8859 | 2.7368 | 2.15x |
| color.lch2lab | float32 | four-channels | 1 | 2.1146 | 2.1436 | 0.99x |
| color.lab2xyz | float32 | negative-Z | 1 | 8.0759 | 1.0659 | 7.58x |
| color.lab2rgb | float32 | negative-Z | 1 | 16.0455 | 3.1110 | 5.16x |
| color.xyz2luv | float32 | D50/10 | 1 | 6.1240 | 2.5269 | 2.42x |
| color.luv2xyz | float32 | D50/10 | 1 | 6.0339 | 2.6943 | 2.24x |
| color.rgb2luv | float32 | dark-sRGB | 1 | 7.9184 | 0.3569 | 22.19x |
| color.luv2rgb | float32 | low-lightness | 1 | 17.4101 | 5.4689 | 3.18x |
| color.luv2rgb | float32 | Luv-midtones | 1 | 19.5735 | 8.1503 | 2.40x |
| color.rgb2hsv | float64 | grayscale | 1 | 40.6408 | 2.0424 | 19.90x |
| color.hsv2rgb | float64 | wrapped-hue | 1 | 40.2382 | 2.2675 | 17.75x |
| color.xyz2lab | float64 | D50/10 | 1 | 20.1732 | 7.3170 | 2.76x |
| color.lab2xyz | float64 | D50/10 | 1 | 17.7009 | 2.0413 | 8.67x |
| color.rgb2lab | float64 | D50/10 | 1 | 33.3383 | 12.3781 | 2.69x |
| color.lab2rgb | float64 | D50/10 | 1 | 36.4590 | 8.9700 | 4.06x |
| color.lab2lch | float64 | four-channels | 1 | 9.9667 | 4.4841 | 2.22x |
| color.lch2lab | float64 | four-channels | 1 | 8.0148 | 3.3785 | 2.37x |
| color.lab2xyz | float64 | negative-Z | 1 | 12.5998 | 1.0827 | 11.64x |
| color.lab2rgb | float64 | negative-Z | 1 | 23.3876 | 3.4028 | 6.87x |
| color.xyz2luv | float64 | D50/10 | 1 | 10.1686 | 2.6074 | 3.90x |
| color.luv2xyz | float64 | D50/10 | 1 | 12.8881 | 1.2248 | 10.52x |
| color.rgb2luv | float64 | dark-sRGB | 1 | 13.0479 | 0.7966 | 16.38x |
| color.luv2rgb | float64 | low-lightness | 1 | 28.7761 | 6.5476 | 4.39x |
| color.luv2rgb | float64 | Luv-midtones | 1 | 30.9905 | 6.5812 | 4.71x |
