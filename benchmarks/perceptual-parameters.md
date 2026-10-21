# Perceptual color parameter performance

Strict native dispatch versus original scikit-image. Values, dtype and warnings are checked before timing. Seven warmed allocation-inclusive samples; one-thread reference. UserWarning is ignored during timing, while native negative-Z counting remains included. Speedup = reference/current; slower cases remain visible.

| API | Input | Workload | Workers | Original ms | C++ ms | Speedup |
|---|---|---|---:|---:|---:|---:|
| color.rgb2hsv | float32 | grayscale | 1 | 32.5043 | 0.3684 | 88.23x |
| color.rgb2hsv | float32 | grayscale | 4 | 32.3209 | 0.1405 | 230.03x |
| color.hsv2rgb | float32 | wrapped-hue | 1 | 27.1808 | 0.9792 | 27.76x |
| color.hsv2rgb | float32 | wrapped-hue | 4 | 27.0054 | 0.6315 | 42.76x |
| color.xyz2lab | float32 | D50/10 | 1 | 14.3050 | 5.2815 | 2.71x |
| color.xyz2lab | float32 | D50/10 | 4 | 15.6055 | 1.5502 | 10.07x |
| color.lab2xyz | float32 | D50/10 | 1 | 17.4629 | 1.2226 | 14.28x |
| color.lab2xyz | float32 | D50/10 | 4 | 13.2110 | 0.1798 | 73.47x |
| color.rgb2lab | float32 | D50/10 | 1 | 24.5488 | 10.1980 | 2.41x |
| color.rgb2lab | float32 | D50/10 | 4 | 22.9971 | 5.1843 | 4.44x |
| color.lab2rgb | float32 | D50/10 | 1 | 22.4358 | 6.1483 | 3.65x |
| color.lab2rgb | float32 | D50/10 | 4 | 22.8092 | 1.7979 | 12.69x |
| color.lab2lch | float32 | four-channels | 1 | 8.9186 | 3.3664 | 2.65x |
| color.lab2lch | float32 | four-channels | 4 | 8.5734 | 1.5591 | 5.50x |
| color.lch2lab | float32 | four-channels | 1 | 3.3067 | 1.5840 | 2.09x |
| color.lch2lab | float32 | four-channels | 4 | 2.9173 | 1.2061 | 2.42x |
| color.lab2xyz | float32 | negative-Z | 1 | 12.1118 | 0.7443 | 16.27x |
| color.lab2xyz | float32 | negative-Z | 4 | 9.9320 | 0.4427 | 22.44x |
| color.lab2rgb | float32 | negative-Z | 1 | 22.2462 | 7.5326 | 2.95x |
| color.lab2rgb | float32 | negative-Z | 4 | 22.3743 | 3.0302 | 7.38x |
| color.rgb2hsv | float64 | grayscale | 1 | 43.1269 | 1.8700 | 23.06x |
| color.rgb2hsv | float64 | grayscale | 4 | 39.2388 | 1.6357 | 23.99x |
| color.hsv2rgb | float64 | wrapped-hue | 1 | 45.5468 | 1.5527 | 29.33x |
| color.hsv2rgb | float64 | wrapped-hue | 4 | 46.9454 | 1.6079 | 29.20x |
| color.xyz2lab | float64 | D50/10 | 1 | 21.7347 | 7.8243 | 2.78x |
| color.xyz2lab | float64 | D50/10 | 4 | 21.2784 | 3.2637 | 6.52x |
| color.lab2xyz | float64 | D50/10 | 1 | 20.5706 | 1.4951 | 13.76x |
| color.lab2xyz | float64 | D50/10 | 4 | 20.6583 | 1.4981 | 13.79x |
| color.rgb2lab | float64 | D50/10 | 1 | 43.5536 | 14.9305 | 2.92x |
| color.rgb2lab | float64 | D50/10 | 4 | 45.1662 | 4.9398 | 9.14x |
| color.lab2rgb | float64 | D50/10 | 1 | 34.5937 | 7.7581 | 4.46x |
| color.lab2rgb | float64 | D50/10 | 4 | 35.9267 | 2.6334 | 13.64x |
| color.lab2lch | float64 | four-channels | 1 | 10.3460 | 4.9740 | 2.08x |
| color.lab2lch | float64 | four-channels | 4 | 10.3347 | 2.4616 | 4.20x |
| color.lch2lab | float64 | four-channels | 1 | 10.0118 | 3.0896 | 3.24x |
| color.lch2lab | float64 | four-channels | 4 | 8.9775 | 1.6425 | 5.47x |
| color.lab2xyz | float64 | negative-Z | 1 | 15.1634 | 1.4013 | 10.82x |
| color.lab2xyz | float64 | negative-Z | 4 | 14.9581 | 0.6112 | 24.47x |
| color.lab2rgb | float64 | negative-Z | 1 | 30.8476 | 7.6438 | 4.04x |
| color.lab2rgb | float64 | negative-Z | 4 | 29.1513 | 2.8555 | 10.21x |
