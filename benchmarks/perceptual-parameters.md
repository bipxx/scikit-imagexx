# Perceptual color parameter performance

Strict native dispatch versus original scikit-image. Values, dtype and warnings are checked before timing. Seven warmed allocation-inclusive samples; one-thread reference. UserWarning is ignored during timing, while native negative-Z counting remains included. Speedup = reference/current; slower cases remain visible.

| API | Input | Workload | Workers | Original ms | C++ ms | Speedup |
|---|---|---|---:|---:|---:|---:|
| color.rgb2hsv | float32 | grayscale | 1 | 31.4120 | 0.3908 | 80.38x |
| color.rgb2hsv | float32 | grayscale | 4 | 29.9421 | 0.1369 | 218.76x |
| color.hsv2rgb | float32 | wrapped-hue | 1 | 24.2934 | 0.5868 | 41.40x |
| color.hsv2rgb | float32 | wrapped-hue | 4 | 23.9263 | 0.2099 | 113.97x |
| color.xyz2lab | float32 | D50/10 | 1 | 13.0319 | 5.2988 | 2.46x |
| color.xyz2lab | float32 | D50/10 | 4 | 13.7814 | 2.0110 | 6.85x |
| color.lab2xyz | float32 | D50/10 | 1 | 18.3162 | 0.7315 | 25.04x |
| color.lab2xyz | float32 | D50/10 | 4 | 13.5523 | 0.2447 | 55.37x |
| color.rgb2lab | float32 | D50/10 | 1 | 25.6727 | 10.9736 | 2.34x |
| color.rgb2lab | float32 | D50/10 | 4 | 22.4413 | 2.8635 | 7.84x |
| color.lab2rgb | float32 | D50/10 | 1 | 22.9132 | 6.5433 | 3.50x |
| color.lab2rgb | float32 | D50/10 | 4 | 22.9885 | 2.0418 | 11.26x |
| color.lab2lch | float32 | four-channels | 1 | 7.3144 | 2.8708 | 2.55x |
| color.lab2lch | float32 | four-channels | 4 | 7.0307 | 1.3466 | 5.22x |
| color.lch2lab | float32 | four-channels | 1 | 1.6649 | 1.4415 | 1.15x |
| color.lch2lab | float32 | four-channels | 4 | 1.7516 | 0.7815 | 2.24x |
| color.lab2xyz | float32 | negative-Z | 1 | 11.9790 | 0.6020 | 19.90x |
| color.lab2xyz | float32 | negative-Z | 4 | 11.9811 | 0.3910 | 30.64x |
| color.lab2rgb | float32 | negative-Z | 1 | 19.4626 | 8.1802 | 2.38x |
| color.lab2rgb | float32 | negative-Z | 4 | 19.2806 | 2.1812 | 8.84x |
| color.xyz2luv | float32 | D50/10 | 1 | 5.7737 | 2.1307 | 2.71x |
| color.xyz2luv | float32 | D50/10 | 4 | 6.4572 | 1.1756 | 5.49x |
| color.luv2xyz | float32 | D50/10 | 1 | 4.8807 | 2.1400 | 2.28x |
| color.luv2xyz | float32 | D50/10 | 4 | 7.2052 | 1.6285 | 4.42x |
| color.rgb2luv | float32 | dark-sRGB | 1 | 6.8541 | 7.8616 | 0.87x |
| color.rgb2luv | float32 | dark-sRGB | 4 | 6.0426 | 2.3207 | 2.60x |
| color.luv2rgb | float32 | low-lightness | 1 | 15.6946 | 5.6458 | 2.78x |
| color.luv2rgb | float32 | low-lightness | 4 | 17.4070 | 1.5704 | 11.08x |
| color.luv2rgb | float32 | Luv-midtones | 1 | 19.9953 | 10.5655 | 1.89x |
| color.luv2rgb | float32 | Luv-midtones | 4 | 21.2862 | 2.9481 | 7.22x |
| color.rgb2hsv | float64 | grayscale | 1 | 33.5050 | 2.0428 | 16.40x |
| color.rgb2hsv | float64 | grayscale | 4 | 33.0767 | 1.1070 | 29.88x |
| color.hsv2rgb | float64 | wrapped-hue | 1 | 42.1941 | 2.1634 | 19.50x |
| color.hsv2rgb | float64 | wrapped-hue | 4 | 39.8781 | 1.2186 | 32.72x |
| color.xyz2lab | float64 | D50/10 | 1 | 17.8010 | 5.9440 | 2.99x |
| color.xyz2lab | float64 | D50/10 | 4 | 18.4510 | 2.4792 | 7.44x |
| color.lab2xyz | float64 | D50/10 | 1 | 18.8640 | 1.2549 | 15.03x |
| color.lab2xyz | float64 | D50/10 | 4 | 18.1646 | 1.0843 | 16.75x |
| color.rgb2lab | float64 | D50/10 | 1 | 35.4181 | 11.7487 | 3.01x |
| color.rgb2lab | float64 | D50/10 | 4 | 35.0269 | 3.6606 | 9.57x |
| color.lab2rgb | float64 | D50/10 | 1 | 33.3018 | 7.4585 | 4.46x |
| color.lab2rgb | float64 | D50/10 | 4 | 34.1286 | 2.4947 | 13.68x |
| color.lab2lch | float64 | four-channels | 1 | 9.0521 | 4.6065 | 1.97x |
| color.lab2lch | float64 | four-channels | 4 | 10.2631 | 1.7531 | 5.85x |
| color.lch2lab | float64 | four-channels | 1 | 8.8029 | 3.2639 | 2.70x |
| color.lch2lab | float64 | four-channels | 4 | 8.5890 | 1.8892 | 4.55x |
| color.lab2xyz | float64 | negative-Z | 1 | 13.2275 | 1.1625 | 11.38x |
| color.lab2xyz | float64 | negative-Z | 4 | 13.5270 | 0.4497 | 30.08x |
| color.lab2rgb | float64 | negative-Z | 1 | 23.8697 | 6.1328 | 3.89x |
| color.lab2rgb | float64 | negative-Z | 4 | 22.1800 | 1.9884 | 11.15x |
| color.xyz2luv | float64 | D50/10 | 1 | 13.2326 | 3.7944 | 3.49x |
| color.xyz2luv | float64 | D50/10 | 4 | 9.5002 | 1.6481 | 5.76x |
| color.luv2xyz | float64 | D50/10 | 1 | 14.7884 | 1.0432 | 14.18x |
| color.luv2xyz | float64 | D50/10 | 4 | 12.7498 | 0.2953 | 43.17x |
| color.rgb2luv | float64 | dark-sRGB | 1 | 12.2188 | 10.0961 | 1.21x |
| color.rgb2luv | float64 | dark-sRGB | 4 | 14.5473 | 3.9579 | 3.68x |
| color.luv2rgb | float64 | low-lightness | 1 | 25.6214 | 6.8730 | 3.73x |
| color.luv2rgb | float64 | low-lightness | 4 | 25.6233 | 2.2121 | 11.58x |
| color.luv2rgb | float64 | Luv-midtones | 1 | 30.9188 | 7.2720 | 4.25x |
| color.luv2rgb | float64 | Luv-midtones | 4 | 32.3396 | 2.3183 | 13.95x |
