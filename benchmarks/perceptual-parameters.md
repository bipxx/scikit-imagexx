# Perceptual color parameter performance

Strict native dispatch versus original scikit-image. Values, dtype and warnings are checked before timing. Seven warmed allocation-inclusive samples; one-thread reference. UserWarning is ignored during timing, while native negative-Z counting remains included. Float32 inverse Luv checks each rounding stage and a propagated direct RGB bound; the JSON records its contract and measured errors. Speedup = reference/current; slower cases remain visible.

| API | Input | Workload | Workers | Original ms | C++ ms | Speedup |
|---|---|---|---:|---:|---:|---:|
| color.rgb2hsv | float32 | grayscale | 1 | 33.2150 | 0.4047 | 82.07x |
| color.hsv2rgb | float32 | wrapped-hue | 1 | 29.5633 | 0.5899 | 50.12x |
| color.xyz2lab | float32 | D50/10 | 1 | 14.6995 | 5.0739 | 2.90x |
| color.lab2xyz | float32 | D50/10 | 1 | 15.5959 | 0.5292 | 29.47x |
| color.rgb2lab | float32 | D50/10 | 1 | 37.0207 | 13.6890 | 2.70x |
| color.lab2rgb | float32 | D50/10 | 1 | 30.1962 | 7.1140 | 4.24x |
| color.lab2lch | float32 | four-channels | 1 | 10.1705 | 3.5566 | 2.86x |
| color.lch2lab | float32 | four-channels | 1 | 2.6797 | 2.2905 | 1.17x |
| color.lab2xyz | float32 | negative-Z | 1 | 10.3967 | 1.1480 | 9.06x |
| color.lab2rgb | float32 | negative-Z | 1 | 21.6561 | 5.0775 | 4.27x |
| color.xyz2luv | float32 | D50/10 | 1 | 6.8560 | 2.7483 | 2.49x |
| color.luv2xyz | float32 | D50/10 | 1 | 8.1897 | 2.9321 | 2.79x |
| color.rgb2luv | float32 | dark-sRGB | 1 | 7.3621 | 0.5187 | 14.19x |
| color.luv2rgb | float32 | low-lightness | 1 | 19.8927 | 7.8461 | 2.54x |
| color.luv2rgb | float32 | Luv-midtones | 1 | 21.9786 | 9.5551 | 2.30x |
| color.rgb2hsv | float64 | grayscale | 1 | 42.8107 | 2.5332 | 16.90x |
| color.hsv2rgb | float64 | wrapped-hue | 1 | 53.0890 | 2.9731 | 17.86x |
| color.xyz2lab | float64 | D50/10 | 1 | 23.7118 | 8.3306 | 2.85x |
| color.lab2xyz | float64 | D50/10 | 1 | 24.6019 | 2.9591 | 8.31x |
| color.rgb2lab | float64 | D50/10 | 1 | 37.5031 | 13.2606 | 2.83x |
| color.lab2rgb | float64 | D50/10 | 1 | 36.9489 | 8.1692 | 4.52x |
| color.lab2lch | float64 | four-channels | 1 | 14.0839 | 5.8415 | 2.41x |
| color.lch2lab | float64 | four-channels | 1 | 15.3436 | 6.0772 | 2.52x |
| color.lab2xyz | float64 | negative-Z | 1 | 21.9428 | 1.4596 | 15.03x |
| color.lab2rgb | float64 | negative-Z | 1 | 29.7673 | 3.5186 | 8.46x |
| color.xyz2luv | float64 | D50/10 | 1 | 18.7872 | 3.7067 | 5.07x |
| color.luv2xyz | float64 | D50/10 | 1 | 14.5551 | 1.0148 | 14.34x |
| color.rgb2luv | float64 | dark-sRGB | 1 | 15.1782 | 1.3444 | 11.29x |
| color.luv2rgb | float64 | low-lightness | 1 | 36.5722 | 9.7615 | 3.75x |
| color.luv2rgb | float64 | Luv-midtones | 1 | 38.3599 | 7.5144 | 5.10x |
