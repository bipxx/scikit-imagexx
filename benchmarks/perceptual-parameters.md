# Perceptual color parameter performance

Strict native dispatch versus original scikit-image. Values, dtype and warnings are checked before timing. Seven warmed allocation-inclusive samples; one-thread reference. UserWarning is ignored during timing, while native negative-Z counting remains included. Float32 inverse Luv checks each rounding stage and a propagated direct RGB bound; the JSON records its contract and measured errors. Speedup = reference/current; slower cases remain visible.

| API | Input | Workload | Workers | Original ms | C++ ms | Speedup |
|---|---|---|---:|---:|---:|---:|
| color.rgb2hsv | float32 | grayscale | 1 | 31.5765 | 0.4181 | 75.53x |
| color.hsv2rgb | float32 | wrapped-hue | 1 | 27.4700 | 0.5362 | 51.23x |
| color.xyz2lab | float32 | D50/10 | 1 | 13.2853 | 5.0757 | 2.62x |
| color.lab2xyz | float32 | D50/10 | 1 | 14.9126 | 0.5826 | 25.60x |
| color.rgb2lab | float32 | D50/10 | 1 | 37.2520 | 16.0281 | 2.32x |
| color.lab2rgb | float32 | D50/10 | 1 | 30.2837 | 7.3533 | 4.12x |
| color.lab2lch | float32 | four-channels | 1 | 7.7219 | 3.6293 | 2.13x |
| color.lch2lab | float32 | four-channels | 1 | 2.7382 | 2.4865 | 1.10x |
| color.lab2xyz | float32 | negative-Z | 1 | 9.3167 | 1.2907 | 7.22x |
| color.lab2rgb | float32 | negative-Z | 1 | 18.7436 | 3.2961 | 5.69x |
| color.xyz2luv | float32 | D50/10 | 1 | 7.5325 | 2.7354 | 2.75x |
| color.luv2xyz | float32 | D50/10 | 1 | 8.3770 | 3.4226 | 2.45x |
| color.rgb2luv | float32 | dark-sRGB | 1 | 7.8177 | 0.6041 | 12.94x |
| color.luv2rgb | float32 | low-lightness | 1 | 22.5854 | 6.1381 | 3.68x |
| color.luv2rgb | float32 | Luv-midtones | 1 | 22.8760 | 9.0305 | 2.53x |
| color.rgb2hsv | float64 | grayscale | 1 | 41.4282 | 1.2294 | 33.70x |
| color.hsv2rgb | float64 | wrapped-hue | 1 | 48.3037 | 2.4157 | 20.00x |
| color.xyz2lab | float64 | D50/10 | 1 | 22.6033 | 6.5198 | 3.47x |
| color.lab2xyz | float64 | D50/10 | 1 | 21.7679 | 2.5750 | 8.45x |
| color.rgb2lab | float64 | D50/10 | 1 | 37.9550 | 13.0373 | 2.91x |
| color.lab2rgb | float64 | D50/10 | 1 | 38.3578 | 9.0771 | 4.23x |
| color.lab2lch | float64 | four-channels | 1 | 14.5573 | 5.0507 | 2.88x |
| color.lch2lab | float64 | four-channels | 1 | 12.0439 | 4.0397 | 2.98x |
| color.lab2xyz | float64 | negative-Z | 1 | 17.2334 | 1.1432 | 15.07x |
| color.lab2rgb | float64 | negative-Z | 1 | 34.1298 | 4.4989 | 7.59x |
| color.xyz2luv | float64 | D50/10 | 1 | 16.5663 | 3.9548 | 4.19x |
| color.luv2xyz | float64 | D50/10 | 1 | 19.3771 | 1.4285 | 13.56x |
| color.rgb2luv | float64 | dark-sRGB | 1 | 18.8329 | 1.7702 | 10.64x |
| color.luv2rgb | float64 | low-lightness | 1 | 36.5544 | 7.4853 | 4.88x |
| color.luv2rgb | float64 | Luv-midtones | 1 | 42.4895 | 8.6390 | 4.92x |
