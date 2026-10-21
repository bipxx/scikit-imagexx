# Perceptual color parameter performance

Strict native dispatch versus original scikit-image. Values, dtype and warnings are checked before timing. Seven warmed allocation-inclusive samples; one-thread reference. UserWarning is ignored during timing, while native negative-Z counting remains included. Float32 inverse Luv checks each rounding stage and a propagated direct RGB bound; the JSON records its contract and measured errors. Speedup = reference/current; slower cases remain visible.

| API | Input | Workload | Workers | Original ms | C++ ms | Speedup |
|---|---|---|---:|---:|---:|---:|
| color.rgb2hsv | float32 | grayscale | 1 | 78.1086 | 0.9121 | 85.64x |
| color.hsv2rgb | float32 | wrapped-hue | 1 | 45.8725 | 0.7419 | 61.83x |
| color.xyz2lab | float32 | D50/10 | 1 | 24.2591 | 7.2355 | 3.35x |
| color.lab2xyz | float32 | D50/10 | 1 | 22.4940 | 1.7211 | 13.07x |
| color.rgb2lab | float32 | D50/10 | 1 | 43.5551 | 16.6364 | 2.62x |
| color.lab2rgb | float32 | D50/10 | 1 | 36.5544 | 9.2325 | 3.96x |
| color.lab2lch | float32 | four-channels | 1 | 12.0318 | 4.0619 | 2.96x |
| color.lch2lab | float32 | four-channels | 1 | 4.7758 | 2.8667 | 1.67x |
| color.lab2xyz | float32 | negative-Z | 1 | 16.5641 | 1.2206 | 13.57x |
| color.lab2rgb | float32 | negative-Z | 1 | 30.4384 | 4.2358 | 7.19x |
| color.xyz2luv | float32 | D50/10 | 1 | 11.4973 | 3.6462 | 3.15x |
| color.luv2xyz | float32 | D50/10 | 1 | 11.5472 | 3.8811 | 2.98x |
| color.rgb2luv | float32 | dark-sRGB | 1 | 17.0900 | 0.6542 | 26.12x |
| color.luv2rgb | float32 | low-lightness | 1 | 22.6203 | 7.7364 | 2.92x |
| color.luv2rgb | float32 | Luv-midtones | 1 | 32.6906 | 11.2641 | 2.90x |
| color.rgb2hsv | float64 | grayscale | 1 | 88.8532 | 3.0356 | 29.27x |
| color.hsv2rgb | float64 | wrapped-hue | 1 | 90.2491 | 3.6191 | 24.94x |
| color.xyz2lab | float64 | D50/10 | 1 | 34.5630 | 9.7507 | 3.54x |
| color.lab2xyz | float64 | D50/10 | 1 | 37.9921 | 4.2440 | 8.95x |
| color.rgb2lab | float64 | D50/10 | 1 | 77.4094 | 17.4547 | 4.43x |
| color.lab2rgb | float64 | D50/10 | 1 | 80.8861 | 11.4095 | 7.09x |
| color.lab2lch | float64 | four-channels | 1 | 21.3491 | 7.6346 | 2.80x |
| color.lch2lab | float64 | four-channels | 1 | 17.9792 | 6.2748 | 2.87x |
| color.lab2xyz | float64 | negative-Z | 1 | 31.5469 | 1.4234 | 22.16x |
| color.lab2rgb | float64 | negative-Z | 1 | 60.4359 | 4.8950 | 12.35x |
| color.xyz2luv | float64 | D50/10 | 1 | 31.3255 | 5.8290 | 5.37x |
| color.luv2xyz | float64 | D50/10 | 1 | 42.2752 | 3.5105 | 12.04x |
| color.rgb2luv | float64 | dark-sRGB | 1 | 39.9673 | 3.2588 | 12.26x |
| color.luv2rgb | float64 | low-lightness | 1 | 88.3567 | 10.2512 | 8.62x |
| color.luv2rgb | float64 | Luv-midtones | 1 | 77.0421 | 9.9026 | 7.78x |
