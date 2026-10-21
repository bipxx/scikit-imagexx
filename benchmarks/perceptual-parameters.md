# Perceptual color parameter performance

Strict native dispatch versus original scikit-image. Values, dtype and warnings are checked before timing. Seven warmed allocation-inclusive samples; one-thread reference. UserWarning is ignored during timing, while native negative-Z counting remains included. Float32 inverse Luv checks each rounding stage and a propagated direct RGB bound; the JSON records its contract and measured errors. Speedup = reference/current; slower cases remain visible.

| API | Input | Workload | Workers | Original ms | C++ ms | Speedup |
|---|---|---|---:|---:|---:|---:|
| color.rgb2hsv | float32 | grayscale | 1 | 38.3496 | 0.4591 | 83.52x |
| color.hsv2rgb | float32 | wrapped-hue | 1 | 40.6105 | 1.0487 | 38.72x |
| color.xyz2lab | float32 | D50/10 | 1 | 17.0690 | 5.7336 | 2.98x |
| color.lab2xyz | float32 | D50/10 | 1 | 23.3207 | 1.1037 | 21.13x |
| color.rgb2lab | float32 | D50/10 | 1 | 39.3903 | 13.8893 | 2.84x |
| color.lab2rgb | float32 | D50/10 | 1 | 35.6072 | 7.6997 | 4.62x |
| color.lab2lch | float32 | four-channels | 1 | 7.7936 | 3.0904 | 2.52x |
| color.lch2lab | float32 | four-channels | 1 | 4.0161 | 2.3681 | 1.70x |
| color.lab2xyz | float32 | negative-Z | 1 | 12.6955 | 1.2669 | 10.02x |
| color.lab2rgb | float32 | negative-Z | 1 | 22.8347 | 4.1588 | 5.49x |
| color.xyz2luv | float32 | D50/10 | 1 | 7.0635 | 3.7483 | 1.88x |
| color.luv2xyz | float32 | D50/10 | 1 | 9.3682 | 3.0600 | 3.06x |
| color.rgb2luv | float32 | dark-sRGB | 1 | 8.8651 | 0.4022 | 22.04x |
| color.luv2rgb | float32 | low-lightness | 1 | 19.4034 | 6.3167 | 3.07x |
| color.luv2rgb | float32 | Luv-midtones | 1 | 22.8242 | 8.3148 | 2.75x |
| color.rgb2hsv | float64 | grayscale | 1 | 38.2094 | 2.2598 | 16.91x |
| color.hsv2rgb | float64 | wrapped-hue | 1 | 48.0452 | 3.0603 | 15.70x |
| color.xyz2lab | float64 | D50/10 | 1 | 21.5598 | 5.8644 | 3.68x |
| color.lab2xyz | float64 | D50/10 | 1 | 21.0307 | 2.5950 | 8.10x |
| color.rgb2lab | float64 | D50/10 | 1 | 46.2726 | 17.8262 | 2.60x |
| color.lab2rgb | float64 | D50/10 | 1 | 53.1906 | 10.5965 | 5.02x |
| color.lab2lch | float64 | four-channels | 1 | 16.3200 | 5.2190 | 3.13x |
| color.lch2lab | float64 | four-channels | 1 | 10.0434 | 3.3302 | 3.02x |
| color.lab2xyz | float64 | negative-Z | 1 | 18.4045 | 1.3548 | 13.58x |
| color.lab2rgb | float64 | negative-Z | 1 | 25.1333 | 3.3066 | 7.60x |
| color.xyz2luv | float64 | D50/10 | 1 | 20.1698 | 4.2248 | 4.77x |
| color.luv2xyz | float64 | D50/10 | 1 | 20.4461 | 2.2288 | 9.17x |
| color.rgb2luv | float64 | dark-sRGB | 1 | 18.4950 | 2.2997 | 8.04x |
| color.luv2rgb | float64 | low-lightness | 1 | 32.4086 | 6.7951 | 4.77x |
| color.luv2rgb | float64 | Luv-midtones | 1 | 35.0297 | 6.6388 | 5.28x |
