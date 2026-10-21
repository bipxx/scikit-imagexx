# Perceptual color parameter performance

Strict native dispatch versus original scikit-image. Values, dtype and warnings are checked before timing. Seven warmed allocation-inclusive samples; one-thread reference. UserWarning is ignored during timing, while native negative-Z counting remains included. Float32 inverse Luv checks each rounding stage and a propagated direct RGB bound; the JSON records its contract and measured errors. Speedup = reference/current; slower cases remain visible.

| API | Input | Workload | Workers | Original ms | C++ ms | Speedup |
|---|---|---|---:|---:|---:|---:|
| color.rgb2hsv | float32 | grayscale | 1 | 34.3635 | 0.4006 | 85.78x |
| color.rgb2hsv | float32 | grayscale | 4 | 45.6395 | 0.3435 | 132.87x |
| color.hsv2rgb | float32 | wrapped-hue | 1 | 33.8943 | 0.8039 | 42.16x |
| color.hsv2rgb | float32 | wrapped-hue | 4 | 36.6336 | 0.7902 | 46.36x |
| color.xyz2lab | float32 | D50/10 | 1 | 19.8413 | 7.3448 | 2.70x |
| color.xyz2lab | float32 | D50/10 | 4 | 19.9373 | 3.9574 | 5.04x |
| color.lab2xyz | float32 | D50/10 | 1 | 20.5924 | 1.6544 | 12.45x |
| color.lab2xyz | float32 | D50/10 | 4 | 21.3652 | 0.9446 | 22.62x |
| color.rgb2lab | float32 | D50/10 | 1 | 37.2817 | 18.4860 | 2.02x |
| color.rgb2lab | float32 | D50/10 | 4 | 36.9163 | 5.2501 | 7.03x |
| color.lab2rgb | float32 | D50/10 | 1 | 34.6771 | 12.1993 | 2.84x |
| color.lab2rgb | float32 | D50/10 | 4 | 34.7890 | 4.2858 | 8.12x |
| color.lab2lch | float32 | four-channels | 1 | 11.1446 | 4.8424 | 2.30x |
| color.lab2lch | float32 | four-channels | 4 | 10.2467 | 1.9740 | 5.19x |
| color.lch2lab | float32 | four-channels | 1 | 3.7393 | 2.2289 | 1.68x |
| color.lch2lab | float32 | four-channels | 4 | 2.9421 | 1.2882 | 2.28x |
| color.lab2xyz | float32 | negative-Z | 1 | 13.9933 | 1.4165 | 9.88x |
| color.lab2xyz | float32 | negative-Z | 4 | 14.0368 | 0.5010 | 28.02x |
| color.lab2rgb | float32 | negative-Z | 1 | 25.0918 | 4.6944 | 5.35x |
| color.lab2rgb | float32 | negative-Z | 4 | 28.6807 | 2.2024 | 13.02x |
| color.xyz2luv | float32 | D50/10 | 1 | 10.4666 | 3.1537 | 3.32x |
| color.xyz2luv | float32 | D50/10 | 4 | 10.4408 | 1.3972 | 7.47x |
| color.luv2xyz | float32 | D50/10 | 1 | 9.9052 | 4.3142 | 2.30x |
| color.luv2xyz | float32 | D50/10 | 4 | 11.0816 | 2.1360 | 5.19x |
| color.rgb2luv | float32 | dark-sRGB | 1 | 13.1782 | 0.6650 | 19.82x |
| color.rgb2luv | float32 | dark-sRGB | 4 | 8.3126 | 0.4488 | 18.52x |
| color.luv2rgb | float32 | low-lightness | 1 | 24.3655 | 13.2849 | 1.83x |
| color.luv2rgb | float32 | low-lightness | 4 | 20.8080 | 3.6581 | 5.69x |
| color.luv2rgb | float32 | Luv-midtones | 1 | 30.9536 | 17.9523 | 1.72x |
| color.luv2rgb | float32 | Luv-midtones | 4 | 31.2823 | 4.6151 | 6.78x |
| color.rgb2hsv | float64 | grayscale | 1 | 65.3331 | 4.2783 | 15.27x |
| color.rgb2hsv | float64 | grayscale | 4 | 60.9120 | 2.1096 | 28.87x |
| color.hsv2rgb | float64 | wrapped-hue | 1 | 60.7540 | 3.1240 | 19.45x |
| color.hsv2rgb | float64 | wrapped-hue | 4 | 75.6814 | 1.8801 | 40.25x |
| color.xyz2lab | float64 | D50/10 | 1 | 25.6518 | 8.5045 | 3.02x |
| color.xyz2lab | float64 | D50/10 | 4 | 28.0838 | 4.6163 | 6.08x |
| color.lab2xyz | float64 | D50/10 | 1 | 29.0663 | 3.6522 | 7.96x |
| color.lab2xyz | float64 | D50/10 | 4 | 31.6767 | 2.0480 | 15.47x |
| color.rgb2lab | float64 | D50/10 | 1 | 63.5928 | 23.0158 | 2.76x |
| color.rgb2lab | float64 | D50/10 | 4 | 54.6930 | 7.0132 | 7.80x |
| color.lab2rgb | float64 | D50/10 | 1 | 50.0961 | 14.3852 | 3.48x |
| color.lab2rgb | float64 | D50/10 | 4 | 55.6161 | 5.3518 | 10.39x |
| color.lab2lch | float64 | four-channels | 1 | 17.0694 | 7.2093 | 2.37x |
| color.lab2lch | float64 | four-channels | 4 | 13.1120 | 3.1557 | 4.15x |
| color.lch2lab | float64 | four-channels | 1 | 14.6663 | 4.3858 | 3.34x |
| color.lch2lab | float64 | four-channels | 4 | 11.0736 | 2.3922 | 4.63x |
| color.lab2xyz | float64 | negative-Z | 1 | 23.2440 | 1.8242 | 12.74x |
| color.lab2xyz | float64 | negative-Z | 4 | 22.8392 | 1.0831 | 21.09x |
| color.lab2rgb | float64 | negative-Z | 1 | 37.5175 | 5.4775 | 6.85x |
| color.lab2rgb | float64 | negative-Z | 4 | 37.3311 | 2.5810 | 14.46x |
| color.xyz2luv | float64 | D50/10 | 1 | 19.5465 | 6.0600 | 3.23x |
| color.xyz2luv | float64 | D50/10 | 4 | 19.6225 | 2.7221 | 7.21x |
| color.luv2xyz | float64 | D50/10 | 1 | 20.7063 | 4.1207 | 5.02x |
| color.luv2xyz | float64 | D50/10 | 4 | 20.9868 | 1.3388 | 15.68x |
| color.rgb2luv | float64 | dark-sRGB | 1 | 25.2896 | 2.9885 | 8.46x |
| color.rgb2luv | float64 | dark-sRGB | 4 | 22.7302 | 1.8655 | 12.18x |
| color.luv2rgb | float64 | low-lightness | 1 | 41.3775 | 10.3291 | 4.01x |
| color.luv2rgb | float64 | low-lightness | 4 | 40.8897 | 4.2395 | 9.64x |
| color.luv2rgb | float64 | Luv-midtones | 1 | 46.3879 | 9.5699 | 4.85x |
| color.luv2rgb | float64 | Luv-midtones | 4 | 44.1531 | 4.3693 | 10.11x |
