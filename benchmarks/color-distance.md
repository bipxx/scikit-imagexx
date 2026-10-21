# Color distance and route performance

Native dispatch versus original scikit-image, with parity checked before warmed, allocation-inclusive timing. Reference pools use one thread. Raw Lab integer values are not intensity-normalized. Mixed dtypes, broadcast/strided/planar inputs, extra channels and RGB intermediate allocations are included. Ill-conditioned HSV routes check each stage and a propagated ratio error bound. RGB identity rows perform no image processing and are excluded from processing speedup counts.

| API | Input | Workload | Workers | Original ms | C++ ms | Speedup |
|---|---|---|---:|---:|---:|---:|
| color.deltaE_cie76 | uint8 | contiguous | 1 | 2.7780 | 0.6046 | 4.59x |
| color.deltaE_cie76 | uint8 | contiguous | 4 | 3.5884 | 0.2000 | 17.94x |
| color.deltaE_cie76 | uint8 | channel-first | 1 | 2.3420 | 0.5723 | 4.09x |
| color.deltaE_cie76 | uint8 | channel-first | 4 | 3.0966 | 0.2129 | 14.55x |
| color.deltaE_cie76 | uint8 | strided | 1 | 4.7078 | 1.4508 | 3.24x |
| color.deltaE_cie76 | uint8 | strided | 4 | 5.8422 | 1.3660 | 4.28x |
| color.deltaE_cie76 | uint8 | broadcast | 1 | 3.3665 | 1.1845 | 2.84x |
| color.deltaE_cie76 | uint8 | broadcast | 4 | 3.1807 | 0.2124 | 14.98x |
| color.deltaE_cie76 | uint8 | extra-channels | 1 | 5.0633 | 0.7330 | 6.91x |
| color.deltaE_cie76 | uint8 | extra-channels | 4 | 6.5007 | 0.3895 | 16.69x |
| color.convert_colorspace | uint8 | RGB->HSV | 1 | 37.0465 | 2.1390 | 17.32x |
| color.convert_colorspace | uint8 | RGB->HSV | 4 | 36.7021 | 0.7570 | 48.48x |
| color.convert_colorspace | float64 | HSV->XYZ | 1 | 44.3422 | 6.3777 | 6.95x |
| color.convert_colorspace | float64 | HSV->XYZ | 4 | 45.8846 | 1.7840 | 25.72x |
| color.convert_colorspace | float64 | XYZ->HSV | 1 | 48.3449 | 6.2593 | 7.72x |
| color.convert_colorspace | float64 | XYZ->HSV | 4 | 50.3092 | 1.8031 | 27.90x |
| color.convert_colorspace | float64 | YUV->YCbCr | 1 | 2.6785 | 0.7128 | 3.76x |
| color.convert_colorspace | float64 | YUV->YCbCr | 4 | 3.7696 | 0.4033 | 9.35x |
| color.convert_colorspace | float64 | YCbCr->YUV | 1 | 3.0845 | 0.7650 | 4.03x |
| color.convert_colorspace | float64 | YCbCr->YUV | 4 | 3.2723 | 0.3969 | 8.24x |
| color.convert_colorspace | float64 | RGB CIE->RGB | 1 | 1.1444 | 0.5712 | 2.00x |
| color.convert_colorspace | float64 | RGB CIE->RGB | 4 | 1.0867 | 0.1548 | 7.02x |
| color.convert_colorspace | uint8 | RGB-identity | 1 | 0.0009 | 0.0009 | 1.02x |
| color.convert_colorspace | uint8 | RGB-identity | 4 | 0.0007 | 0.0009 | 0.85x |
| color.deltaE_cie76 | float32 | contiguous | 1 | 0.8760 | 0.2535 | 3.46x |
| color.deltaE_cie76 | float32 | contiguous | 4 | 1.3171 | 0.2007 | 6.56x |
| color.deltaE_cie76 | float32 | channel-first | 1 | 0.5276 | 0.2870 | 1.84x |
| color.deltaE_cie76 | float32 | channel-first | 4 | 0.4862 | 0.2276 | 2.14x |
| color.deltaE_cie76 | float32 | strided | 1 | 1.2387 | 2.0297 | 0.61x |
| color.deltaE_cie76 | float32 | strided | 4 | 1.7011 | 1.4520 | 1.17x |
| color.deltaE_cie76 | float32 | broadcast | 1 | 0.7659 | 0.2040 | 3.75x |
| color.deltaE_cie76 | float32 | broadcast | 4 | 0.7998 | 0.2149 | 3.72x |
| color.deltaE_cie76 | float32 | extra-channels | 1 | 0.8235 | 0.3395 | 2.43x |
| color.deltaE_cie76 | float32 | extra-channels | 4 | 1.0667 | 0.2913 | 3.66x |
| color.convert_colorspace | float32 | RGB->HSV | 1 | 33.2294 | 0.4139 | 80.29x |
| color.convert_colorspace | float32 | RGB->HSV | 4 | 33.8836 | 0.2719 | 124.63x |
| color.convert_colorspace | float32 | HSV->XYZ | 1 | 29.6656 | 5.3392 | 5.56x |
| color.convert_colorspace | float32 | HSV->XYZ | 4 | 29.4399 | 2.4410 | 12.06x |
| color.convert_colorspace | float32 | XYZ->HSV | 1 | 43.1171 | 5.4002 | 7.98x |
| color.convert_colorspace | float32 | XYZ->HSV | 4 | 40.7914 | 1.7015 | 23.97x |
| color.convert_colorspace | float32 | YUV->YCbCr | 1 | 1.8272 | 0.4722 | 3.87x |
| color.convert_colorspace | float32 | YUV->YCbCr | 4 | 1.8473 | 0.3468 | 5.33x |
| color.convert_colorspace | float32 | YCbCr->YUV | 1 | 2.0452 | 0.5359 | 3.82x |
| color.convert_colorspace | float32 | YCbCr->YUV | 4 | 1.8584 | 0.2037 | 9.12x |
| color.convert_colorspace | float32 | RGB CIE->RGB | 1 | 0.7174 | 0.2660 | 2.70x |
| color.convert_colorspace | float32 | RGB CIE->RGB | 4 | 0.6375 | 0.0883 | 7.22x |
| color.convert_colorspace | float32 | RGB-identity | 1 | 0.0013 | 0.0016 | 0.82x |
| color.convert_colorspace | float32 | RGB-identity | 4 | 0.0016 | 0.0017 | 0.91x |
| color.deltaE_cie76 | float64 | contiguous | 1 | 2.4738 | 0.6832 | 3.62x |
| color.deltaE_cie76 | float64 | contiguous | 4 | 1.9777 | 0.2896 | 6.83x |
| color.deltaE_cie76 | float64 | channel-first | 1 | 1.4289 | 0.6531 | 2.19x |
| color.deltaE_cie76 | float64 | channel-first | 4 | 1.3962 | 0.3290 | 4.24x |
| color.deltaE_cie76 | float64 | strided | 1 | 3.1995 | 1.7490 | 1.83x |
| color.deltaE_cie76 | float64 | strided | 4 | 3.4249 | 1.7004 | 2.01x |
| color.deltaE_cie76 | float64 | broadcast | 1 | 1.5479 | 0.4367 | 3.54x |
| color.deltaE_cie76 | float64 | broadcast | 4 | 1.3722 | 0.1960 | 7.00x |
| color.deltaE_cie76 | float64 | extra-channels | 1 | 2.1590 | 0.5724 | 3.77x |
| color.deltaE_cie76 | float64 | extra-channels | 4 | 2.3950 | 0.7219 | 3.32x |
| color.convert_colorspace | float64 | RGB->HSV | 1 | 35.1688 | 1.0162 | 34.61x |
| color.convert_colorspace | float64 | RGB->HSV | 4 | 34.3554 | 0.3948 | 87.02x |
| color.convert_colorspace | float64 | HSV->XYZ | 1 | 46.1148 | 6.2693 | 7.36x |
| color.convert_colorspace | float64 | HSV->XYZ | 4 | 49.6382 | 1.9948 | 24.88x |
| color.convert_colorspace | float64 | XYZ->HSV | 1 | 48.3329 | 6.5536 | 7.38x |
| color.convert_colorspace | float64 | XYZ->HSV | 4 | 48.7578 | 2.4522 | 19.88x |
| color.convert_colorspace | float64 | YUV->YCbCr | 1 | 3.2337 | 0.8815 | 3.67x |
| color.convert_colorspace | float64 | YUV->YCbCr | 4 | 3.2898 | 0.5115 | 6.43x |
| color.convert_colorspace | float64 | YCbCr->YUV | 1 | 3.4688 | 0.6400 | 5.42x |
| color.convert_colorspace | float64 | YCbCr->YUV | 4 | 3.4959 | 0.6681 | 5.23x |
| color.convert_colorspace | float64 | RGB CIE->RGB | 1 | 1.3991 | 0.8737 | 1.60x |
| color.convert_colorspace | float64 | RGB CIE->RGB | 4 | 1.0418 | 0.4300 | 2.42x |
| color.convert_colorspace | float64 | RGB-identity | 1 | 0.0006 | 0.0006 | 1.00x |
| color.convert_colorspace | float64 | RGB-identity | 4 | 0.0015 | 0.0012 | 1.24x |
| color.deltaE_cie76 | uint16/float32 | mixed-dtypes | 1 | 3.9361 | 0.6691 | 5.88x |
| color.deltaE_cie76 | uint16/float32 | mixed-dtypes | 4 | 3.7703 | 0.2745 | 13.73x |
| color.deltaE_cie76 | float32/float64 | mixed-dtypes | 1 | 3.6831 | 0.5916 | 6.23x |
| color.deltaE_cie76 | float32/float64 | mixed-dtypes | 4 | 2.5190 | 0.3522 | 7.15x |
| color.deltaE_cie76 | int64/float32 | mixed-dtypes | 1 | 4.0748 | 1.7144 | 2.38x |
| color.deltaE_cie76 | int64/float32 | mixed-dtypes | 4 | 4.7460 | 0.4735 | 10.02x |
| color.deltaE_cie76 | bool/float32 | mixed-dtypes | 1 | 3.9287 | 0.7230 | 5.43x |
| color.deltaE_cie76 | bool/float32 | mixed-dtypes | 4 | 5.0587 | 0.3626 | 13.95x |
| color.convert_colorspace | float32 | out-of-gamut-YUV->HSV | 1 | 34.5166 | 0.6190 | 55.76x |
| color.convert_colorspace | float32 | out-of-gamut-YUV->HSV | 4 | 32.3531 | 0.2317 | 139.61x |
| color.convert_colorspace | float64 | out-of-gamut-YUV->HSV | 1 | 41.2559 | 1.7940 | 23.00x |
| color.convert_colorspace | float64 | out-of-gamut-YUV->HSV | 4 | 36.1880 | 0.5950 | 60.82x |
