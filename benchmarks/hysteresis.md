# Hysteresis threshold parity checkpoint

Native clipping and comparison use AVX2 where supported; connectivity uses serial flood fill with one worker and component merging with optional global workers. The default remains one worker. Face connectivity and upstream label-zero selection are preserved. Real dtype, storage, broadcast, NaN, scalar, warning and error coverage is documented in the main README.

## Source-specific initial measurements

[Successful Linux/Windows run](https://github.com/bipxx/scikit-imagexx/actions/runs/39793874955) measured `fa6a453`, product SHA256 `1d2108d5d6cd1a234ca14eace05b00872c0fe46742df7be8507e4c9d42e07c0c`, against scikit-image 0.26.0. Each OS has 32 cases at one and four workers, seven randomized timing samples, one reference thread, seed 1934 and size 512. Allocation, conversion and Python dispatch are included. Every timed result passed exact boolean parity with fallback disabled. Exact wheels and extracted packages are retained with manifests in the run artifacts.

Windows one-worker geometric mean across all 32 cases: **1.667x** upstream/native. [Raw data](hysteresis-initial-windows.json).
Linux one-worker geometric mean across all 32 cases: **1.733x** upstream/native. [Raw data](hysteresis-initial-linux.json).
The retained compatibility correction `cef6663` additionally matches legacy NumPy mixed integer comparisons and explicitly delegates ordinary Python scalar images. Its minimum-NumPy SIMD/scalar checks each pass 2,156 cases; modern Windows scoped SIMD/scalar checks each pass 6,300 cases. Native normal/dynamic/portable and ASan/UBSan normal/dynamic checks pass 2/2 each. Full CI and final-source timings remain pending; initial-source numbers below must not be attributed to the corrected fingerprint.

## Bottlenecks

Random, strided, Fortran and threshold-field cases can trail upstream: graph traversal and layout normalization remain optimization targets. SIMD masks do not remove irregular connectivity costs. All adverse cases and every raw sample are included below and in the linked JSON. This is partial coverage, not whole-library parity or a whole-library performance refresh.

| OS | dtype | layout | workers | upstream ms | native ms | upstream/native |
|---|---|---|---:|---:|---:|---:|
| Linux | uint8 | random | 1 | 4.91003 | 6.26278 | 0.784x |
| Linux | uint8 | random | 4 | 4.78192 | 3.31954 | 1.441x |
| Linux | uint8 | fortran | 1 | 7.90880 | 8.70182 | 0.909x |
| Linux | uint8 | fortran | 4 | 8.00991 | 5.40538 | 1.482x |
| Linux | uint8 | strided | 1 | 2.55621 | 3.25516 | 0.785x |
| Linux | uint8 | strided | 4 | 2.55568 | 3.25928 | 0.784x |
| Linux | uint8 | all-weak | 1 | 3.98193 | 0.98572 | 4.040x |
| Linux | uint8 | all-weak | 4 | 3.93987 | 0.72153 | 5.460x |
| Linux | uint8 | no-seeds | 1 | 4.87279 | 1.94332 | 2.507x |
| Linux | uint8 | no-seeds | 4 | 4.90033 | 1.06461 | 4.603x |
| Linux | uint8 | equal | 1 | 6.67676 | 2.10008 | 3.179x |
| Linux | uint8 | equal | 4 | 6.89178 | 1.12004 | 6.153x |
| Linux | uint8 | threshold-fields | 1 | 4.78112 | 6.02883 | 0.793x |
| Linux | uint8 | threshold-fields | 4 | 4.75643 | 3.25322 | 1.462x |
| Linux | uint8 | snake | 1 | 3.57305 | 1.57892 | 2.263x |
| Linux | uint8 | snake | 4 | 3.55116 | 1.40223 | 2.533x |
| Linux | uint8 | checkerboard | 1 | 6.04604 | 1.09661 | 5.513x |
| Linux | uint8 | checkerboard | 4 | 6.14983 | 1.35186 | 4.549x |
| Linux | uint8 | tiny | 1 | 0.10623 | 0.09723 | 1.093x |
| Linux | uint8 | tiny | 4 | 0.10539 | 0.09481 | 1.112x |
| Linux | float32 | random | 1 | 4.73430 | 6.27470 | 0.755x |
| Linux | float32 | random | 4 | 4.70029 | 3.33115 | 1.411x |
| Linux | float32 | fortran | 1 | 7.68675 | 9.97254 | 0.771x |
| Linux | float32 | fortran | 4 | 5.82249 | 5.19167 | 1.122x |
| Linux | float32 | strided | 1 | 2.56816 | 3.32630 | 0.772x |
| Linux | float32 | strided | 4 | 2.62211 | 3.36135 | 0.780x |
| Linux | float32 | all-weak | 1 | 4.03452 | 1.15624 | 3.489x |
| Linux | float32 | all-weak | 4 | 3.98511 | 0.79105 | 5.038x |
| Linux | float32 | no-seeds | 1 | 4.99850 | 2.05753 | 2.429x |
| Linux | float32 | no-seeds | 4 | 4.90319 | 1.01319 | 4.839x |
| Linux | float32 | equal | 1 | 7.02409 | 3.17549 | 2.212x |
| Linux | float32 | equal | 4 | 6.88474 | 2.75292 | 2.501x |
| Linux | float32 | threshold-fields | 1 | 7.39398 | 7.58408 | 0.975x |
| Linux | float32 | threshold-fields | 4 | 4.90929 | 3.45378 | 1.421x |
| Linux | float32 | snake | 1 | 3.51550 | 1.71575 | 2.049x |
| Linux | float32 | snake | 4 | 3.58988 | 1.47301 | 2.437x |
| Linux | float32 | checkerboard | 1 | 6.32356 | 1.18077 | 5.355x |
| Linux | float32 | checkerboard | 4 | 6.71331 | 1.52073 | 4.415x |
| Linux | float32 | tiny | 1 | 0.09914 | 0.08694 | 1.140x |
| Linux | float32 | tiny | 4 | 0.10014 | 0.08655 | 1.157x |
| Linux | float32 | nan-background | 1 | 3.24210 | 1.05231 | 3.081x |
| Linux | float32 | nan-background | 4 | 3.23682 | 0.74254 | 4.359x |
| Linux | float64 | random | 1 | 4.76044 | 6.22496 | 0.765x |
| Linux | float64 | random | 4 | 4.73903 | 3.26535 | 1.451x |
| Linux | float64 | fortran | 1 | 7.98825 | 9.21940 | 0.866x |
| Linux | float64 | fortran | 4 | 5.90260 | 5.44087 | 1.085x |
| Linux | float64 | strided | 1 | 2.57552 | 3.35988 | 0.767x |
| Linux | float64 | strided | 4 | 2.57032 | 3.33786 | 0.770x |
| Linux | float64 | all-weak | 1 | 3.88627 | 1.06901 | 3.635x |
| Linux | float64 | all-weak | 4 | 3.94601 | 0.70282 | 5.615x |
| Linux | float64 | no-seeds | 1 | 4.78125 | 1.94984 | 2.452x |
| Linux | float64 | no-seeds | 4 | 4.75864 | 0.92489 | 5.145x |
| Linux | float64 | equal | 1 | 7.17591 | 1.98956 | 3.607x |
| Linux | float64 | equal | 4 | 7.17988 | 0.92470 | 7.765x |
| Linux | float64 | threshold-fields | 1 | 5.06618 | 6.24216 | 0.812x |
| Linux | float64 | threshold-fields | 4 | 5.10025 | 3.46824 | 1.471x |
| Linux | float64 | snake | 1 | 3.54566 | 1.68826 | 2.100x |
| Linux | float64 | snake | 4 | 3.52692 | 1.37291 | 2.569x |
| Linux | float64 | checkerboard | 1 | 6.53146 | 1.06642 | 6.125x |
| Linux | float64 | checkerboard | 4 | 6.20793 | 1.34210 | 4.626x |
| Linux | float64 | tiny | 1 | 0.09861 | 0.07978 | 1.236x |
| Linux | float64 | tiny | 4 | 0.09923 | 0.08200 | 1.210x |
| Linux | float64 | nan-background | 1 | 3.27601 | 0.97533 | 3.359x |
| Linux | float64 | nan-background | 4 | 3.24323 | 0.66923 | 4.846x |
| Windows | uint8 | random | 1 | 8.61960 | 10.17730 | 0.847x |
| Windows | uint8 | random | 4 | 8.70190 | 6.40520 | 1.359x |
| Windows | uint8 | fortran | 1 | 9.43290 | 11.94560 | 0.790x |
| Windows | uint8 | fortran | 4 | 9.46190 | 7.86280 | 1.203x |
| Windows | uint8 | strided | 1 | 4.42720 | 5.15325 | 0.859x |
| Windows | uint8 | strided | 4 | 4.31597 | 5.13670 | 0.840x |
| Windows | uint8 | all-weak | 1 | 7.39490 | 2.95535 | 2.502x |
| Windows | uint8 | all-weak | 4 | 7.26095 | 1.97248 | 3.681x |
| Windows | uint8 | no-seeds | 1 | 8.63590 | 4.50427 | 1.917x |
| Windows | uint8 | no-seeds | 4 | 8.62520 | 3.08735 | 2.794x |
| Windows | uint8 | equal | 1 | 8.49640 | 3.50715 | 2.423x |
| Windows | uint8 | equal | 4 | 8.35530 | 2.17006 | 3.850x |
| Windows | uint8 | threshold-fields | 1 | 8.39200 | 9.09340 | 0.923x |
| Windows | uint8 | threshold-fields | 4 | 8.73290 | 5.86355 | 1.489x |
| Windows | uint8 | snake | 1 | 7.01840 | 3.05728 | 2.296x |
| Windows | uint8 | snake | 4 | 7.02835 | 3.54818 | 1.981x |
| Windows | uint8 | checkerboard | 1 | 8.96280 | 2.36512 | 3.790x |
| Windows | uint8 | checkerboard | 4 | 9.15130 | 3.18025 | 2.878x |
| Windows | uint8 | tiny | 1 | 0.06192 | 0.05856 | 1.057x |
| Windows | uint8 | tiny | 4 | 0.05639 | 0.05767 | 0.978x |
| Windows | float32 | random | 1 | 8.51550 | 10.14060 | 0.840x |
| Windows | float32 | random | 4 | 8.50270 | 6.20820 | 1.370x |
| Windows | float32 | fortran | 1 | 9.53180 | 12.16080 | 0.784x |
| Windows | float32 | fortran | 4 | 9.34080 | 8.29220 | 1.126x |
| Windows | float32 | strided | 1 | 4.41750 | 5.31370 | 0.831x |
| Windows | float32 | strided | 4 | 4.50043 | 5.29570 | 0.850x |
| Windows | float32 | all-weak | 1 | 7.59180 | 2.95763 | 2.567x |
| Windows | float32 | all-weak | 4 | 7.37415 | 1.98638 | 3.712x |
| Windows | float32 | no-seeds | 1 | 8.61220 | 3.57853 | 2.407x |
| Windows | float32 | no-seeds | 4 | 8.65770 | 2.18967 | 3.954x |
| Windows | float32 | equal | 1 | 8.70260 | 4.81730 | 1.807x |
| Windows | float32 | equal | 4 | 8.65670 | 5.18485 | 1.670x |
| Windows | float32 | threshold-fields | 1 | 9.07450 | 9.63570 | 0.942x |
| Windows | float32 | threshold-fields | 4 | 9.00990 | 5.95405 | 1.513x |
| Windows | float32 | snake | 1 | 7.07275 | 3.11852 | 2.268x |
| Windows | float32 | snake | 4 | 7.15665 | 3.48660 | 2.053x |
| Windows | float32 | checkerboard | 1 | 8.99280 | 2.12842 | 4.225x |
| Windows | float32 | checkerboard | 4 | 8.95940 | 3.24450 | 2.761x |
| Windows | float32 | tiny | 1 | 0.06268 | 0.05632 | 1.113x |
| Windows | float32 | tiny | 4 | 0.05529 | 0.05635 | 0.981x |
| Windows | float32 | nan-background | 1 | 7.08390 | 2.96840 | 2.386x |
| Windows | float32 | nan-background | 4 | 6.95920 | 1.96953 | 3.533x |
| Windows | float64 | random | 1 | 8.67310 | 9.44000 | 0.919x |
| Windows | float64 | random | 4 | 8.69950 | 5.56410 | 1.564x |
| Windows | float64 | fortran | 1 | 9.57780 | 12.09240 | 0.792x |
| Windows | float64 | fortran | 4 | 9.46220 | 8.15630 | 1.160x |
| Windows | float64 | strided | 1 | 4.57680 | 5.86320 | 0.781x |
| Windows | float64 | strided | 4 | 4.54667 | 5.83430 | 0.779x |
| Windows | float64 | all-weak | 1 | 7.52660 | 2.08368 | 3.612x |
| Windows | float64 | all-weak | 4 | 7.42415 | 1.15134 | 6.448x |
| Windows | float64 | no-seeds | 1 | 8.84450 | 2.75218 | 3.214x |
| Windows | float64 | no-seeds | 4 | 8.74160 | 1.38611 | 6.307x |
| Windows | float64 | equal | 1 | 9.09980 | 2.70440 | 3.365x |
| Windows | float64 | equal | 4 | 8.46250 | 1.32149 | 6.404x |
| Windows | float64 | threshold-fields | 1 | 9.67010 | 10.41030 | 0.929x |
| Windows | float64 | threshold-fields | 4 | 9.84190 | 6.42430 | 1.532x |
| Windows | float64 | snake | 1 | 7.06250 | 2.24568 | 3.145x |
| Windows | float64 | snake | 4 | 7.10760 | 2.85214 | 2.492x |
| Windows | float64 | checkerboard | 1 | 8.94130 | 1.33937 | 6.676x |
| Windows | float64 | checkerboard | 4 | 8.92350 | 2.39416 | 3.727x |
| Windows | float64 | tiny | 1 | 0.05535 | 0.05346 | 1.035x |
| Windows | float64 | tiny | 4 | 0.05353 | 0.05349 | 1.001x |
| Windows | float64 | nan-background | 1 | 6.93210 | 2.04840 | 3.384x |
| Windows | float64 | nan-background | 4 | 6.87675 | 1.09783 | 6.264x |
