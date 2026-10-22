# Histogram-matching input layout tuning

Retained source: `bb0d364f4931b23977bd9b418eca333f4546c602`; product source fingerprint `34e8a3f35f168464771e72cced4468bdffcbe92057b3829341259af137e1bb87`. Previous build: `a0740f9`. Reference: scikit-image 0.26.0.

The input path retains aligned linear spatial/channel views, including planar, negative and zero strides. AVX2 key extraction accepts nonnegative spatial strides; negative spatial strides use scalar key extraction. Nonlinear layouts copy into contiguous storage. Output allocation, numerical mapping, warning behavior, public Python signatures and the global optional worker limit are preserved. C++ adds a stride-aware entry point while retaining the existing contiguous entry point.

All 5,652 scoped matching/API/metadata tests passed in each SIMD mode on Windows/NumPy 2.5.4 and Linux/NumPy 1.24.4. The 54 added cases check exact parity and copy policy for planar, reversed, broadcast and nonlinear views. Native address/undefined-behavior checks passed with AVX2, dynamically reduced OpenMP teams and portable builds without OpenMP/SIMD. Native tests include stride overflow and singleton dimensions with otherwise extreme unused strides.

Full source [CI 39746070152](https://github.com/bipxx/scikit-imagexx/actions/runs/39746070152) failed: Windows/Linux jobs passed, but the minimum-NumPy job reported six CIEDE2000 parity failures. The color-distance source and build settings were identical to the previous build. Both exact Linux packages passed all 1,440 weighted-distance cases locally. A fresh [run of the previous source](https://github.com/bipxx/scikit-imagexx/actions/runs/39741060736/attempts/2) reproduced the same six failures. The subsequent [runtime-math correction](distance-provider.md) identifies and addresses NumPy SVML discontinuity behavior; its actual AVX-512 confirmation passes. These scoped histogram measurements do not establish whole-library completion.

The fixed 24-case corpus uses seed 904, source 512×513×3 and reference 515×511×3 before layout transformations, seven shuffled warmed repetitions, 15 ms target batches and 20 ms cooldowns. Public calls include preparation, allocation and destruction. Original thread pools use one; C++ uses global one/four-worker limits. No local builds or tests overlapped measurement. Exact saved baseline Python/binary hashes and ISA dispatch are checked before timing.

| Run | Workers | Previous/current geometric mean | Original/current | Faster than previous | Faster than original |
|---|---:|---:|---:|---:|---:|
| Primary Windows | 1 | 1.119× | 2.076× | 18/24 | 18/24 |
| Independent Windows | 1 | 1.131× | 2.121× | 18/24 | 19/24 |
| Independent Windows | 4 | 1.066× | 2.394× | 16/24 | 22/24 |
| Independent Linux/NumPy 1.24 | 1 | 1.119× | 3.484× | 13/24 | 23/24 |
| Independent Linux/NumPy 1.24 | 4 | 1.059× | 3.813× | 13/24 | 23/24 |

The primary paired ratio supports keeping this change. All adverse observations remain in the raw reports. The primary int16-color and float32-strided cases regressed to 0.829× and 0.892× previous/current. Independent measurements below show whether these observations repeated. Some cases still trail original scikit-image; this is a scoped improvement, and full-library coverage remains incomplete. Ratios from different runs are not multiplied or inferred from absolute times.

Raw reports are in `match-layout-tuning.json` and separate platform JSON files. Exact measured wheels, packages, source archive and validation logs are retained under `build/checkpoints/match-layout-validation`. The previous baseline checkpoints and historical whole-library reports remain preserved.

| NumPy | Input | Layout | Workers | Original ms | Current ms | Original/current | Previous/current |
|---|---|---|---:|---:|---:|---:|---:|
| 2.5.4 | uint8 | mono | 1 | 1.1679 | 0.5735 | 2.037× | 1.016× |
| 2.5.4 | uint8 | mono | 4 | 1.1757 | 0.5371 | 2.189× | 0.990× |
| 2.5.4 | uint8 | color | 1 | 4.5693 | 3.7020 | 1.234× | 1.044× |
| 2.5.4 | uint8 | color | 4 | 4.5675 | 1.9831 | 2.303× | 1.098× |
| 2.5.4 | uint8 | planar | 1 | 4.7419 | 2.1926 | 2.163× | 3.036× |
| 2.5.4 | uint8 | planar | 4 | 4.6462 | 2.5721 | 1.806× | 1.852× |
| 2.5.4 | uint8 | strided | 1 | 2.6825 | 3.1516 | 0.851× | 1.099× |
| 2.5.4 | uint8 | strided | 4 | 2.4954 | 2.8898 | 0.864× | 0.911× |
| 2.5.4 | uint16 | mono | 1 | 2.8752 | 2.2076 | 1.302× | 1.219× |
| 2.5.4 | uint16 | mono | 4 | 3.0869 | 2.5361 | 1.217× | 1.107× |
| 2.5.4 | uint16 | color | 1 | 10.8661 | 8.6740 | 1.253× | 1.053× |
| 2.5.4 | uint16 | color | 4 | 11.1791 | 7.6955 | 1.453× | 1.035× |
| 2.5.4 | uint16 | planar | 1 | 10.5016 | 8.7080 | 1.206× | 1.568× |
| 2.5.4 | uint16 | planar | 4 | 11.6307 | 8.7477 | 1.330× | 1.150× |
| 2.5.4 | uint16 | strided | 1 | 9.2084 | 7.5566 | 1.219× | 1.083× |
| 2.5.4 | uint16 | strided | 4 | 9.1530 | 7.8121 | 1.172× | 0.993× |
| 2.5.4 | int16 | mono | 1 | 27.4866 | 0.7413 | 37.078× | 1.076× |
| 2.5.4 | int16 | mono | 4 | 24.5320 | 1.0662 | 23.008× | 0.853× |
| 2.5.4 | int16 | color | 1 | 84.5467 | 4.8463 | 17.446× | 1.141× |
| 2.5.4 | int16 | color | 4 | 74.9702 | 3.3941 | 22.088× | 1.073× |
| 2.5.4 | int16 | planar | 1 | 84.4229 | 3.8114 | 22.150× | 2.079× |
| 2.5.4 | int16 | planar | 4 | 81.6954 | 3.1263 | 26.131× | 1.608× |
| 2.5.4 | int16 | strided | 1 | 43.7836 | 4.2765 | 10.238× | 0.939× |
| 2.5.4 | int16 | strided | 4 | 40.4318 | 4.0724 | 9.928× | 1.051× |
| 2.5.4 | float32 | mono | 1 | 14.1458 | 11.2997 | 1.252× | 0.916× |
| 2.5.4 | float32 | mono | 4 | 13.5365 | 8.6088 | 1.572× | 1.052× |
| 2.5.4 | float32 | color | 1 | 67.9251 | 44.1321 | 1.539× | 0.959× |
| 2.5.4 | float32 | color | 4 | 62.6710 | 35.3786 | 1.771× | 1.107× |
| 2.5.4 | float32 | planar | 1 | 61.0663 | 44.7667 | 1.364× | 1.086× |
| 2.5.4 | float32 | planar | 4 | 57.0720 | 36.9469 | 1.545× | 1.018× |
| 2.5.4 | float32 | strided | 1 | 33.6660 | 24.7317 | 1.361× | 1.045× |
| 2.5.4 | float32 | strided | 4 | 26.2428 | 21.1197 | 1.243× | 0.972× |
| 2.5.4 | float64 | mono | 1 | 16.9596 | 18.2346 | 0.930× | 1.081× |
| 2.5.4 | float64 | mono | 4 | 17.7867 | 13.3692 | 1.330× | 1.015× |
| 2.5.4 | float64 | color | 1 | 62.1716 | 74.1190 | 0.839× | 0.888× |
| 2.5.4 | float64 | color | 4 | 74.1315 | 45.9171 | 1.614× | 1.016× |
| 2.5.4 | float64 | planar | 1 | 47.6929 | 51.5049 | 0.926× | 1.046× |
| 2.5.4 | float64 | planar | 4 | 46.0495 | 34.6604 | 1.329× | 1.036× |
| 2.5.4 | float64 | strided | 1 | 25.1552 | 25.6691 | 0.980× | 1.003× |
| 2.5.4 | float64 | strided | 4 | 21.4561 | 26.4787 | 0.810× | 1.085× |
| 2.5.4 | float32 | quantized-mono | 1 | 4.9165 | 2.5916 | 1.897× | 1.019× |
| 2.5.4 | float32 | quantized-mono | 4 | 4.9061 | 2.1533 | 2.278× | 1.001× |
| 2.5.4 | float32 | quantized-color | 1 | 16.4649 | 8.3458 | 1.973× | 1.056× |
| 2.5.4 | float32 | quantized-color | 4 | 15.4551 | 6.8873 | 2.244× | 0.987× |
| 2.5.4 | float64 | quantized-mono | 1 | 6.3307 | 2.9460 | 2.149× | 0.998× |
| 2.5.4 | float64 | quantized-mono | 4 | 6.1460 | 2.3170 | 2.653× | 0.989× |
| 2.5.4 | float64 | quantized-color | 1 | 21.3889 | 10.3155 | 2.073× | 0.957× |
| 2.5.4 | float64 | quantized-color | 4 | 22.8106 | 8.6897 | 2.625× | 0.968× |
| 1.24.4 | uint8 | mono | 1 | 0.9766 | 0.5084 | 1.921× | 0.990× |
| 1.24.4 | uint8 | mono | 4 | 1.0175 | 0.6605 | 1.541× | 0.956× |
| 1.24.4 | uint8 | color | 1 | 3.4297 | 3.2924 | 1.042× | 0.993× |
| 1.24.4 | uint8 | color | 4 | 3.3441 | 2.1957 | 1.523× | 0.997× |
| 1.24.4 | uint8 | planar | 1 | 3.3160 | 1.7507 | 1.894× | 2.798× |
| 1.24.4 | uint8 | planar | 4 | 3.7799 | 1.9798 | 1.909× | 1.949× |
| 1.24.4 | uint8 | strided | 1 | 2.0539 | 2.5848 | 0.795× | 0.983× |
| 1.24.4 | uint8 | strided | 4 | 1.9545 | 2.5045 | 0.780× | 1.020× |
| 1.24.4 | uint16 | mono | 1 | 2.4709 | 1.8271 | 1.352× | 0.999× |
| 1.24.4 | uint16 | mono | 4 | 2.5663 | 1.9320 | 1.328× | 1.003× |
| 1.24.4 | uint16 | color | 1 | 8.4839 | 6.8969 | 1.230× | 1.003× |
| 1.24.4 | uint16 | color | 4 | 9.4825 | 6.1685 | 1.537× | 0.962× |
| 1.24.4 | uint16 | planar | 1 | 9.3725 | 6.4289 | 1.458× | 1.581× |
| 1.24.4 | uint16 | planar | 4 | 7.8035 | 5.6092 | 1.391× | 1.332× |
| 1.24.4 | uint16 | strided | 1 | 6.5020 | 5.8560 | 1.110× | 0.991× |
| 1.24.4 | uint16 | strided | 4 | 6.5598 | 5.9530 | 1.102× | 1.000× |
| 1.24.4 | int16 | mono | 1 | 24.8650 | 0.7300 | 34.059× | 0.913× |
| 1.24.4 | int16 | mono | 4 | 24.9216 | 0.8432 | 29.555× | 0.948× |
| 1.24.4 | int16 | color | 1 | 75.9019 | 4.4421 | 17.087× | 0.981× |
| 1.24.4 | int16 | color | 4 | 71.6147 | 2.7112 | 26.414× | 1.000× |
| 1.24.4 | int16 | planar | 1 | 71.5250 | 2.2614 | 31.628× | 2.605× |
| 1.24.4 | int16 | planar | 4 | 71.2119 | 2.2188 | 32.095× | 1.967× |
| 1.24.4 | int16 | strided | 1 | 39.9255 | 3.2159 | 12.415× | 1.046× |
| 1.24.4 | int16 | strided | 4 | 39.3730 | 3.2137 | 12.252× | 1.030× |
| 1.24.4 | float32 | mono | 1 | 39.5696 | 11.6603 | 3.394× | 0.777× |
| 1.24.4 | float32 | mono | 4 | 39.3678 | 10.4922 | 3.752× | 0.823× |
| 1.24.4 | float32 | color | 1 | 126.0679 | 39.0359 | 3.230× | 1.009× |
| 1.24.4 | float32 | color | 4 | 125.1003 | 34.0886 | 3.670× | 1.018× |
| 1.24.4 | float32 | planar | 1 | 150.7941 | 52.8671 | 2.852× | 1.178× |
| 1.24.4 | float32 | planar | 4 | 121.5613 | 33.8465 | 3.592× | 1.103× |
| 1.24.4 | float32 | strided | 1 | 54.3185 | 15.6228 | 3.477× | 0.996× |
| 1.24.4 | float32 | strided | 4 | 61.1628 | 17.7628 | 3.443× | 1.038× |
| 1.24.4 | float64 | mono | 1 | 43.0709 | 20.5732 | 2.094× | 1.015× |
| 1.24.4 | float64 | mono | 4 | 42.5802 | 13.8349 | 3.078× | 1.038× |
| 1.24.4 | float64 | color | 1 | 131.8769 | 68.4551 | 1.926× | 1.022× |
| 1.24.4 | float64 | color | 4 | 138.2339 | 51.8317 | 2.667× | 1.030× |
| 1.24.4 | float64 | planar | 1 | 134.3687 | 63.4519 | 2.118× | 1.300× |
| 1.24.4 | float64 | planar | 4 | 131.8992 | 52.3681 | 2.519× | 1.105× |
| 1.24.4 | float64 | strided | 1 | 70.1502 | 35.7403 | 1.963× | 0.955× |
| 1.24.4 | float64 | strided | 4 | 61.8267 | 28.6813 | 2.156× | 0.959× |
| 1.24.4 | float32 | quantized-mono | 1 | 25.2681 | 3.0879 | 8.183× | 1.027× |
| 1.24.4 | float32 | quantized-mono | 4 | 25.0148 | 2.9751 | 8.408× | 0.940× |
| 1.24.4 | float32 | quantized-color | 1 | 75.5929 | 11.1895 | 6.756× | 1.028× |
| 1.24.4 | float32 | quantized-color | 4 | 79.6336 | 9.8345 | 8.097× | 1.028× |
| 1.24.4 | float64 | quantized-mono | 1 | 25.2476 | 2.9904 | 8.443× | 1.153× |
| 1.24.4 | float64 | quantized-mono | 4 | 25.1999 | 3.2027 | 7.868× | 0.902× |
| 1.24.4 | float64 | quantized-color | 1 | 84.0593 | 12.9892 | 6.471× | 0.993× |
| 1.24.4 | float64 | quantized-color | 4 | 81.9678 | 12.5222 | 6.546× | 0.885× |
