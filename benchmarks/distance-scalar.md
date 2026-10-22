# Scalar color-distance promotion

Measured source: `5afe25c1f98f730c21913dae7eb58ce40c96f2e3`; source fingerprint `43b7d7419ceba969254af6ad68742c7f4231d199ecc92138641edfc9c4b5a852`.

Native CIE76/CIE94 versus scikit-image 0.26.0. Each environment has 32 fixed comparisons: float32/float64, one scalar color or a 256×256 image, default CIE76 and three CIE94 weight configurations, with one/four global workers. Timing includes Python dispatch, input preparation, allocation and output destruction. Seven shuffled warmed repetitions use 15 ms target batches and 20 ms cooldowns; all reference thread pools are limited to one. Builds/tests were stopped during timing. These scoped results do not establish whole-library performance.

The NumPy 1.24 scalar paths preserve float32 differences and chroma before promoted double arithmetic. The CIE94 path distinguishes NumPy float32 weight products from Python scalar products. A runtime scalar-power probe selects legacy promotion. Arrays retain their SIMD kernels and optional global worker limit. The public Python signatures are unchanged.

Validation: 2,066 scalar-promotion/metadata tests passed on NumPy 1.24.4 in each SIMD mode. On Windows/NumPy 2.5.4, 4,784 scoped distance/metadata tests passed in each mode (the final SIMD build was rechecked after removing a redundant alignment pass). Portable C++ tests passed with address/undefined-behavior sanitizers, including independent double-precision golden values and signed strides. The previous 16 scalar metadata failures and six existing scalar distance failures are fixed. Eight separate NumPy 1.24 CIEDE2000 failures remain, reproduced with the preserved pre-change wheel; failing logs and unchanged oracles are retained.

Source CI: [39731363740](https://github.com/bipxx/scikit-imagexx/actions/runs/39731363740) (running when this report was published).

| Environment | One-thread cases faster | Geometric mean original/C++ |
|---|---:|---:|
| Linux-6.18.54.1-microsoft-standard-WSL2-x86_64-with-glibc2.39, NumPy 1.24.4 | 15/16 | 2.961× |
| Windows-11-10.0.26200-SP0, NumPy 2.5.4 | 14/16 | 3.226× |

CIE76 scalar calls remain slower in several cases; Python preparation/dispatch is a remaining bottleneck. All adverse rows are included below. Ratios from the two environments are independent and must not be interpreted as a before/after comparison. Exact measured wheels/packages and validation logs are preserved locally in `build/checkpoints/scalar-distance-validation`.

| NumPy | API | Input | Layout | Options | Workers | Original µs | C++ µs | Original/C++ |
|---|---|---|---|---|---:|---:|---:|---:|
| 1.24.4 | color.deltaE_cie76 | float32 | scalar | default | 1 | 13.531 | 7.625 | 1.775× |
| 1.24.4 | color.deltaE_cie76 | float32 | scalar | default | 4 | 12.841 | 7.801 | 1.646× |
| 1.24.4 | color.deltaE_ciede94 | float32 | scalar | default | 1 | 57.124 | 20.346 | 2.808× |
| 1.24.4 | color.deltaE_ciede94 | float32 | scalar | default | 4 | 52.877 | 21.262 | 2.487× |
| 1.24.4 | color.deltaE_ciede94 | float32 | scalar | kL=2 (int), k1=0.048 (float), k2=0.014 (float) | 1 | 51.571 | 23.131 | 2.229× |
| 1.24.4 | color.deltaE_ciede94 | float32 | scalar | kL=2 (int), k1=0.048 (float), k2=0.014 (float) | 4 | 57.742 | 24.104 | 2.396× |
| 1.24.4 | color.deltaE_ciede94 | float32 | scalar | k1=0.731 (float32) | 1 | 73.379 | 21.205 | 3.460× |
| 1.24.4 | color.deltaE_ciede94 | float32 | scalar | k1=0.731 (float32) | 4 | 51.000 | 24.014 | 2.124× |
| 1.24.4 | color.deltaE_cie76 | float32 | array | default | 1 | 209.003 | 79.750 | 2.621× |
| 1.24.4 | color.deltaE_cie76 | float32 | array | default | 4 | 223.486 | 70.627 | 3.164× |
| 1.24.4 | color.deltaE_ciede94 | float32 | array | default | 1 | 8418.828 | 2320.959 | 3.627× |
| 1.24.4 | color.deltaE_ciede94 | float32 | array | default | 4 | 8262.344 | 2498.494 | 3.307× |
| 1.24.4 | color.deltaE_ciede94 | float32 | array | kL=2 (int), k1=0.048 (float), k2=0.014 (float) | 1 | 7814.887 | 2376.112 | 3.289× |
| 1.24.4 | color.deltaE_ciede94 | float32 | array | kL=2 (int), k1=0.048 (float), k2=0.014 (float) | 4 | 7744.250 | 2349.981 | 3.295× |
| 1.24.4 | color.deltaE_ciede94 | float32 | array | k1=0.731 (float32) | 1 | 7564.036 | 2337.680 | 3.236× |
| 1.24.4 | color.deltaE_ciede94 | float32 | array | k1=0.731 (float32) | 4 | 7744.671 | 2320.618 | 3.337× |
| 1.24.4 | color.deltaE_cie76 | float64 | scalar | default | 1 | 8.349 | 8.402 | 0.994× |
| 1.24.4 | color.deltaE_cie76 | float64 | scalar | default | 4 | 7.683 | 6.842 | 1.123× |
| 1.24.4 | color.deltaE_ciede94 | float64 | scalar | default | 1 | 38.982 | 12.759 | 3.055× |
| 1.24.4 | color.deltaE_ciede94 | float64 | scalar | default | 4 | 38.990 | 17.256 | 2.259× |
| 1.24.4 | color.deltaE_ciede94 | float64 | scalar | kL=2 (int), k1=0.048 (float), k2=0.014 (float) | 1 | 39.557 | 12.480 | 3.170× |
| 1.24.4 | color.deltaE_ciede94 | float64 | scalar | kL=2 (int), k1=0.048 (float), k2=0.014 (float) | 4 | 36.971 | 14.109 | 2.620× |
| 1.24.4 | color.deltaE_ciede94 | float64 | scalar | k1=0.731 (float32) | 1 | 40.903 | 13.343 | 3.065× |
| 1.24.4 | color.deltaE_ciede94 | float64 | scalar | k1=0.731 (float32) | 4 | 42.510 | 15.960 | 2.663× |
| 1.24.4 | color.deltaE_cie76 | float64 | array | default | 1 | 440.771 | 179.087 | 2.461× |
| 1.24.4 | color.deltaE_cie76 | float64 | array | default | 4 | 542.707 | 146.381 | 3.707× |
| 1.24.4 | color.deltaE_ciede94 | float64 | array | default | 1 | 10949.064 | 2192.262 | 4.994× |
| 1.24.4 | color.deltaE_ciede94 | float64 | array | default | 4 | 10715.251 | 2029.334 | 5.280× |
| 1.24.4 | color.deltaE_ciede94 | float64 | array | kL=2 (int), k1=0.048 (float), k2=0.014 (float) | 1 | 9721.937 | 2064.943 | 4.708× |
| 1.24.4 | color.deltaE_ciede94 | float64 | array | kL=2 (int), k1=0.048 (float), k2=0.014 (float) | 4 | 9283.421 | 1984.170 | 4.679× |
| 1.24.4 | color.deltaE_ciede94 | float64 | array | k1=0.731 (float32) | 1 | 10396.268 | 1979.703 | 5.251× |
| 1.24.4 | color.deltaE_ciede94 | float64 | array | k1=0.731 (float32) | 4 | 10071.112 | 2172.313 | 4.636× |
| 2.5.4 | color.deltaE_cie76 | float32 | scalar | default | 1 | 8.520 | 8.952 | 0.952× |
| 2.5.4 | color.deltaE_cie76 | float32 | scalar | default | 4 | 8.139 | 8.019 | 1.015× |
| 2.5.4 | color.deltaE_ciede94 | float32 | scalar | default | 1 | 41.553 | 16.845 | 2.467× |
| 2.5.4 | color.deltaE_ciede94 | float32 | scalar | default | 4 | 48.569 | 20.057 | 2.421× |
| 2.5.4 | color.deltaE_ciede94 | float32 | scalar | kL=2 (int), k1=0.048 (float), k2=0.014 (float) | 1 | 53.220 | 18.916 | 2.813× |
| 2.5.4 | color.deltaE_ciede94 | float32 | scalar | kL=2 (int), k1=0.048 (float), k2=0.014 (float) | 4 | 46.787 | 22.226 | 2.105× |
| 2.5.4 | color.deltaE_ciede94 | float32 | scalar | k1=0.731 (float32) | 1 | 59.807 | 22.135 | 2.702× |
| 2.5.4 | color.deltaE_ciede94 | float32 | scalar | k1=0.731 (float32) | 4 | 62.700 | 29.616 | 2.117× |
| 2.5.4 | color.deltaE_cie76 | float32 | array | default | 1 | 242.219 | 80.410 | 3.012× |
| 2.5.4 | color.deltaE_cie76 | float32 | array | default | 4 | 300.190 | 96.910 | 3.098× |
| 2.5.4 | color.deltaE_ciede94 | float32 | array | default | 1 | 8226.800 | 1924.620 | 4.275× |
| 2.5.4 | color.deltaE_ciede94 | float32 | array | default | 4 | 7839.300 | 1592.500 | 4.923× |
| 2.5.4 | color.deltaE_ciede94 | float32 | array | kL=2 (int), k1=0.048 (float), k2=0.014 (float) | 1 | 9573.100 | 1767.263 | 5.417× |
| 2.5.4 | color.deltaE_ciede94 | float32 | array | kL=2 (int), k1=0.048 (float), k2=0.014 (float) | 4 | 9604.200 | 1786.820 | 5.375× |
| 2.5.4 | color.deltaE_ciede94 | float32 | array | k1=0.731 (float32) | 1 | 8898.800 | 1962.289 | 4.535× |
| 2.5.4 | color.deltaE_ciede94 | float32 | array | k1=0.731 (float32) | 4 | 9861.300 | 1908.771 | 5.166× |
| 2.5.4 | color.deltaE_cie76 | float64 | scalar | default | 1 | 9.870 | 12.218 | 0.808× |
| 2.5.4 | color.deltaE_cie76 | float64 | scalar | default | 4 | 10.159 | 12.418 | 0.818× |
| 2.5.4 | color.deltaE_ciede94 | float64 | scalar | default | 1 | 54.854 | 16.476 | 3.329× |
| 2.5.4 | color.deltaE_ciede94 | float64 | scalar | default | 4 | 49.897 | 15.696 | 3.179× |
| 2.5.4 | color.deltaE_ciede94 | float64 | scalar | kL=2 (int), k1=0.048 (float), k2=0.014 (float) | 1 | 48.150 | 30.573 | 1.575× |
| 2.5.4 | color.deltaE_ciede94 | float64 | scalar | kL=2 (int), k1=0.048 (float), k2=0.014 (float) | 4 | 68.302 | 22.018 | 3.102× |
| 2.5.4 | color.deltaE_ciede94 | float64 | scalar | k1=0.731 (float32) | 1 | 57.278 | 26.817 | 2.136× |
| 2.5.4 | color.deltaE_ciede94 | float64 | scalar | k1=0.731 (float32) | 4 | 62.609 | 24.458 | 2.560× |
| 2.5.4 | color.deltaE_cie76 | float64 | array | default | 1 | 799.030 | 198.272 | 4.030× |
| 2.5.4 | color.deltaE_cie76 | float64 | array | default | 4 | 1313.586 | 214.857 | 6.114× |
| 2.5.4 | color.deltaE_ciede94 | float64 | array | default | 1 | 11286.900 | 1220.260 | 9.250× |
| 2.5.4 | color.deltaE_ciede94 | float64 | array | default | 4 | 10572.400 | 1353.870 | 7.809× |
| 2.5.4 | color.deltaE_ciede94 | float64 | array | kL=2 (int), k1=0.048 (float), k2=0.014 (float) | 1 | 11662.400 | 1398.373 | 8.340× |
| 2.5.4 | color.deltaE_ciede94 | float64 | array | kL=2 (int), k1=0.048 (float), k2=0.014 (float) | 4 | 10304.800 | 1487.317 | 6.928× |
| 2.5.4 | color.deltaE_ciede94 | float64 | array | k1=0.731 (float32) | 1 | 12199.000 | 1407.183 | 8.669× |
| 2.5.4 | color.deltaE_ciede94 | float64 | array | k1=0.731 (float32) | 4 | 10694.100 | 1532.991 | 6.976× |
