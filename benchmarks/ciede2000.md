# CIEDE2000 mixed-precision parity and performance

Measured source: `f613d5d1fffec5124bdffb5e5065b754bf4f9c25`; source fingerprint `a861d7ba0b1c9c255b76a78cfec72815cb98630e3b8403003252351504f42993`.

NumPy 1.24 promotes a float32 chroma array at `c7 + 25**7`. Later chroma/hue stages therefore use double precision, while lightness and each in-place accumulation into the result retain float32 rounding. NumPy 2 keeps the large Python integer in float32. A dtype probe selects a shared native SIMD/scalar implementation with separate arithmetic types for those stages. Public Python signatures, float32 output, and the global optional worker limit are preserved. No upstream fallback performs image arithmetic.

The eight previous NumPy 1.24 failures (Sharma/reversed pairs, hue branches and zero chroma/hue weights, each with one/four workers) now pass. The new 196-case regression file initially produced 102 numerical failures and 94 passes against the exact preserved prior package. Its corrected broadcast fixtures separately test valid broadcasting and upstream axis-error parity. The earlier fixture-error log is retained.

All 4,980 scoped distance/metadata tests passed in each SIMD mode on NumPy 1.24.4 and Windows/NumPy 2.5.4. Windows C++ SIMD/forced-scalar tests passed, including independent discontinuity golden values across packets, tails and one/four workers. Portable C++ address/undefined-behavior sanitizer tests passed after the fixture was corrected to request four workers only when built with OpenMP; the initial unsupported-worker failure log remains archived. No numerical tolerances or nonfinite masks were relaxed. Minimum CI now includes the complete distance/metadata files and new regressions.

Source CI: [39732776516](https://github.com/bipxx/scikit-imagexx/actions/runs/39732776516) (running when published).

Each environment contains 40 comparisons against scikit-image 0.26.0: float32/float64 contiguous, planar, signed-stride and scalar inputs with default/non-default weights, plus zero weights, NumPy float32 weights and Sharma boundary controls. Image sizes are 512×513 before striding; the boundary control repeats all 34 reference pairs 65 times. Worker limits are one/four, with reference pools limited to one. Seven shuffled warmed repetitions use 15 ms target batches and 20 ms cooldowns. Public dispatch, input preparation, allocations and output destruction are included. No local builds or tests overlapped timing.

| Environment | One-thread faster | Geometric mean original/C++ | All worker rows faster |
|---|---:|---:|---:|
| Linux-6.18.54.1-microsoft-standard-WSL2-x86_64-with-glibc2.39, NumPy 1.24.4 | 20/20 | 3.287× | 40/40 |
| Windows-11-10.0.26200-SP0, NumPy 2.5.4 | 20/20 | 2.513× | 40/40 |

These independent platform/version ratios are comparisons against original scikit-image, not a before/after C++ comparison. All rows and raw samples are retained. They do not establish full-library speed or complete API parity. The separate 21-report whole-library checkpoint remains historical. Exact measured packages/wheels, source and validation evidence are preserved in `build/checkpoints/ciede2000-validation`.

| NumPy | Input | Layout | Options | Workers | Original µs | C++ µs | Original/C++ |
|---|---|---|---|---:|---:|---:|---:|
| 1.24.4 | float32 | contiguous | channel_axis=-1 (int) | 1 | 66425.275 | 24468.693 | 2.715× |
| 1.24.4 | float32 | contiguous | channel_axis=-1 (int) | 4 | 66003.590 | 6674.881 | 9.888× |
| 1.24.4 | float32 | contiguous | channel_axis=-1 (int), kL=2 (int), kC=0.5 (float) | 1 | 67216.209 | 23904.392 | 2.812× |
| 1.24.4 | float32 | contiguous | channel_axis=-1 (int), kL=2 (int), kC=0.5 (float) | 4 | 70292.141 | 7171.450 | 9.802× |
| 1.24.4 | float32 | channel-first | channel_axis=0 (int) | 1 | 70465.543 | 23612.122 | 2.984× |
| 1.24.4 | float32 | channel-first | channel_axis=0 (int) | 4 | 66065.201 | 6471.774 | 10.208× |
| 1.24.4 | float32 | channel-first | channel_axis=0 (int), kL=2 (int), kC=0.5 (float) | 1 | 68118.334 | 22281.452 | 3.057× |
| 1.24.4 | float32 | channel-first | channel_axis=0 (int), kL=2 (int), kC=0.5 (float) | 4 | 73128.418 | 7228.996 | 10.116× |
| 1.24.4 | float32 | strided | channel_axis=-1 (int) | 1 | 36856.448 | 14026.661 | 2.628× |
| 1.24.4 | float32 | strided | channel_axis=-1 (int) | 4 | 39410.330 | 14384.025 | 2.740× |
| 1.24.4 | float32 | strided | channel_axis=-1 (int), kL=2 (int), kC=0.5 (float) | 1 | 37405.656 | 13756.333 | 2.719× |
| 1.24.4 | float32 | strided | channel_axis=-1 (int), kL=2 (int), kC=0.5 (float) | 4 | 37659.435 | 14019.198 | 2.686× |
| 1.24.4 | float32 | scalar | channel_axis=-1 (int) | 1 | 104.938 | 22.897 | 4.583× |
| 1.24.4 | float32 | scalar | channel_axis=-1 (int) | 4 | 88.110 | 24.125 | 3.652× |
| 1.24.4 | float32 | scalar | channel_axis=-1 (int), kL=2 (int), kC=0.5 (float) | 1 | 80.603 | 21.043 | 3.830× |
| 1.24.4 | float32 | scalar | channel_axis=-1 (int), kL=2 (int), kC=0.5 (float) | 4 | 82.248 | 22.497 | 3.656× |
| 1.24.4 | float32 | weight-control | kC=0 (int) | 1 | 78880.298 | 24947.931 | 3.162× |
| 1.24.4 | float32 | weight-control | kC=0 (int) | 4 | 83824.266 | 6940.198 | 12.078× |
| 1.24.4 | float32 | weight-control | kH=0 (int) | 1 | 75497.778 | 24473.582 | 3.085× |
| 1.24.4 | float32 | weight-control | kH=0 (int) | 4 | 77017.748 | 6700.062 | 11.495× |
| 1.24.4 | float32 | weight-control | kL=2 (float32), kC=0.7 (float32) | 1 | 77598.274 | 24642.719 | 3.149× |
| 1.24.4 | float32 | weight-control | kL=2 (float32), kC=0.7 (float32) | 4 | 76409.388 | 6858.234 | 11.141× |
| 1.24.4 | float64 | contiguous | channel_axis=-1 (int) | 1 | 88554.295 | 22176.941 | 3.993× |
| 1.24.4 | float64 | contiguous | channel_axis=-1 (int) | 4 | 84616.264 | 6241.872 | 13.556× |
| 1.24.4 | float64 | contiguous | channel_axis=-1 (int), kL=2 (int), kC=0.5 (float) | 1 | 74005.494 | 22227.585 | 3.329× |
| 1.24.4 | float64 | contiguous | channel_axis=-1 (int), kL=2 (int), kC=0.5 (float) | 4 | 86642.467 | 6288.807 | 13.777× |
| 1.24.4 | float64 | channel-first | channel_axis=0 (int) | 1 | 77628.052 | 22512.514 | 3.448× |
| 1.24.4 | float64 | channel-first | channel_axis=0 (int) | 4 | 79728.136 | 6512.214 | 12.243× |
| 1.24.4 | float64 | channel-first | channel_axis=0 (int), kL=2 (int), kC=0.5 (float) | 1 | 72552.862 | 22026.964 | 3.294× |
| 1.24.4 | float64 | channel-first | channel_axis=0 (int), kL=2 (int), kC=0.5 (float) | 4 | 74792.100 | 6260.792 | 11.946× |
| 1.24.4 | float64 | strided | channel_axis=-1 (int) | 1 | 39300.413 | 12829.218 | 3.063× |
| 1.24.4 | float64 | strided | channel_axis=-1 (int) | 4 | 38184.008 | 12616.903 | 3.026× |
| 1.24.4 | float64 | strided | channel_axis=-1 (int), kL=2 (int), kC=0.5 (float) | 1 | 38020.554 | 13069.536 | 2.909× |
| 1.24.4 | float64 | strided | channel_axis=-1 (int), kL=2 (int), kC=0.5 (float) | 4 | 38378.495 | 13234.044 | 2.900× |
| 1.24.4 | float64 | scalar | channel_axis=-1 (int) | 1 | 75.225 | 13.983 | 5.380× |
| 1.24.4 | float64 | scalar | channel_axis=-1 (int) | 4 | 73.512 | 13.530 | 5.433× |
| 1.24.4 | float64 | scalar | channel_axis=-1 (int), kL=2 (int), kC=0.5 (float) | 1 | 76.778 | 13.941 | 5.507× |
| 1.24.4 | float64 | scalar | channel_axis=-1 (int), kL=2 (int), kC=0.5 (float) | 4 | 76.929 | 23.042 | 3.339× |
| 1.24.4 | float32 | Sharma-boundaries | default | 1 | 346.834 | 170.863 | 2.030× |
| 1.24.4 | float32 | Sharma-boundaries | default | 4 | 332.933 | 171.405 | 1.942× |
| 2.5.4 | float32 | contiguous | channel_axis=-1 (int) | 1 | 58967.200 | 31874.600 | 1.850× |
| 2.5.4 | float32 | contiguous | channel_axis=-1 (int) | 4 | 59910.400 | 11327.000 | 5.289× |
| 2.5.4 | float32 | contiguous | channel_axis=-1 (int), kL=2 (int), kC=0.5 (float) | 1 | 52145.000 | 35262.700 | 1.479× |
| 2.5.4 | float32 | contiguous | channel_axis=-1 (int), kL=2 (int), kC=0.5 (float) | 4 | 68576.300 | 13839.200 | 4.955× |
| 2.5.4 | float32 | channel-first | channel_axis=0 (int) | 1 | 71609.000 | 38479.500 | 1.861× |
| 2.5.4 | float32 | channel-first | channel_axis=0 (int) | 4 | 52205.400 | 11356.300 | 4.597× |
| 2.5.4 | float32 | channel-first | channel_axis=0 (int), kL=2 (int), kC=0.5 (float) | 1 | 66834.900 | 40957.700 | 1.632× |
| 2.5.4 | float32 | channel-first | channel_axis=0 (int), kL=2 (int), kC=0.5 (float) | 4 | 62672.300 | 11666.400 | 5.372× |
| 2.5.4 | float32 | strided | channel_axis=-1 (int) | 1 | 39982.300 | 21552.000 | 1.855× |
| 2.5.4 | float32 | strided | channel_axis=-1 (int) | 4 | 36524.600 | 21453.300 | 1.703× |
| 2.5.4 | float32 | strided | channel_axis=-1 (int), kL=2 (int), kC=0.5 (float) | 1 | 42535.600 | 23005.300 | 1.849× |
| 2.5.4 | float32 | strided | channel_axis=-1 (int), kL=2 (int), kC=0.5 (float) | 4 | 30952.200 | 21151.600 | 1.463× |
| 2.5.4 | float32 | scalar | channel_axis=-1 (int) | 1 | 142.206 | 31.820 | 4.469× |
| 2.5.4 | float32 | scalar | channel_axis=-1 (int) | 4 | 89.870 | 30.527 | 2.944× |
| 2.5.4 | float32 | scalar | channel_axis=-1 (int), kL=2 (int), kC=0.5 (float) | 1 | 181.322 | 28.675 | 6.323× |
| 2.5.4 | float32 | scalar | channel_axis=-1 (int), kL=2 (int), kC=0.5 (float) | 4 | 196.010 | 32.334 | 6.062× |
| 2.5.4 | float32 | weight-control | kC=0 (int) | 1 | 69230.600 | 40696.300 | 1.701× |
| 2.5.4 | float32 | weight-control | kC=0 (int) | 4 | 58405.000 | 12235.600 | 4.773× |
| 2.5.4 | float32 | weight-control | kH=0 (int) | 1 | 86509.100 | 43405.900 | 1.993× |
| 2.5.4 | float32 | weight-control | kH=0 (int) | 4 | 64754.300 | 12926.800 | 5.009× |
| 2.5.4 | float32 | weight-control | kL=2 (float32), kC=0.7 (float32) | 1 | 72398.400 | 41950.700 | 1.726× |
| 2.5.4 | float32 | weight-control | kL=2 (float32), kC=0.7 (float32) | 4 | 58047.400 | 12191.000 | 4.761× |
| 2.5.4 | float64 | contiguous | channel_axis=-1 (int) | 1 | 99151.100 | 30630.300 | 3.237× |
| 2.5.4 | float64 | contiguous | channel_axis=-1 (int) | 4 | 87811.600 | 11533.000 | 7.614× |
| 2.5.4 | float64 | contiguous | channel_axis=-1 (int), kL=2 (int), kC=0.5 (float) | 1 | 87273.800 | 32392.600 | 2.694× |
| 2.5.4 | float64 | contiguous | channel_axis=-1 (int), kL=2 (int), kC=0.5 (float) | 4 | 87634.200 | 11204.200 | 7.822× |
| 2.5.4 | float64 | channel-first | channel_axis=0 (int) | 1 | 82557.100 | 26896.600 | 3.069× |
| 2.5.4 | float64 | channel-first | channel_axis=0 (int) | 4 | 92976.000 | 11476.300 | 8.102× |
| 2.5.4 | float64 | channel-first | channel_axis=0 (int), kL=2 (int), kC=0.5 (float) | 1 | 76257.800 | 28403.300 | 2.685× |
| 2.5.4 | float64 | channel-first | channel_axis=0 (int), kL=2 (int), kC=0.5 (float) | 4 | 75265.500 | 13525.700 | 5.565× |
| 2.5.4 | float64 | strided | channel_axis=-1 (int) | 1 | 44725.300 | 18089.600 | 2.472× |
| 2.5.4 | float64 | strided | channel_axis=-1 (int) | 4 | 51933.500 | 18491.700 | 2.808× |
| 2.5.4 | float64 | strided | channel_axis=-1 (int), kL=2 (int), kC=0.5 (float) | 1 | 48005.600 | 16309.200 | 2.943× |
| 2.5.4 | float64 | strided | channel_axis=-1 (int), kL=2 (int), kC=0.5 (float) | 4 | 48638.900 | 20816.900 | 2.337× |
| 2.5.4 | float64 | scalar | channel_axis=-1 (int) | 1 | 168.047 | 34.173 | 4.918× |
| 2.5.4 | float64 | scalar | channel_axis=-1 (int) | 4 | 146.178 | 35.365 | 4.133× |
| 2.5.4 | float64 | scalar | channel_axis=-1 (int), kL=2 (int), kC=0.5 (float) | 1 | 125.828 | 25.566 | 4.922× |
| 2.5.4 | float64 | scalar | channel_axis=-1 (int), kL=2 (int), kC=0.5 (float) | 4 | 94.052 | 33.222 | 2.831× |
| 2.5.4 | float32 | Sharma-boundaries | default | 1 | 436.971 | 258.250 | 1.692× |
| 2.5.4 | float32 | Sharma-boundaries | default | 4 | 555.392 | 331.564 | 1.675× |
