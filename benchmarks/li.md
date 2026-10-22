# Li threshold initial performance checkpoint

Source: `a686283`; original scikit-image 0.26.0. These are initial measurements, with known regressions retained. Each runner built and installed its exact archived wheel, passed all 2,650 scoped Li/API parity cases, then timed allocation-inclusive calls with fallback disabled. Separate isolated GitHub runners used fixed seed 817, size 512 by 513, seven randomized paired samples, 15 ms target batches and 20 ms cooldown. Reference thread pools use one thread.

[Exact wheels, preserved packages and raw reports](https://github.com/bipxx/scikit-imagexx/actions/runs/39761807459) are also retained locally in `build/checkpoints/li-initial`. The JSON files retain every sample and binary/source hashes. Continuous floating cases use tolerance 0.001 because the reference can cycle at its default tolerance; tests separately compare bounded traces of genuine cycles. Callable guesses/callbacks remain explicit fallback.

Local validation: Windows 43,798 passed, two skipped; Windows and Linux NumPy 1.24 scoped suites each 2,650 passed with SIMD enabled and disabled. Native ASan/UBSan, dynamic OpenMP teams and portable no-SIMD/no-OpenMP builds each passed 2/2 checks. Full CI run 39761597770 passed Li and minimum-NumPy checks but failed eight existing CIEDE2000 antipodal layout cases on Linux; whole-library CI is not green.

| Host | One-worker geometric mean | Slowest one-worker case |
|---|---:|---|
| Linux | 5.382x | float64 constant: 0.429x |
| Windows | 4.069x | float64 numeric-guess: 0.526x |

Constant floating images are a measured bottleneck: the native preparation scan does more work than the reference early return. Four-worker calls can also regress on small/memory-bound cases. These initial results establish the fixed corpus for subsequent tuning; they do not establish that every call is faster.

| Host | Dtype | Case | Workers | Original ms | C++ ms | Original / C++ |
|---|---|---|---:|---:|---:|---:|
| Linux | uint8 | quantized | 1 | 0.84083 | 0.07987 | 10.527x |
| Linux | uint8 | quantized | 4 | 0.83557 | 0.06946 | 12.030x |
| Linux | uint8 | fortran | 1 | 1.05540 | 0.23380 | 4.514x |
| Linux | uint8 | fortran | 4 | 1.06366 | 0.24318 | 4.374x |
| Linux | uint8 | strided | 1 | 0.59858 | 0.08222 | 7.280x |
| Linux | uint8 | strided | 4 | 0.59047 | 0.08246 | 7.161x |
| Linux | uint8 | constant | 1 | 0.12424 | 0.07665 | 1.621x |
| Linux | uint8 | constant | 4 | 0.12364 | 0.06752 | 1.831x |
| Linux | uint8 | tiny | 1 | 0.06444 | 0.00590 | 10.922x |
| Linux | uint8 | tiny | 4 | 0.06285 | 0.00574 | 10.942x |
| Linux | uint8 | explicit-tolerance | 1 | 0.83428 | 0.08012 | 10.413x |
| Linux | uint8 | explicit-tolerance | 4 | 0.83811 | 0.06997 | 11.979x |
| Linux | uint8 | numeric-guess | 1 | 0.62549 | 0.08327 | 7.511x |
| Linux | uint8 | numeric-guess | 4 | 0.63362 | 0.07512 | 8.435x |
| Linux | float32 | quantized | 1 | 25.54376 | 2.63824 | 9.682x |
| Linux | float32 | quantized | 4 | 25.55784 | 2.66823 | 9.579x |
| Linux | float32 | fortran | 1 | 26.94045 | 3.28403 | 8.203x |
| Linux | float32 | fortran | 4 | 26.85463 | 3.36135 | 7.989x |
| Linux | float32 | strided | 1 | 12.10358 | 1.35940 | 8.904x |
| Linux | float32 | strided | 4 | 12.15250 | 1.34897 | 9.009x |
| Linux | float32 | constant | 1 | 0.15817 | 0.24922 | 0.635x |
| Linux | float32 | constant | 4 | 0.15689 | 0.24929 | 0.629x |
| Linux | float32 | tiny | 1 | 0.04629 | 0.00850 | 5.445x |
| Linux | float32 | tiny | 4 | 0.04638 | 0.00867 | 5.351x |
| Linux | float32 | explicit-tolerance | 1 | 19.65215 | 1.13378 | 17.333x |
| Linux | float32 | explicit-tolerance | 4 | 19.66683 | 1.66242 | 11.830x |
| Linux | float32 | numeric-guess | 1 | 11.06578 | 2.00835 | 5.510x |
| Linux | float32 | numeric-guess | 4 | 11.04029 | 1.72192 | 6.412x |
| Linux | float32 | continuous | 1 | 34.72916 | 1.74876 | 19.859x |
| Linux | float32 | continuous | 4 | 34.84192 | 2.58060 | 13.501x |
| Linux | float32 | nonfinite | 1 | 24.23396 | 2.72974 | 8.878x |
| Linux | float32 | nonfinite | 4 | 24.23987 | 2.72881 | 8.883x |
| Linux | float64 | quantized | 1 | 29.80904 | 7.03700 | 4.236x |
| Linux | float64 | quantized | 4 | 29.86760 | 6.21802 | 4.803x |
| Linux | float64 | fortran | 1 | 31.64510 | 8.26217 | 3.830x |
| Linux | float64 | fortran | 4 | 31.48277 | 8.01251 | 3.929x |
| Linux | float64 | strided | 1 | 14.68745 | 2.65095 | 5.540x |
| Linux | float64 | strided | 4 | 14.64454 | 2.62515 | 5.579x |
| Linux | float64 | constant | 1 | 0.21226 | 0.49497 | 0.429x |
| Linux | float64 | constant | 4 | 0.21466 | 0.49448 | 0.434x |
| Linux | float64 | tiny | 1 | 0.06221 | 0.00933 | 6.666x |
| Linux | float64 | tiny | 4 | 0.06173 | 0.00958 | 6.445x |
| Linux | float64 | explicit-tolerance | 1 | 21.03550 | 4.40758 | 4.773x |
| Linux | float64 | explicit-tolerance | 4 | 21.14520 | 4.70773 | 4.492x |
| Linux | float64 | numeric-guess | 1 | 15.66874 | 5.95094 | 2.633x |
| Linux | float64 | numeric-guess | 4 | 15.59262 | 5.21855 | 2.988x |
| Linux | float64 | continuous | 1 | 38.74178 | 5.56704 | 6.959x |
| Linux | float64 | continuous | 4 | 38.76102 | 5.81831 | 6.662x |
| Linux | float64 | nonfinite | 1 | 28.68217 | 6.76185 | 4.242x |
| Linux | float64 | nonfinite | 4 | 28.68408 | 6.81355 | 4.210x |
| Windows | uint8 | quantized | 1 | 2.01587 | 0.09896 | 20.371x |
| Windows | uint8 | quantized | 4 | 1.99253 | 0.10877 | 18.318x |
| Windows | uint8 | fortran | 1 | 2.73232 | 0.47795 | 5.717x |
| Windows | uint8 | fortran | 4 | 2.73470 | 0.48621 | 5.625x |
| Windows | uint8 | strided | 1 | 1.31291 | 0.11553 | 11.365x |
| Windows | uint8 | strided | 4 | 1.32183 | 0.11240 | 11.760x |
| Windows | uint8 | constant | 1 | 0.22610 | 0.10343 | 2.186x |
| Windows | uint8 | constant | 4 | 0.23032 | 0.10026 | 2.297x |
| Windows | uint8 | tiny | 1 | 0.11748 | 0.00843 | 13.933x |
| Windows | uint8 | tiny | 4 | 0.11504 | 0.00843 | 13.646x |
| Windows | uint8 | explicit-tolerance | 1 | 2.02863 | 0.09824 | 20.649x |
| Windows | uint8 | explicit-tolerance | 4 | 2.05682 | 0.10848 | 18.961x |
| Windows | uint8 | numeric-guess | 1 | 1.59529 | 0.10548 | 15.125x |
| Windows | uint8 | numeric-guess | 4 | 1.58956 | 0.12241 | 12.986x |
| Windows | float32 | quantized | 1 | 25.21360 | 12.01330 | 2.099x |
| Windows | float32 | quantized | 4 | 25.17620 | 7.21800 | 3.488x |
| Windows | float32 | fortran | 1 | 26.16700 | 12.87060 | 2.033x |
| Windows | float32 | fortran | 4 | 26.07630 | 8.11320 | 3.214x |
| Windows | float32 | strided | 1 | 11.48950 | 6.22020 | 1.847x |
| Windows | float32 | strided | 4 | 11.43920 | 6.21440 | 1.841x |
| Windows | float32 | constant | 1 | 0.42994 | 0.05958 | 7.217x |
| Windows | float32 | constant | 4 | 0.43388 | 0.06002 | 7.229x |
| Windows | float32 | tiny | 1 | 0.08146 | 0.01444 | 5.643x |
| Windows | float32 | tiny | 4 | 0.08126 | 0.01391 | 5.841x |
| Windows | float32 | explicit-tolerance | 1 | 23.27380 | 2.63738 | 8.825x |
| Windows | float32 | explicit-tolerance | 4 | 23.28880 | 3.17238 | 7.341x |
| Windows | float32 | numeric-guess | 1 | 8.35760 | 10.59490 | 0.789x |
| Windows | float32 | numeric-guess | 4 | 8.39450 | 5.55860 | 1.510x |
| Windows | float32 | continuous | 1 | 41.29170 | 3.93440 | 10.495x |
| Windows | float32 | continuous | 4 | 41.34440 | 4.71155 | 8.775x |
| Windows | float32 | nonfinite | 1 | 24.10490 | 12.09550 | 1.993x |
| Windows | float32 | nonfinite | 4 | 24.09870 | 12.13820 | 1.985x |
| Windows | float64 | quantized | 1 | 29.47820 | 22.02000 | 1.339x |
| Windows | float64 | quantized | 4 | 29.53940 | 11.91270 | 2.480x |
| Windows | float64 | fortran | 1 | 31.20370 | 24.20490 | 1.289x |
| Windows | float64 | fortran | 4 | 31.24990 | 14.04980 | 2.224x |
| Windows | float64 | strided | 1 | 14.54430 | 11.50510 | 1.264x |
| Windows | float64 | strided | 4 | 14.57240 | 11.57800 | 1.259x |
| Windows | float64 | constant | 1 | 0.72853 | 0.10243 | 7.112x |
| Windows | float64 | constant | 4 | 0.71885 | 0.10364 | 6.936x |
| Windows | float64 | tiny | 1 | 0.10193 | 0.01519 | 6.708x |
| Windows | float64 | tiny | 4 | 0.10080 | 0.01501 | 6.716x |
| Windows | float64 | explicit-tolerance | 1 | 27.30370 | 4.70500 | 5.803x |
| Windows | float64 | explicit-tolerance | 4 | 27.24290 | 4.68047 | 5.821x |
| Windows | float64 | numeric-guess | 1 | 10.39030 | 19.74770 | 0.526x |
| Windows | float64 | numeric-guess | 4 | 10.37790 | 9.79180 | 1.060x |
| Windows | float64 | continuous | 1 | 50.99640 | 7.22735 | 7.056x |
| Windows | float64 | continuous | 4 | 50.70470 | 7.25780 | 6.986x |
| Windows | float64 | nonfinite | 1 | 28.52720 | 21.77980 | 1.310x |
| Windows | float64 | nonfinite | 4 | 28.56120 | 21.70030 | 1.316x |
