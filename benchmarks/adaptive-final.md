# Adaptive threshold overflow-policy correction

Measured source: `7e4a197`; product fingerprint `31039773248a8b5e60bde95aa4ed83d1aa9689f82f5c2e19637823c4743d7c5b`. Reference: scikit-image 0.26.0. The [initial port and corpus](adaptive.md) remain archived unchanged.

The initial bounds guard omitted constant-border magnitude when checking the final offset subtraction. Extreme finite borders plus a large offset could overflow silently in C++, instead of reproducing NumPy warnings or its `raise` policy. The corrected guard delegates those exceptional calls explicitly. Safety limits are Python floats so comparisons do not themselves cast overflowing values to float32. Added 36 cases cover all three methods, two precisions, three error policies and worker limits one/four.

Windows adaptive/API validation passes 5,020 cases in each SIMD mode; Linux/NumPy 1.24 passes 2,958 adaptive cases in each mode. Full CI for the initial port [39775491426](https://github.com/bipxx/scikit-imagexx/actions/runs/39775491426) passes all three jobs, including both SIMD modes and normal/dynamic/portable native checks. Full CI for this correction [39777159350](https://github.com/bipxx/scikit-imagexx/actions/runs/39777159350) also passes all three jobs: Linux 47,409 tests and Windows 47,407 plus two skips in each SIMD mode, with normal/dynamic/portable native checks passing 2/2 on both OSes.

The unchanged 63-case corpus runs at one/four workers with seven randomized paired samples, 15 ms target batches and 20 ms cooldown. Both packages pass parity with fallback disabled before timing. [Exact initial/current packages and all samples](https://github.com/bipxx/scikit-imagexx/actions/runs/39777205177) are retained. Each ratio below is a geometric mean of 21 cases, including tiny calls and regressions. These are source-specific GitHub-hosted runner results.

| Host | Method | Workers | Original / current | Initial native / current |
|---|---|---:|---:|---:|
| Linux | gaussian | 1 | 2.786x | 1.022x |
| Linux | gaussian | 4 | 2.932x | 1.024x |
| Linux | mean | 1 | 1.343x | 1.017x |
| Linux | mean | 4 | 1.628x | 1.022x |
| Linux | median | 1 | 6.197x | 1.018x |
| Linux | median | 4 | 10.770x | 1.014x |
| Windows | gaussian | 1 | 1.694x | 1.016x |
| Windows | gaussian | 4 | 2.067x | 1.015x |
| Windows | mean | 1 | 0.793x | 1.013x |
| Windows | mean | 4 | 1.093x | 1.014x |
| Windows | median | 1 | 5.552x | 1.012x |
| Windows | median | 4 | 8.256x | 1.010x |

The small 1-2% aggregate native gains do not establish a broad optimization. Windows mean remains slower than upstream in aggregate, while tiny calls, Fortran mean inputs and Windows large-window median remain bottlenecks. All per-case adverse results are retained below. Optional workers do not guarantee a speedup.

| Host | Dtype | Layout | Method | Window | Workers | Original ms | Current ms | Original / current | Initial / current |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| Linux | uint8 | contiguous | gaussian | 3 | 1 | 1.67113 | 0.36377 | 4.594x | 1.004x |
| Linux | uint8 | contiguous | gaussian | 3 | 4 | 1.60149 | 0.38340 | 4.177x | 1.028x |
| Linux | uint8 | contiguous | gaussian | 15 | 1 | 3.58538 | 2.39625 | 1.496x | 1.004x |
| Linux | uint8 | contiguous | gaussian | 15 | 4 | 3.56286 | 1.25429 | 2.841x | 0.987x |
| Linux | uint8 | fortran | gaussian | 3 | 1 | 1.49414 | 0.73608 | 2.030x | 1.006x |
| Linux | uint8 | fortran | gaussian | 3 | 4 | 1.49155 | 0.81791 | 1.824x | 1.002x |
| Linux | uint8 | strided | gaussian | 3 | 1 | 0.81100 | 0.25868 | 3.135x | 1.016x |
| Linux | uint8 | strided | gaussian | 3 | 4 | 0.80854 | 0.25490 | 3.172x | 1.025x |
| Linux | uint8 | offset | gaussian | 3 | 1 | 1.58524 | 0.38709 | 4.095x | 1.015x |
| Linux | uint8 | offset | gaussian | 3 | 4 | 1.58609 | 0.43755 | 3.625x | 1.012x |
| Linux | uint8 | constant-border | gaussian | 3 | 1 | 1.56878 | 0.35025 | 4.479x | 1.011x |
| Linux | uint8 | constant-border | gaussian | 3 | 4 | 1.58824 | 0.38277 | 4.149x | 1.011x |
| Linux | uint8 | tiny | gaussian | 3 | 1 | 0.05910 | 0.03683 | 1.605x | 1.086x |
| Linux | uint8 | tiny | gaussian | 3 | 4 | 0.06043 | 0.03668 | 1.647x | 1.096x |
| Linux | uint8 | contiguous | mean | 3 | 1 | 1.68290 | 0.97190 | 1.732x | 0.994x |
| Linux | uint8 | contiguous | mean | 3 | 4 | 1.68687 | 0.72877 | 2.315x | 1.012x |
| Linux | uint8 | contiguous | mean | 15 | 1 | 1.67542 | 0.97175 | 1.724x | 1.000x |
| Linux | uint8 | contiguous | mean | 15 | 4 | 1.67606 | 0.73272 | 2.287x | 1.003x |
| Linux | uint8 | fortran | mean | 3 | 1 | 1.58229 | 1.34335 | 1.178x | 1.016x |
| Linux | uint8 | fortran | mean | 3 | 4 | 1.61035 | 1.10256 | 1.461x | 1.025x |
| Linux | uint8 | strided | mean | 3 | 1 | 0.83610 | 0.51713 | 1.617x | 1.027x |
| Linux | uint8 | strided | mean | 3 | 4 | 0.83942 | 0.51711 | 1.623x | 1.013x |
| Linux | uint8 | offset | mean | 3 | 1 | 1.65571 | 1.02241 | 1.619x | 0.983x |
| Linux | uint8 | offset | mean | 3 | 4 | 1.67890 | 0.79001 | 2.125x | 0.989x |
| Linux | uint8 | constant-border | mean | 3 | 1 | 1.67290 | 0.96488 | 1.734x | 1.003x |
| Linux | uint8 | constant-border | mean | 3 | 4 | 1.67451 | 0.75449 | 2.219x | 0.972x |
| Linux | uint8 | tiny | mean | 3 | 1 | 0.01427 | 0.02428 | 0.588x | 1.127x |
| Linux | uint8 | tiny | mean | 3 | 4 | 0.01422 | 0.02381 | 0.597x | 1.137x |
| Linux | uint8 | contiguous | median | 3 | 1 | 32.47097 | 2.56019 | 12.683x | 1.009x |
| Linux | uint8 | contiguous | median | 3 | 4 | 32.44666 | 1.37850 | 23.538x | 0.994x |
| Linux | uint8 | contiguous | median | 15 | 1 | 785.19759 | 737.46352 | 1.065x | 1.000x |
| Linux | uint8 | contiguous | median | 15 | 4 | 785.33724 | 203.59905 | 3.857x | 0.998x |
| Linux | uint8 | fortran | median | 3 | 1 | 32.79786 | 2.97473 | 11.026x | 1.005x |
| Linux | uint8 | fortran | median | 3 | 4 | 32.80890 | 1.76413 | 18.598x | 0.985x |
| Linux | uint8 | strided | median | 3 | 1 | 16.46712 | 1.37392 | 11.985x | 1.015x |
| Linux | uint8 | strided | median | 3 | 4 | 16.52920 | 1.37110 | 12.055x | 1.005x |
| Linux | uint8 | offset | median | 3 | 1 | 32.36308 | 2.60798 | 12.409x | 1.007x |
| Linux | uint8 | offset | median | 3 | 4 | 32.45404 | 1.44839 | 22.407x | 0.982x |
| Linux | uint8 | constant-border | median | 3 | 1 | 32.45202 | 2.59057 | 12.527x | 1.001x |
| Linux | uint8 | constant-border | median | 3 | 4 | 32.47518 | 1.26513 | 25.669x | 0.992x |
| Linux | uint8 | tiny | median | 3 | 1 | 0.02456 | 0.02404 | 1.021x | 1.084x |
| Linux | uint8 | tiny | median | 3 | 4 | 0.02438 | 0.02378 | 1.025x | 1.107x |
| Linux | float32 | contiguous | gaussian | 3 | 1 | 1.32527 | 0.27377 | 4.841x | 1.016x |
| Linux | float32 | contiguous | gaussian | 3 | 4 | 1.37764 | 0.29095 | 4.735x | 1.024x |
| Linux | float32 | contiguous | gaussian | 15 | 1 | 3.34340 | 1.48876 | 2.246x | 0.998x |
| Linux | float32 | contiguous | gaussian | 15 | 4 | 3.29732 | 0.91034 | 3.622x | 0.983x |
| Linux | float32 | fortran | gaussian | 3 | 1 | 1.26684 | 0.89013 | 1.423x | 1.012x |
| Linux | float32 | fortran | gaussian | 3 | 4 | 1.27726 | 0.95635 | 1.336x | 1.009x |
| Linux | float32 | strided | gaussian | 3 | 1 | 0.67315 | 0.22215 | 3.030x | 1.021x |
| Linux | float32 | strided | gaussian | 3 | 4 | 0.68317 | 0.22563 | 3.028x | 1.018x |
| Linux | float32 | offset | gaussian | 3 | 1 | 1.33011 | 0.28769 | 4.623x | 1.014x |
| Linux | float32 | offset | gaussian | 3 | 4 | 1.35459 | 0.33722 | 4.017x | 1.008x |
| Linux | float32 | constant-border | gaussian | 3 | 1 | 1.32646 | 0.27855 | 4.762x | 1.001x |
| Linux | float32 | constant-border | gaussian | 3 | 4 | 1.33879 | 0.28846 | 4.641x | 1.033x |
| Linux | float32 | tiny | gaussian | 3 | 1 | 0.05771 | 0.03305 | 1.746x | 1.108x |
| Linux | float32 | tiny | gaussian | 3 | 4 | 0.05700 | 0.03262 | 1.747x | 1.110x |
| Linux | float32 | contiguous | mean | 3 | 1 | 1.42090 | 0.81258 | 1.749x | 0.991x |
| Linux | float32 | contiguous | mean | 3 | 4 | 1.41378 | 0.58815 | 2.404x | 1.011x |
| Linux | float32 | contiguous | mean | 15 | 1 | 1.41989 | 0.82743 | 1.716x | 0.999x |
| Linux | float32 | contiguous | mean | 15 | 4 | 1.43007 | 0.59632 | 2.398x | 1.003x |
| Linux | float32 | fortran | mean | 3 | 1 | 1.35555 | 1.44262 | 0.940x | 0.979x |
| Linux | float32 | fortran | mean | 3 | 4 | 1.34361 | 1.26992 | 1.058x | 1.033x |
| Linux | float32 | strided | mean | 3 | 1 | 0.69753 | 0.48416 | 1.441x | 0.999x |
| Linux | float32 | strided | mean | 3 | 4 | 0.69565 | 0.47967 | 1.450x | 1.013x |
| Linux | float32 | offset | mean | 3 | 1 | 1.42372 | 0.85272 | 1.670x | 0.978x |
| Linux | float32 | offset | mean | 3 | 4 | 1.42347 | 0.63565 | 2.239x | 1.017x |
| Linux | float32 | constant-border | mean | 3 | 1 | 1.42851 | 0.83795 | 1.705x | 0.971x |
| Linux | float32 | constant-border | mean | 3 | 4 | 1.42451 | 0.59496 | 2.394x | 0.997x |
| Linux | float32 | tiny | mean | 3 | 1 | 0.01277 | 0.02120 | 0.602x | 1.163x |
| Linux | float32 | tiny | mean | 3 | 4 | 0.01290 | 0.02188 | 0.590x | 1.124x |
| Linux | float32 | contiguous | median | 3 | 1 | 32.58478 | 2.22612 | 14.637x | 1.008x |
| Linux | float32 | contiguous | median | 3 | 4 | 32.51223 | 1.06680 | 30.476x | 0.995x |
| Linux | float32 | contiguous | median | 15 | 1 | 784.81864 | 725.15334 | 1.082x | 1.000x |
| Linux | float32 | contiguous | median | 15 | 4 | 784.34241 | 189.33614 | 4.143x | 1.001x |
| Linux | float32 | fortran | median | 3 | 1 | 32.89053 | 2.82716 | 11.634x | 1.003x |
| Linux | float32 | fortran | median | 3 | 4 | 32.88093 | 1.73303 | 18.973x | 1.007x |
| Linux | float32 | strided | median | 3 | 1 | 16.53147 | 1.19916 | 13.786x | 1.001x |
| Linux | float32 | strided | median | 3 | 4 | 16.42023 | 1.19736 | 13.714x | 0.997x |
| Linux | float32 | offset | median | 3 | 1 | 32.52862 | 2.24228 | 14.507x | 1.002x |
| Linux | float32 | offset | median | 3 | 4 | 32.54519 | 1.11649 | 29.150x | 0.999x |
| Linux | float32 | constant-border | median | 3 | 1 | 32.51396 | 2.23425 | 14.552x | 0.997x |
| Linux | float32 | constant-border | median | 3 | 4 | 32.62455 | 1.00832 | 32.355x | 0.983x |
| Linux | float32 | tiny | median | 3 | 1 | 0.02344 | 0.02160 | 1.085x | 1.153x |
| Linux | float32 | tiny | median | 3 | 4 | 0.02324 | 0.02191 | 1.061x | 1.105x |
| Linux | float64 | contiguous | gaussian | 3 | 1 | 1.53058 | 0.30874 | 4.958x | 1.019x |
| Linux | float64 | contiguous | gaussian | 3 | 4 | 1.51004 | 0.31524 | 4.790x | 1.018x |
| Linux | float64 | contiguous | gaussian | 15 | 1 | 3.51425 | 2.34467 | 1.499x | 1.003x |
| Linux | float64 | contiguous | gaussian | 15 | 4 | 3.50408 | 1.16816 | 3.000x | 0.972x |
| Linux | float64 | fortran | gaussian | 3 | 1 | 1.46306 | 1.25052 | 1.170x | 1.008x |
| Linux | float64 | fortran | gaussian | 3 | 4 | 1.46136 | 1.30846 | 1.117x | 1.014x |
| Linux | float64 | strided | gaussian | 3 | 1 | 0.75856 | 0.24739 | 3.066x | 1.011x |
| Linux | float64 | strided | gaussian | 3 | 4 | 0.75251 | 0.24323 | 3.094x | 1.011x |
| Linux | float64 | offset | gaussian | 3 | 1 | 1.52618 | 0.34024 | 4.486x | 1.026x |
| Linux | float64 | offset | gaussian | 3 | 4 | 1.54554 | 0.37079 | 4.168x | 1.018x |
| Linux | float64 | constant-border | gaussian | 3 | 1 | 1.51712 | 0.32012 | 4.739x | 0.996x |
| Linux | float64 | constant-border | gaussian | 3 | 4 | 1.52208 | 0.30898 | 4.926x | 1.025x |
| Linux | float64 | tiny | gaussian | 3 | 1 | 0.05675 | 0.03375 | 1.681x | 1.099x |
| Linux | float64 | tiny | gaussian | 3 | 4 | 0.05570 | 0.03365 | 1.655x | 1.104x |
| Linux | float64 | contiguous | mean | 3 | 1 | 1.62382 | 0.92092 | 1.763x | 0.986x |
| Linux | float64 | contiguous | mean | 3 | 4 | 1.61488 | 0.64604 | 2.500x | 1.001x |
| Linux | float64 | contiguous | mean | 15 | 1 | 1.64821 | 0.93216 | 1.768x | 0.995x |
| Linux | float64 | contiguous | mean | 15 | 4 | 1.63026 | 0.65261 | 2.498x | 1.014x |
| Linux | float64 | fortran | mean | 3 | 1 | 1.51958 | 1.81353 | 0.838x | 1.013x |
| Linux | float64 | fortran | mean | 3 | 4 | 1.53752 | 1.67718 | 0.917x | 0.985x |
| Linux | float64 | strided | mean | 3 | 1 | 0.76477 | 0.50127 | 1.526x | 1.017x |
| Linux | float64 | strided | mean | 3 | 4 | 0.76502 | 0.50970 | 1.501x | 0.992x |
| Linux | float64 | offset | mean | 3 | 1 | 1.63753 | 0.94948 | 1.725x | 1.006x |
| Linux | float64 | offset | mean | 3 | 4 | 1.60954 | 0.72304 | 2.226x | 0.996x |
| Linux | float64 | constant-border | mean | 3 | 1 | 1.61616 | 0.91091 | 1.774x | 1.006x |
| Linux | float64 | constant-border | mean | 3 | 4 | 1.61856 | 0.65515 | 2.471x | 1.007x |
| Linux | float64 | tiny | mean | 3 | 1 | 0.01354 | 0.02204 | 0.615x | 1.130x |
| Linux | float64 | tiny | mean | 3 | 4 | 0.01269 | 0.02133 | 0.595x | 1.143x |
| Linux | float64 | contiguous | median | 3 | 1 | 32.32718 | 2.52846 | 12.785x | 1.000x |
| Linux | float64 | contiguous | median | 3 | 4 | 32.40532 | 1.28655 | 25.188x | 1.005x |
| Linux | float64 | contiguous | median | 15 | 1 | 785.68501 | 737.50546 | 1.065x | 1.000x |
| Linux | float64 | contiguous | median | 15 | 4 | 785.48594 | 203.93937 | 3.852x | 0.997x |
| Linux | float64 | fortran | median | 3 | 1 | 32.77126 | 3.43527 | 9.540x | 0.993x |
| Linux | float64 | fortran | median | 3 | 4 | 32.64892 | 2.24425 | 14.548x | 1.013x |
| Linux | float64 | strided | median | 3 | 1 | 16.51733 | 1.37359 | 12.025x | 1.012x |
| Linux | float64 | strided | median | 3 | 4 | 16.48030 | 1.35681 | 12.146x | 1.031x |
| Linux | float64 | offset | median | 3 | 1 | 32.51151 | 2.56028 | 12.698x | 1.010x |
| Linux | float64 | offset | median | 3 | 4 | 32.56698 | 1.35516 | 24.032x | 1.006x |
| Linux | float64 | constant-border | median | 3 | 1 | 32.53269 | 2.53589 | 12.829x | 0.995x |
| Linux | float64 | constant-border | median | 3 | 4 | 32.38431 | 1.18032 | 27.437x | 1.008x |
| Linux | float64 | tiny | median | 3 | 1 | 0.02292 | 0.02226 | 1.030x | 1.103x |
| Linux | float64 | tiny | median | 3 | 4 | 0.02378 | 0.02231 | 1.066x | 1.089x |
| Windows | uint8 | contiguous | gaussian | 3 | 1 | 2.54990 | 1.36840 | 1.863x | 1.026x |
| Windows | uint8 | contiguous | gaussian | 3 | 4 | 2.59700 | 1.07719 | 2.411x | 0.988x |
| Windows | uint8 | contiguous | gaussian | 15 | 1 | 4.19710 | 3.66310 | 1.146x | 1.001x |
| Windows | uint8 | contiguous | gaussian | 15 | 4 | 4.20920 | 1.91822 | 2.194x | 0.999x |
| Windows | uint8 | fortran | gaussian | 3 | 1 | 2.48836 | 1.54170 | 1.614x | 1.006x |
| Windows | uint8 | fortran | gaussian | 3 | 4 | 2.56342 | 1.22962 | 2.085x | 0.984x |
| Windows | uint8 | strided | gaussian | 3 | 1 | 1.28410 | 0.79955 | 1.606x | 1.012x |
| Windows | uint8 | strided | gaussian | 3 | 4 | 1.27712 | 0.79557 | 1.605x | 1.004x |
| Windows | uint8 | offset | gaussian | 3 | 1 | 2.52332 | 1.42146 | 1.775x | 1.000x |
| Windows | uint8 | offset | gaussian | 3 | 4 | 2.56268 | 1.08749 | 2.357x | 1.001x |
| Windows | uint8 | constant-border | gaussian | 3 | 1 | 2.51488 | 1.35374 | 1.858x | 0.997x |
| Windows | uint8 | constant-border | gaussian | 3 | 4 | 2.58302 | 1.05336 | 2.452x | 1.002x |
| Windows | uint8 | tiny | gaussian | 3 | 1 | 0.03071 | 0.01923 | 1.597x | 1.094x |
| Windows | uint8 | tiny | gaussian | 3 | 4 | 0.03004 | 0.01921 | 1.564x | 1.110x |
| Windows | uint8 | contiguous | mean | 3 | 1 | 2.56240 | 2.73956 | 0.935x | 1.001x |
| Windows | uint8 | contiguous | mean | 3 | 4 | 2.58578 | 1.75752 | 1.471x | 1.001x |
| Windows | uint8 | contiguous | mean | 15 | 1 | 2.58053 | 2.76626 | 0.933x | 1.002x |
| Windows | uint8 | contiguous | mean | 15 | 4 | 2.68354 | 1.78127 | 1.507x | 1.008x |
| Windows | uint8 | fortran | mean | 3 | 1 | 2.52722 | 2.90530 | 0.870x | 1.005x |
| Windows | uint8 | fortran | mean | 3 | 4 | 2.55884 | 1.94010 | 1.319x | 0.999x |
| Windows | uint8 | strided | mean | 3 | 1 | 1.28925 | 1.48784 | 0.867x | 1.006x |
| Windows | uint8 | strided | mean | 3 | 4 | 1.26525 | 1.46163 | 0.866x | 1.002x |
| Windows | uint8 | offset | mean | 3 | 1 | 2.54130 | 2.76982 | 0.917x | 1.003x |
| Windows | uint8 | offset | mean | 3 | 4 | 2.56862 | 1.81307 | 1.417x | 1.002x |
| Windows | uint8 | constant-border | mean | 3 | 1 | 2.56656 | 2.73246 | 0.939x | 1.006x |
| Windows | uint8 | constant-border | mean | 3 | 4 | 2.58228 | 1.75379 | 1.472x | 1.008x |
| Windows | uint8 | tiny | mean | 3 | 1 | 0.00899 | 0.01534 | 0.586x | 1.081x |
| Windows | uint8 | tiny | mean | 3 | 4 | 0.00911 | 0.01582 | 0.576x | 1.096x |
| Windows | uint8 | contiguous | median | 3 | 1 | 29.49000 | 2.87685 | 10.251x | 0.999x |
| Windows | uint8 | contiguous | median | 3 | 4 | 29.58320 | 1.90687 | 15.514x | 1.005x |
| Windows | uint8 | contiguous | median | 15 | 1 | 690.46690 | 761.49700 | 0.907x | 0.988x |
| Windows | uint8 | contiguous | median | 15 | 4 | 691.48040 | 227.51980 | 3.039x | 1.034x |
| Windows | uint8 | fortran | median | 3 | 1 | 30.07920 | 3.03087 | 9.924x | 1.001x |
| Windows | uint8 | fortran | median | 3 | 4 | 30.08650 | 2.07420 | 14.505x | 1.005x |
| Windows | uint8 | strided | median | 3 | 1 | 14.97600 | 1.56123 | 9.592x | 1.013x |
| Windows | uint8 | strided | median | 3 | 4 | 14.92310 | 1.56208 | 9.553x | 0.999x |
| Windows | uint8 | offset | median | 3 | 1 | 29.49660 | 2.90480 | 10.154x | 1.004x |
| Windows | uint8 | offset | median | 3 | 4 | 29.63720 | 1.99537 | 14.853x | 1.004x |
| Windows | uint8 | constant-border | median | 3 | 1 | 29.48660 | 2.84786 | 10.354x | 1.003x |
| Windows | uint8 | constant-border | median | 3 | 4 | 29.41910 | 1.90736 | 15.424x | 1.003x |
| Windows | uint8 | tiny | median | 3 | 1 | 0.01443 | 0.01506 | 0.958x | 1.073x |
| Windows | uint8 | tiny | median | 3 | 4 | 0.01505 | 0.01520 | 0.990x | 1.059x |
| Windows | float32 | contiguous | gaussian | 3 | 1 | 1.64770 | 0.61618 | 2.674x | 1.001x |
| Windows | float32 | contiguous | gaussian | 3 | 4 | 1.63262 | 0.58053 | 2.812x | 1.014x |
| Windows | float32 | contiguous | gaussian | 15 | 1 | 3.36338 | 2.10202 | 1.600x | 0.986x |
| Windows | float32 | contiguous | gaussian | 15 | 4 | 3.36580 | 1.21188 | 2.777x | 0.994x |
| Windows | float32 | fortran | gaussian | 3 | 1 | 1.57618 | 1.32640 | 1.188x | 0.994x |
| Windows | float32 | fortran | gaussian | 3 | 4 | 1.55981 | 1.23840 | 1.260x | 1.018x |
| Windows | float32 | strided | gaussian | 3 | 1 | 0.60096 | 0.42440 | 1.416x | 1.022x |
| Windows | float32 | strided | gaussian | 3 | 4 | 0.60321 | 0.44248 | 1.363x | 1.000x |
| Windows | float32 | offset | gaussian | 3 | 1 | 1.65899 | 0.63364 | 2.618x | 1.056x |
| Windows | float32 | offset | gaussian | 3 | 4 | 1.66014 | 0.62440 | 2.659x | 1.008x |
| Windows | float32 | constant-border | gaussian | 3 | 1 | 1.63840 | 0.61233 | 2.676x | 1.018x |
| Windows | float32 | constant-border | gaussian | 3 | 4 | 1.66183 | 0.56370 | 2.948x | 1.002x |
| Windows | float32 | tiny | gaussian | 3 | 1 | 0.03024 | 0.01849 | 1.636x | 1.054x |
| Windows | float32 | tiny | gaussian | 3 | 4 | 0.02984 | 0.01772 | 1.683x | 1.088x |
| Windows | float32 | contiguous | mean | 3 | 1 | 1.72063 | 1.98469 | 0.867x | 1.002x |
| Windows | float32 | contiguous | mean | 3 | 4 | 1.74450 | 1.17952 | 1.479x | 0.999x |
| Windows | float32 | contiguous | mean | 15 | 1 | 1.73277 | 2.00548 | 0.864x | 0.994x |
| Windows | float32 | contiguous | mean | 15 | 4 | 1.73026 | 1.18895 | 1.455x | 1.006x |
| Windows | float32 | fortran | mean | 3 | 1 | 1.63038 | 2.68200 | 0.608x | 0.997x |
| Windows | float32 | fortran | mean | 3 | 4 | 1.62237 | 1.90463 | 0.852x | 1.000x |
| Windows | float32 | strided | mean | 3 | 1 | 0.64557 | 1.10713 | 0.583x | 1.010x |
| Windows | float32 | strided | mean | 3 | 4 | 0.62294 | 1.22689 | 0.508x | 1.003x |
| Windows | float32 | offset | mean | 3 | 1 | 1.70419 | 1.98734 | 0.858x | 1.004x |
| Windows | float32 | offset | mean | 3 | 4 | 1.68096 | 1.20397 | 1.396x | 1.003x |
| Windows | float32 | constant-border | mean | 3 | 1 | 1.69249 | 1.98219 | 0.854x | 0.998x |
| Windows | float32 | constant-border | mean | 3 | 4 | 1.70220 | 1.18283 | 1.439x | 1.000x |
| Windows | float32 | tiny | mean | 3 | 1 | 0.00859 | 0.01413 | 0.608x | 1.086x |
| Windows | float32 | tiny | mean | 3 | 4 | 0.00872 | 0.01449 | 0.602x | 1.065x |
| Windows | float32 | contiguous | median | 3 | 1 | 28.63020 | 1.94166 | 14.745x | 1.004x |
| Windows | float32 | contiguous | median | 3 | 4 | 28.66680 | 1.23867 | 23.143x | 0.991x |
| Windows | float32 | contiguous | median | 15 | 1 | 687.23230 | 756.28630 | 0.909x | 0.986x |
| Windows | float32 | contiguous | median | 15 | 4 | 686.54320 | 222.83020 | 3.081x | 1.046x |
| Windows | float32 | fortran | median | 3 | 1 | 28.91080 | 2.63278 | 10.981x | 1.000x |
| Windows | float32 | fortran | median | 3 | 4 | 29.03340 | 1.97293 | 14.716x | 0.993x |
| Windows | float32 | strided | median | 3 | 1 | 14.42680 | 1.10556 | 13.049x | 0.994x |
| Windows | float32 | strided | median | 3 | 4 | 14.28000 | 0.87820 | 16.261x | 1.042x |
| Windows | float32 | offset | median | 3 | 1 | 28.66580 | 1.95069 | 14.695x | 0.998x |
| Windows | float32 | offset | median | 3 | 4 | 28.56960 | 1.25750 | 22.719x | 1.007x |
| Windows | float32 | constant-border | median | 3 | 1 | 28.59690 | 1.93933 | 14.746x | 0.999x |
| Windows | float32 | constant-border | median | 3 | 4 | 28.66250 | 1.24193 | 23.079x | 0.995x |
| Windows | float32 | tiny | median | 3 | 1 | 0.01352 | 0.01385 | 0.976x | 1.110x |
| Windows | float32 | tiny | median | 3 | 4 | 0.01346 | 0.01411 | 0.954x | 1.113x |
| Windows | float64 | contiguous | gaussian | 3 | 1 | 2.08313 | 0.97332 | 2.140x | 0.978x |
| Windows | float64 | contiguous | gaussian | 3 | 4 | 2.10785 | 0.72671 | 2.901x | 1.006x |
| Windows | float64 | contiguous | gaussian | 15 | 1 | 3.75120 | 3.18375 | 1.178x | 0.995x |
| Windows | float64 | contiguous | gaussian | 15 | 4 | 3.80900 | 1.59141 | 2.393x | 1.001x |
| Windows | float64 | fortran | gaussian | 3 | 1 | 2.03136 | 1.97146 | 1.030x | 0.990x |
| Windows | float64 | fortran | gaussian | 3 | 4 | 2.05340 | 1.72091 | 1.193x | 1.001x |
| Windows | float64 | strided | gaussian | 3 | 1 | 1.06301 | 0.78275 | 1.358x | 1.011x |
| Windows | float64 | strided | gaussian | 3 | 4 | 1.07150 | 0.77771 | 1.378x | 1.000x |
| Windows | float64 | offset | gaussian | 3 | 1 | 2.06028 | 1.01058 | 2.039x | 1.002x |
| Windows | float64 | offset | gaussian | 3 | 4 | 2.08630 | 0.75550 | 2.761x | 1.013x |
| Windows | float64 | constant-border | gaussian | 3 | 1 | 2.07063 | 0.93551 | 2.213x | 1.001x |
| Windows | float64 | constant-border | gaussian | 3 | 4 | 2.06130 | 0.74901 | 2.752x | 1.003x |
| Windows | float64 | tiny | gaussian | 3 | 1 | 0.02999 | 0.01783 | 1.682x | 1.099x |
| Windows | float64 | tiny | gaussian | 3 | 4 | 0.03062 | 0.01800 | 1.701x | 1.090x |
| Windows | float64 | contiguous | mean | 3 | 1 | 2.11883 | 2.32585 | 0.911x | 0.995x |
| Windows | float64 | contiguous | mean | 3 | 4 | 2.12667 | 1.42401 | 1.493x | 1.004x |
| Windows | float64 | contiguous | mean | 15 | 1 | 2.12970 | 2.33527 | 0.912x | 0.998x |
| Windows | float64 | contiguous | mean | 15 | 4 | 2.12930 | 1.43624 | 1.483x | 1.008x |
| Windows | float64 | fortran | mean | 3 | 1 | 2.08940 | 3.33380 | 0.627x | 0.998x |
| Windows | float64 | fortran | mean | 3 | 4 | 2.06670 | 2.45336 | 0.842x | 1.001x |
| Windows | float64 | strided | mean | 3 | 1 | 1.06044 | 1.43840 | 0.737x | 1.001x |
| Windows | float64 | strided | mean | 3 | 4 | 1.07251 | 1.45443 | 0.737x | 1.000x |
| Windows | float64 | offset | mean | 3 | 1 | 2.12620 | 2.36997 | 0.897x | 0.998x |
| Windows | float64 | offset | mean | 3 | 4 | 2.23343 | 1.49104 | 1.498x | 1.012x |
| Windows | float64 | constant-border | mean | 3 | 1 | 2.11113 | 2.30132 | 0.917x | 1.005x |
| Windows | float64 | constant-border | mean | 3 | 4 | 2.13250 | 1.42366 | 1.498x | 1.013x |
| Windows | float64 | tiny | mean | 3 | 1 | 0.00859 | 0.01386 | 0.620x | 1.100x |
| Windows | float64 | tiny | mean | 3 | 4 | 0.00857 | 0.01401 | 0.612x | 1.078x |
| Windows | float64 | contiguous | median | 3 | 1 | 29.00090 | 2.44650 | 11.854x | 0.997x |
| Windows | float64 | contiguous | median | 3 | 4 | 29.01560 | 1.56839 | 18.500x | 1.000x |
| Windows | float64 | contiguous | median | 15 | 1 | 695.46540 | 765.41140 | 0.909x | 0.991x |
| Windows | float64 | contiguous | median | 15 | 4 | 698.89580 | 263.48310 | 2.653x | 0.930x |
| Windows | float64 | fortran | median | 3 | 1 | 29.69890 | 3.52090 | 8.435x | 0.997x |
| Windows | float64 | fortran | median | 3 | 4 | 29.96980 | 3.15310 | 9.505x | 0.875x |
| Windows | float64 | strided | median | 3 | 1 | 14.72870 | 1.56242 | 9.427x | 1.000x |
| Windows | float64 | strided | median | 3 | 4 | 14.85470 | 1.58891 | 9.349x | 1.027x |
| Windows | float64 | offset | median | 3 | 1 | 29.16580 | 2.51502 | 11.597x | 0.996x |
| Windows | float64 | offset | median | 3 | 4 | 29.05070 | 1.64700 | 17.639x | 0.999x |
| Windows | float64 | constant-border | median | 3 | 1 | 29.05650 | 2.43244 | 11.945x | 1.004x |
| Windows | float64 | constant-border | median | 3 | 4 | 29.09960 | 1.57451 | 18.482x | 1.006x |
| Windows | float64 | tiny | median | 3 | 1 | 0.01375 | 0.01372 | 1.003x | 1.107x |
| Windows | float64 | tiny | median | 3 | 4 | 0.01394 | 0.01400 | 0.995x | 1.094x |
