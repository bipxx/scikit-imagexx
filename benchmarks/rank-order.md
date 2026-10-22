# Rank ordering initial checkpoint

Measured source: `65bcc9e`; product fingerprint `86b8cbccebfe105ec7fcc281169d644188da3e4a66e74e39fa4e3263f77fd569`. Reference: original scikit-image 0.26.0, NumPy 2.5.4. This is a partial native port, not full scikit-image parity.

The C++ implementation transforms real values into ordered integer keys with runtime SIMD, sorts them with the existing optional parallel radix sorter, then emits ranks and unique representatives. All integer widths, bool, float16/32/64, N-D arrays/scalars, empty errors, native/endian/unaligned storage, dtype metadata and memmap output class have dedicated parity tests. NaNs and mixed signed zeros require explicit fallback; unusual array types are not claimed. The initial baseline missed noncanonical bool representative bytes, corrected in the subsequent uniform checkpoint.

Initial scoped validation: Windows passes 6,348 rank/API/metadata/histogram-matching cases in each SIMD mode; Linux/NumPy 1.24 passes 2,230 rank/histogram-matching cases in each mode. Native release, portable, ASan/UBSan and dynamic-team checks pass 2/2. The generic coverage runner now compares tuple members independently, retaining dtype/shape checks and numerical budgets.

The fixed 35-case corpus uses seed 3019; uint8/uint64/float16/float32/float64; 512 by 513 images; quantized, Fortran, reversed-strided, constant, tiny, monotone and wide-range workloads. One/four worker limits use seven randomized samples, 15 ms target batches and 20 ms cooldown. Every call includes allocation, normalization, sorting and Python dispatch. Exact values/bytes and reconstruction are checked before timing with fallback disabled and reference pools fixed to one. [Exact packages, wheels and all samples](https://github.com/bipxx/scikit-imagexx/actions/runs/39780318912) are archived.

| Host | Workers | Original / current geometric mean |
|---|---:|---:|
| Linux | 1 | 1.591x |
| Linux | 4 | 1.915x |
| Windows | 1 | 1.081x |
| Windows | 4 | 1.415x |

Speedup is original/current; below 1 is slower. Continuous/wide-range floating sorts and Fortran normalization remain bottlenecks; aggregates do not establish a win for every case. All adverse results are retained below.

| Host | Dtype | Case | Workers | Original ms | Current ms | Original / current |
|---|---|---|---:|---:|---:|---:|
| Linux | uint8 | quantized | 1 | 13.83230 | 5.17557 | 2.673x |
| Linux | uint8 | quantized | 4 | 13.81653 | 4.19452 | 3.294x |
| Linux | uint8 | fortran | 1 | 13.79841 | 5.44926 | 2.532x |
| Linux | uint8 | fortran | 4 | 13.79640 | 4.93157 | 2.798x |
| Linux | uint8 | strided | 1 | 6.49893 | 1.57037 | 4.138x |
| Linux | uint8 | strided | 4 | 6.52658 | 1.57944 | 4.132x |
| Linux | uint8 | constant | 1 | 4.05587 | 4.10285 | 0.989x |
| Linux | uint8 | constant | 4 | 4.04885 | 3.68790 | 1.098x |
| Linux | uint8 | tiny | 1 | 0.02104 | 0.00318 | 6.620x |
| Linux | uint8 | tiny | 4 | 0.01978 | 0.00298 | 6.634x |
| Linux | uint8 | monotone | 1 | 6.58705 | 4.91762 | 1.339x |
| Linux | uint8 | monotone | 4 | 6.54727 | 3.91287 | 1.673x |
| Linux | uint8 | wide-range | 1 | 13.18309 | 4.94256 | 2.667x |
| Linux | uint8 | wide-range | 4 | 13.09641 | 4.29928 | 3.046x |
| Linux | uint64 | quantized | 1 | 6.05354 | 3.67528 | 1.647x |
| Linux | uint64 | quantized | 4 | 5.99332 | 2.44931 | 2.447x |
| Linux | uint64 | fortran | 1 | 6.98407 | 7.08555 | 0.986x |
| Linux | uint64 | fortran | 4 | 6.98685 | 6.56620 | 1.064x |
| Linux | uint64 | strided | 1 | 2.96015 | 1.86607 | 1.586x |
| Linux | uint64 | strided | 4 | 2.96910 | 1.79044 | 1.658x |
| Linux | uint64 | constant | 1 | 2.45576 | 4.67904 | 0.525x |
| Linux | uint64 | constant | 4 | 2.45099 | 4.05188 | 0.605x |
| Linux | uint64 | tiny | 1 | 0.02106 | 0.00343 | 6.143x |
| Linux | uint64 | tiny | 4 | 0.02096 | 0.00339 | 6.176x |
| Linux | uint64 | monotone | 1 | 5.27842 | 5.65722 | 0.933x |
| Linux | uint64 | monotone | 4 | 5.27089 | 4.67344 | 1.128x |
| Linux | uint64 | wide-range | 1 | 13.24029 | 13.82323 | 0.958x |
| Linux | uint64 | wide-range | 4 | 13.12560 | 9.73683 | 1.348x |
| Linux | float16 | quantized | 1 | 17.01146 | 6.85931 | 2.480x |
| Linux | float16 | quantized | 4 | 16.89292 | 4.57678 | 3.691x |
| Linux | float16 | fortran | 1 | 17.82761 | 8.33038 | 2.140x |
| Linux | float16 | fortran | 4 | 17.51836 | 5.91784 | 2.960x |
| Linux | float16 | strided | 1 | 8.09284 | 2.66913 | 3.032x |
| Linux | float16 | strided | 4 | 8.09525 | 2.66996 | 3.032x |
| Linux | float16 | constant | 1 | 6.46458 | 4.30741 | 1.501x |
| Linux | float16 | constant | 4 | 6.47740 | 3.87063 | 1.673x |
| Linux | float16 | tiny | 1 | 0.02132 | 0.00425 | 5.018x |
| Linux | float16 | tiny | 4 | 0.02122 | 0.00426 | 4.984x |
| Linux | float16 | monotone | 1 | 14.17802 | 7.26366 | 1.952x |
| Linux | float16 | monotone | 4 | 14.40778 | 5.28130 | 2.728x |
| Linux | float16 | wide-range | 1 | 26.16331 | 9.15637 | 2.857x |
| Linux | float16 | wide-range | 4 | 26.07206 | 7.18085 | 3.631x |
| Linux | float32 | quantized | 1 | 5.74543 | 5.13137 | 1.120x |
| Linux | float32 | quantized | 4 | 5.68151 | 2.97111 | 1.912x |
| Linux | float32 | fortran | 1 | 6.71014 | 8.47304 | 0.792x |
| Linux | float32 | fortran | 4 | 6.69840 | 6.37016 | 1.052x |
| Linux | float32 | strided | 1 | 2.73271 | 2.65588 | 1.029x |
| Linux | float32 | strided | 4 | 2.76603 | 2.64238 | 1.047x |
| Linux | float32 | constant | 1 | 2.48705 | 4.58129 | 0.543x |
| Linux | float32 | constant | 4 | 2.51736 | 4.31349 | 0.584x |
| Linux | float32 | tiny | 1 | 0.02064 | 0.00322 | 6.400x |
| Linux | float32 | tiny | 4 | 0.02251 | 0.00369 | 6.104x |
| Linux | float32 | monotone | 1 | 5.50800 | 7.64449 | 0.721x |
| Linux | float32 | monotone | 4 | 5.54572 | 5.46832 | 1.014x |
| Linux | float32 | wide-range | 1 | 11.10822 | 12.02179 | 0.924x |
| Linux | float32 | wide-range | 4 | 10.86625 | 9.31165 | 1.167x |
| Linux | float64 | quantized | 1 | 5.76849 | 5.73261 | 1.006x |
| Linux | float64 | quantized | 4 | 5.64649 | 3.55056 | 1.590x |
| Linux | float64 | fortran | 1 | 6.96173 | 9.89711 | 0.703x |
| Linux | float64 | fortran | 4 | 7.05825 | 8.29754 | 0.851x |
| Linux | float64 | strided | 1 | 2.80803 | 3.42202 | 0.821x |
| Linux | float64 | strided | 4 | 2.78244 | 3.38017 | 0.823x |
| Linux | float64 | constant | 1 | 2.55737 | 3.31402 | 0.772x |
| Linux | float64 | constant | 4 | 2.57086 | 2.78661 | 0.923x |
| Linux | float64 | tiny | 1 | 0.02223 | 0.00330 | 6.728x |
| Linux | float64 | tiny | 4 | 0.02071 | 0.00319 | 6.499x |
| Linux | float64 | monotone | 1 | 5.57033 | 8.21427 | 0.678x |
| Linux | float64 | monotone | 4 | 5.62854 | 5.51386 | 1.021x |
| Linux | float64 | wide-range | 1 | 12.94195 | 15.92417 | 0.813x |
| Linux | float64 | wide-range | 4 | 12.53171 | 9.48981 | 1.321x |
| Windows | uint8 | quantized | 1 | 16.85910 | 11.05010 | 1.526x |
| Windows | uint8 | quantized | 4 | 16.84800 | 7.00345 | 2.406x |
| Windows | uint8 | fortran | 1 | 16.48110 | 11.35360 | 1.452x |
| Windows | uint8 | fortran | 4 | 16.98070 | 7.82030 | 2.171x |
| Windows | uint8 | strided | 1 | 7.96760 | 5.29040 | 1.506x |
| Windows | uint8 | strided | 4 | 8.64840 | 5.39595 | 1.603x |
| Windows | uint8 | constant | 1 | 6.46020 | 9.20060 | 0.702x |
| Windows | uint8 | constant | 4 | 6.11775 | 5.81545 | 1.052x |
| Windows | uint8 | tiny | 1 | 0.01300 | 0.00418 | 3.112x |
| Windows | uint8 | tiny | 4 | 0.01255 | 0.00406 | 3.095x |
| Windows | uint8 | monotone | 1 | 9.30480 | 11.20260 | 0.831x |
| Windows | uint8 | monotone | 4 | 9.66250 | 6.61510 | 1.461x |
| Windows | uint8 | wide-range | 1 | 16.97470 | 10.72860 | 1.582x |
| Windows | uint8 | wide-range | 4 | 16.54700 | 7.23980 | 2.286x |
| Windows | uint64 | quantized | 1 | 9.08720 | 10.82360 | 0.840x |
| Windows | uint64 | quantized | 4 | 8.91060 | 7.05250 | 1.263x |
| Windows | uint64 | fortran | 1 | 10.07480 | 13.38320 | 0.753x |
| Windows | uint64 | fortran | 4 | 10.44570 | 9.27980 | 1.126x |
| Windows | uint64 | strided | 1 | 4.43330 | 6.11685 | 0.725x |
| Windows | uint64 | strided | 4 | 4.30517 | 6.06805 | 0.709x |
| Windows | uint64 | constant | 1 | 5.59840 | 9.15470 | 0.612x |
| Windows | uint64 | constant | 4 | 5.44580 | 6.31260 | 0.863x |
| Windows | uint64 | tiny | 1 | 0.01371 | 0.00350 | 3.921x |
| Windows | uint64 | tiny | 4 | 0.01419 | 0.00348 | 4.076x |
| Windows | uint64 | monotone | 1 | 8.56030 | 11.84700 | 0.723x |
| Windows | uint64 | monotone | 4 | 8.67690 | 8.14930 | 1.065x |
| Windows | uint64 | wide-range | 1 | 17.52020 | 20.56120 | 0.852x |
| Windows | uint64 | wide-range | 4 | 16.46990 | 16.45430 | 1.001x |
| Windows | float16 | quantized | 1 | 22.62170 | 11.65360 | 1.941x |
| Windows | float16 | quantized | 4 | 23.70800 | 8.76330 | 2.705x |
| Windows | float16 | fortran | 1 | 23.79590 | 12.36270 | 1.925x |
| Windows | float16 | fortran | 4 | 24.50280 | 9.05280 | 2.707x |
| Windows | float16 | strided | 1 | 11.81660 | 6.59100 | 1.793x |
| Windows | float16 | strided | 4 | 11.66480 | 6.20720 | 1.879x |
| Windows | float16 | constant | 1 | 12.52060 | 10.23290 | 1.224x |
| Windows | float16 | constant | 4 | 12.74280 | 6.46755 | 1.970x |
| Windows | float16 | tiny | 1 | 0.01289 | 0.00470 | 2.745x |
| Windows | float16 | tiny | 4 | 0.01333 | 0.00472 | 2.826x |
| Windows | float16 | monotone | 1 | 20.84290 | 11.98130 | 1.740x |
| Windows | float16 | monotone | 4 | 21.37900 | 8.28760 | 2.580x |
| Windows | float16 | wide-range | 1 | 32.66550 | 12.72830 | 2.566x |
| Windows | float16 | wide-range | 4 | 32.06580 | 9.13900 | 3.509x |
| Windows | float32 | quantized | 1 | 7.61705 | 12.32710 | 0.618x |
| Windows | float32 | quantized | 4 | 7.62900 | 8.31190 | 0.918x |
| Windows | float32 | fortran | 1 | 8.24330 | 14.34910 | 0.574x |
| Windows | float32 | fortran | 4 | 8.25080 | 9.69430 | 0.851x |
| Windows | float32 | strided | 1 | 3.83007 | 6.97745 | 0.549x |
| Windows | float32 | strided | 4 | 3.71893 | 6.81405 | 0.546x |
| Windows | float32 | constant | 1 | 5.36285 | 10.12900 | 0.529x |
| Windows | float32 | constant | 4 | 5.27430 | 6.44665 | 0.818x |
| Windows | float32 | tiny | 1 | 0.01260 | 0.00366 | 3.440x |
| Windows | float32 | tiny | 4 | 0.01408 | 0.00371 | 3.792x |
| Windows | float32 | monotone | 1 | 7.97450 | 12.79600 | 0.623x |
| Windows | float32 | monotone | 4 | 7.84670 | 8.47550 | 0.926x |
| Windows | float32 | wide-range | 1 | 12.91460 | 18.70980 | 0.690x |
| Windows | float32 | wide-range | 4 | 12.67930 | 14.31900 | 0.885x |
| Windows | float64 | quantized | 1 | 8.49850 | 15.47610 | 0.549x |
| Windows | float64 | quantized | 4 | 8.61060 | 9.44720 | 0.911x |
| Windows | float64 | fortran | 1 | 9.71090 | 17.78370 | 0.546x |
| Windows | float64 | fortran | 4 | 9.49810 | 11.54590 | 0.823x |
| Windows | float64 | strided | 1 | 4.18130 | 8.51570 | 0.491x |
| Windows | float64 | strided | 4 | 4.07143 | 8.27540 | 0.492x |
| Windows | float64 | constant | 1 | 5.74915 | 10.26090 | 0.560x |
| Windows | float64 | constant | 4 | 5.53810 | 7.57565 | 0.731x |
| Windows | float64 | tiny | 1 | 0.01407 | 0.00331 | 4.256x |
| Windows | float64 | tiny | 4 | 0.01490 | 0.00345 | 4.319x |
| Windows | float64 | monotone | 1 | 8.27810 | 13.11610 | 0.631x |
| Windows | float64 | monotone | 4 | 8.22120 | 9.54330 | 0.861x |
| Windows | float64 | wide-range | 1 | 15.04980 | 21.62490 | 0.696x |
| Windows | float64 | wide-range | 4 | 14.23460 | 19.47190 | 0.731x |
