# Rank ordering uniform fast path

Measured source: `8b53a66`; product fingerprint `ddd19fc58f20139478690f1dae654f0b19594f4cffcb1883a460612b1c66c02e`. Reference: original scikit-image 0.26.0, NumPy 2.5.4. This is a partial native port, not full scikit-image parity.

The initial algorithm built and radix-sorted entries even for constant images. The retained fast path compares exact storage with SIMD and emits zero labels without sort buffers. Portable builds compare the same bits. Mixed signed zeros do not enter the uniform path; NaNs and noncanonical boolean storage require explicit upstream fallback because NumPy unstable sorting can affect rank or representative bytes. The latter guard corrects a representative-byte gap discovered after the initial baseline.

Validation: Windows passes 6,742 scoped rank/API/metadata/histogram-matching cases in each SIMD mode; Linux/NumPy 1.24 passes 2,624 rank/histogram-matching cases in each mode. Added 394 cases exercise uniform inputs, late mismatches, SIMD tails and noncanonical bool fallback. Native release, portable, ASan/UBSan and dynamic-team checks pass 2/2. Full CI [39780981659](https://github.com/bipxx/scikit-imagexx/actions/runs/39780981659) passes all three jobs: Linux 48,453 tests and Windows 48,451 plus two skips in each SIMD mode, with normal/dynamic/portable native checks passing 2/2 on both OSes.

The fixed 35-case corpus uses seed 3019; uint8/uint64/float16/float32/float64; 512 by 513 images; quantized, Fortran, reversed-strided, constant, tiny, monotone and wide-range workloads. One/four worker limits use seven randomized samples, 15 ms target batches and 20 ms cooldown. Every call includes allocation, normalization, sorting and Python dispatch. Exact values/bytes and reconstruction are checked before timing with fallback disabled and reference pools fixed to one. [Exact packages, wheels and all samples](https://github.com/bipxx/scikit-imagexx/actions/runs/39781023090) are archived.

| Host | Workers | Original / current geometric mean | Initial / current geometric mean |
|---|---:|---:|---:|
| Linux | 1 | 2.545x | 1.912x |
| Linux | 4 | 2.925x | 1.910x |
| Windows | 1 | 1.671x | 1.583x |
| Windows | 4 | 2.281x | 1.464x |

Speedup is original/current; below 1 is slower. Continuous/wide-range floating sorts and Fortran normalization remain bottlenecks; aggregates do not establish a win for every case. One-worker constant-image gains against the native baseline are 56-184x on Linux and 16-30x on Windows. Retained regressions include Linux tiny uint8 (0.919x initial/current), tiny float64 (0.926x), four-worker strided uint8 (0.938x), Windows tiny uint64 (0.912x) and four-worker half-precision wide-range input (0.940x). All adverse results are retained below.

| Host | Dtype | Case | Workers | Original ms | Current ms | Original / current | Initial / current |
|---|---|---|---:|---:|---:|---:|---:|
| Linux | uint8 | quantized | 1 | 13.90491 | 6.87591 | 2.022x | 1.029x |
| Linux | uint8 | quantized | 4 | 13.91635 | 6.02891 | 2.308x | 1.023x |
| Linux | uint8 | fortran | 1 | 13.50127 | 7.98363 | 1.691x | 0.976x |
| Linux | uint8 | fortran | 4 | 13.52556 | 7.63858 | 1.771x | 0.985x |
| Linux | uint8 | strided | 1 | 6.63829 | 1.59894 | 4.152x | 0.996x |
| Linux | uint8 | strided | 4 | 6.51286 | 1.69311 | 3.847x | 0.938x |
| Linux | uint8 | constant | 1 | 4.05342 | 0.03223 | 125.763x | 183.628x |
| Linux | uint8 | constant | 4 | 4.07777 | 0.03150 | 129.461x | 176.052x |
| Linux | uint8 | tiny | 1 | 0.02172 | 0.00330 | 6.572x | 0.919x |
| Linux | uint8 | tiny | 4 | 0.02064 | 0.00305 | 6.762x | 0.986x |
| Linux | uint8 | monotone | 1 | 6.73856 | 6.74837 | 0.999x | 1.005x |
| Linux | uint8 | monotone | 4 | 7.49171 | 5.93967 | 1.261x | 1.022x |
| Linux | uint8 | wide-range | 1 | 13.01580 | 7.25052 | 1.795x | 1.016x |
| Linux | uint8 | wide-range | 4 | 13.08855 | 6.33814 | 2.065x | 1.001x |
| Linux | uint64 | quantized | 1 | 6.16915 | 3.59392 | 1.717x | 0.994x |
| Linux | uint64 | quantized | 4 | 6.12840 | 2.21746 | 2.764x | 1.026x |
| Linux | uint64 | fortran | 1 | 7.21605 | 9.70441 | 0.744x | 0.988x |
| Linux | uint64 | fortran | 4 | 7.06564 | 8.86582 | 0.797x | 1.005x |
| Linux | uint64 | strided | 1 | 3.09190 | 1.87237 | 1.651x | 0.986x |
| Linux | uint64 | strided | 4 | 3.05923 | 1.83089 | 1.671x | 0.998x |
| Linux | uint64 | constant | 1 | 2.46546 | 0.07568 | 32.579x | 82.834x |
| Linux | uint64 | constant | 4 | 2.48918 | 0.07116 | 34.982x | 83.178x |
| Linux | uint64 | tiny | 1 | 0.02109 | 0.00346 | 6.088x | 0.962x |
| Linux | uint64 | tiny | 4 | 0.02114 | 0.00341 | 6.194x | 1.023x |
| Linux | uint64 | monotone | 1 | 5.38747 | 7.60791 | 0.708x | 0.970x |
| Linux | uint64 | monotone | 4 | 5.28012 | 6.29260 | 0.839x | 1.013x |
| Linux | uint64 | wide-range | 1 | 13.86092 | 15.23119 | 0.910x | 1.016x |
| Linux | uint64 | wide-range | 4 | 13.54771 | 10.68966 | 1.267x | 1.132x |
| Linux | float16 | quantized | 1 | 17.65713 | 8.90228 | 1.983x | 1.062x |
| Linux | float16 | quantized | 4 | 17.60751 | 6.79472 | 2.591x | 1.001x |
| Linux | float16 | fortran | 1 | 17.40061 | 10.06325 | 1.729x | 1.000x |
| Linux | float16 | fortran | 4 | 17.43762 | 8.16122 | 2.137x | 1.000x |
| Linux | float16 | strided | 1 | 8.18419 | 2.67660 | 3.058x | 0.987x |
| Linux | float16 | strided | 4 | 8.22164 | 2.67182 | 3.077x | 0.994x |
| Linux | float16 | constant | 1 | 6.69546 | 0.11578 | 57.831x | 56.074x |
| Linux | float16 | constant | 4 | 6.67666 | 0.12285 | 54.348x | 49.135x |
| Linux | float16 | tiny | 1 | 0.02183 | 0.00417 | 5.235x | 0.956x |
| Linux | float16 | tiny | 4 | 0.02046 | 0.00413 | 4.951x | 0.971x |
| Linux | float16 | monotone | 1 | 14.51337 | 8.83480 | 1.643x | 1.032x |
| Linux | float16 | monotone | 4 | 14.55199 | 6.72939 | 2.162x | 1.027x |
| Linux | float16 | wide-range | 1 | 26.13865 | 11.23072 | 2.327x | 0.995x |
| Linux | float16 | wide-range | 4 | 26.24146 | 9.20445 | 2.851x | 0.993x |
| Linux | float32 | quantized | 1 | 5.89281 | 9.44428 | 0.624x | 0.985x |
| Linux | float32 | quantized | 4 | 5.73878 | 7.42148 | 0.773x | 0.952x |
| Linux | float32 | fortran | 1 | 6.59361 | 11.00496 | 0.599x | 0.997x |
| Linux | float32 | fortran | 4 | 6.45285 | 9.07990 | 0.711x | 0.983x |
| Linux | float32 | strided | 1 | 2.74964 | 2.73060 | 1.007x | 0.998x |
| Linux | float32 | strided | 4 | 2.75175 | 2.72121 | 1.011x | 1.002x |
| Linux | float32 | constant | 1 | 2.41631 | 0.05166 | 46.778x | 125.133x |
| Linux | float32 | constant | 4 | 2.41718 | 0.05086 | 47.526x | 120.593x |
| Linux | float32 | tiny | 1 | 0.02102 | 0.00339 | 6.200x | 1.003x |
| Linux | float32 | tiny | 4 | 0.02066 | 0.00336 | 6.139x | 0.981x |
| Linux | float32 | monotone | 1 | 5.44522 | 9.21740 | 0.591x | 0.986x |
| Linux | float32 | monotone | 4 | 5.42242 | 6.82570 | 0.794x | 1.001x |
| Linux | float32 | wide-range | 1 | 10.85174 | 12.62332 | 0.860x | 1.008x |
| Linux | float32 | wide-range | 4 | 10.98508 | 11.01887 | 0.997x | 1.005x |
| Linux | float64 | quantized | 1 | 5.90792 | 6.00212 | 0.984x | 0.992x |
| Linux | float64 | quantized | 4 | 5.75902 | 3.52951 | 1.632x | 0.965x |
| Linux | float64 | fortran | 1 | 6.84786 | 12.62316 | 0.542x | 1.002x |
| Linux | float64 | fortran | 4 | 6.84146 | 10.36812 | 0.660x | 0.990x |
| Linux | float64 | strided | 1 | 2.92087 | 3.62798 | 0.805x | 0.971x |
| Linux | float64 | strided | 4 | 2.91584 | 3.58731 | 0.813x | 1.006x |
| Linux | float64 | constant | 1 | 2.62479 | 0.07381 | 35.563x | 87.175x |
| Linux | float64 | constant | 4 | 2.69553 | 0.07694 | 35.033x | 78.283x |
| Linux | float64 | tiny | 1 | 0.02088 | 0.00359 | 5.812x | 0.926x |
| Linux | float64 | tiny | 4 | 0.02340 | 0.00337 | 6.951x | 0.984x |
| Linux | float64 | monotone | 1 | 5.53679 | 9.90778 | 0.559x | 0.950x |
| Linux | float64 | monotone | 4 | 5.57119 | 7.03251 | 0.792x | 1.040x |
| Linux | float64 | wide-range | 1 | 12.62377 | 15.08889 | 0.837x | 1.043x |
| Linux | float64 | wide-range | 4 | 12.84190 | 11.65197 | 1.102x | 0.970x |
| Windows | uint8 | quantized | 1 | 14.55790 | 9.45890 | 1.539x | 0.986x |
| Windows | uint8 | quantized | 4 | 14.58530 | 5.43340 | 2.684x | 0.995x |
| Windows | uint8 | fortran | 1 | 14.85830 | 10.07170 | 1.475x | 1.015x |
| Windows | uint8 | fortran | 4 | 14.75210 | 6.15550 | 2.397x | 1.001x |
| Windows | uint8 | strided | 1 | 7.23380 | 4.74770 | 1.524x | 1.001x |
| Windows | uint8 | strided | 4 | 7.28955 | 4.75093 | 1.534x | 0.994x |
| Windows | uint8 | constant | 1 | 5.42830 | 0.25233 | 21.512x | 30.225x |
| Windows | uint8 | constant | 4 | 5.49120 | 0.25346 | 21.665x | 17.612x |
| Windows | uint8 | tiny | 1 | 0.02213 | 0.00432 | 5.118x | 0.998x |
| Windows | uint8 | tiny | 4 | 0.02100 | 0.00439 | 4.784x | 1.026x |
| Windows | uint8 | monotone | 1 | 8.23800 | 9.07960 | 0.907x | 0.996x |
| Windows | uint8 | monotone | 4 | 8.24900 | 5.15855 | 1.599x | 0.996x |
| Windows | uint8 | wide-range | 1 | 14.39960 | 9.27220 | 1.553x | 0.998x |
| Windows | uint8 | wide-range | 4 | 14.48230 | 5.45900 | 2.653x | 0.992x |
| Windows | uint64 | quantized | 1 | 8.65760 | 9.59150 | 0.903x | 0.998x |
| Windows | uint64 | quantized | 4 | 8.23140 | 5.49885 | 1.497x | 0.997x |
| Windows | uint64 | fortran | 1 | 10.72050 | 13.84110 | 0.775x | 0.985x |
| Windows | uint64 | fortran | 4 | 10.31530 | 9.79990 | 1.053x | 0.998x |
| Windows | uint64 | strided | 1 | 4.23653 | 5.29115 | 0.801x | 0.996x |
| Windows | uint64 | strided | 4 | 4.30123 | 5.30930 | 0.810x | 0.990x |
| Windows | uint64 | constant | 1 | 4.32050 | 0.30881 | 13.991x | 25.484x |
| Windows | uint64 | constant | 4 | 4.23483 | 0.30171 | 14.036x | 15.091x |
| Windows | uint64 | tiny | 1 | 0.02048 | 0.00453 | 4.524x | 0.912x |
| Windows | uint64 | tiny | 4 | 0.02104 | 0.00433 | 4.859x | 0.963x |
| Windows | uint64 | monotone | 1 | 6.90875 | 9.38460 | 0.736x | 0.993x |
| Windows | uint64 | monotone | 4 | 6.89250 | 5.23740 | 1.316x | 1.007x |
| Windows | uint64 | wide-range | 1 | 15.35010 | 26.68170 | 0.575x | 0.996x |
| Windows | uint64 | wide-range | 4 | 14.83440 | 16.43470 | 0.903x | 0.993x |
| Windows | float16 | quantized | 1 | 22.71170 | 12.84450 | 1.768x | 0.996x |
| Windows | float16 | quantized | 4 | 22.72960 | 6.78140 | 3.352x | 1.008x |
| Windows | float16 | fortran | 1 | 23.32010 | 13.72260 | 1.699x | 1.006x |
| Windows | float16 | fortran | 4 | 23.27840 | 7.80040 | 2.984x | 0.998x |
| Windows | float16 | strided | 1 | 11.29080 | 6.82605 | 1.654x | 0.997x |
| Windows | float16 | strided | 4 | 11.36740 | 6.83470 | 1.663x | 1.024x |
| Windows | float16 | constant | 1 | 12.58690 | 0.54046 | 23.289x | 15.990x |
| Windows | float16 | constant | 4 | 12.61120 | 0.57743 | 21.840x | 8.995x |
| Windows | float16 | tiny | 1 | 0.02049 | 0.00550 | 3.725x | 1.028x |
| Windows | float16 | tiny | 4 | 0.02014 | 0.00528 | 3.811x | 1.053x |
| Windows | float16 | monotone | 1 | 20.84320 | 12.74870 | 1.635x | 1.000x |
| Windows | float16 | monotone | 4 | 20.80580 | 6.85485 | 3.035x | 0.989x |
| Windows | float16 | wide-range | 1 | 29.55770 | 15.08010 | 1.960x | 0.997x |
| Windows | float16 | wide-range | 4 | 29.68510 | 8.92440 | 3.326x | 0.940x |
| Windows | float32 | quantized | 1 | 7.65330 | 13.82490 | 0.554x | 1.014x |
| Windows | float32 | quantized | 4 | 7.51350 | 7.00480 | 1.073x | 1.005x |
| Windows | float32 | fortran | 1 | 8.48040 | 15.52960 | 0.546x | 1.007x |
| Windows | float32 | fortran | 4 | 8.53610 | 8.94740 | 0.954x | 1.000x |
| Windows | float32 | strided | 1 | 3.63453 | 7.56440 | 0.480x | 1.002x |
| Windows | float32 | strided | 4 | 3.66727 | 7.46820 | 0.491x | 1.013x |
| Windows | float32 | constant | 1 | 4.16213 | 0.27984 | 14.873x | 30.207x |
| Windows | float32 | constant | 4 | 4.13990 | 0.27645 | 14.975x | 17.541x |
| Windows | float32 | tiny | 1 | 0.02106 | 0.00446 | 4.718x | 1.053x |
| Windows | float32 | tiny | 4 | 0.02107 | 0.00501 | 4.208x | 0.990x |
| Windows | float32 | monotone | 1 | 7.03770 | 12.61580 | 0.558x | 1.009x |
| Windows | float32 | monotone | 4 | 6.90940 | 6.65010 | 1.039x | 0.993x |
| Windows | float32 | wide-range | 1 | 13.47060 | 20.59570 | 0.654x | 1.004x |
| Windows | float32 | wide-range | 4 | 13.39360 | 13.64780 | 0.981x | 0.996x |
| Windows | float64 | quantized | 1 | 8.41430 | 18.38190 | 0.458x | 0.996x |
| Windows | float64 | quantized | 4 | 8.12280 | 8.96440 | 0.906x | 1.000x |
| Windows | float64 | fortran | 1 | 10.18460 | 22.10870 | 0.461x | 0.998x |
| Windows | float64 | fortran | 4 | 10.03090 | 12.96700 | 0.774x | 0.993x |
| Windows | float64 | strided | 1 | 3.98637 | 10.00490 | 0.398x | 0.996x |
| Windows | float64 | strided | 4 | 4.05823 | 9.89840 | 0.410x | 0.992x |
| Windows | float64 | constant | 1 | 4.32340 | 0.30334 | 14.253x | 27.823x |
| Windows | float64 | constant | 4 | 4.37953 | 0.30862 | 14.191x | 15.765x |
| Windows | float64 | tiny | 1 | 0.02164 | 0.00456 | 4.751x | 0.957x |
| Windows | float64 | tiny | 4 | 0.02005 | 0.00431 | 4.655x | 1.011x |
| Windows | float64 | monotone | 1 | 7.21695 | 13.59570 | 0.531x | 0.999x |
| Windows | float64 | monotone | 4 | 7.17775 | 6.95335 | 1.032x | 0.991x |
| Windows | float64 | wide-range | 1 | 13.90270 | 27.08030 | 0.513x | 0.994x |
| Windows | float64 | wide-range | 4 | 14.05350 | 16.37690 | 0.858x | 1.004x |
