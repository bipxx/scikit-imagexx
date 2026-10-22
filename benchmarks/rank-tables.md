# Rank lookup tables

Measured source: `0e129c7`; product fingerprint `a8b505163140912c0ca5822a87bbcd9a9e94e8c67012e195dac1426be138097b`. The exact [retained uniform package](rank-uniform.md) is the paired baseline. Reference: original scikit-image 0.26.0; NumPy 2.5.4.

The previous implementation radix-sorted one entry per pixel even for images with few unique values. The retained path builds a direct table when the ordered-key span is below 65,536, or discovers up to 4,096 unique keys with a bounded sparse table. Only sparse unique keys are sorted. Dense rank assignment uses AVX2 gathers and unsigned narrowing; portable code uses identical table values. High-cardinality images retain radix sorting, and the prior exact-bit uniform fast path remains. The public signature, dtype promotion, output ownership and global optional worker limit are unchanged.

Validation: Windows passes 6,898 scoped rank/API/metadata/histogram-matching cases in scalar mode; the SIMD pass has 6,874 plus 24 separately passing final boundary cases. Linux/NumPy 1.24 passes all 2,780 rank/histogram-matching cases in each SIMD mode. The 156 new cases cover dense dispatch thresholds, the full unsigned-16 rank range, key spans near uint64 maximum, the 4,096-key hash capacity and its radix fallback. Native release, portable, ASan/UBSan and dynamic-team checks pass 2/2, including a dense unsigned-16 rank check. Full candidate CI [39782801321](https://github.com/bipxx/scikit-imagexx/actions/runs/39782801321) remains in progress; its minimum-runtime job passes.

The fixed 35-case corpus is unchanged: seed 3019, five dtypes, 512 by 513 images, quantized/Fortran/reversed-strided/constant/tiny/monotone/wide-range cases. One/four worker limits use seven randomized paired samples, 15 ms target batches and 20 ms cooldown. Allocation, normalization and dispatch are included. Both packages pass exact values/bytes and reconstruction checks before timing with fallback disabled; reference thread pools are fixed to one. [Exact packages, wheels and all raw samples](https://github.com/bipxx/scikit-imagexx/actions/runs/39782915248) are retained.

| Host | Workers | Original / current geometric mean | Previous native / current geometric mean |
|---|---:|---:|---:|
| Linux | 1 | 5.672x | 1.910x |
| Linux | 4 | 6.341x | 1.792x |
| Windows | 1 | 3.431x | 2.097x |
| Windows | 4 | 4.105x | 1.797x |

The fixed one-worker metric improves on both OSes, so the table path is retained. Linux quantized and monotone cases improve roughly 2.5-3.9x over the native baseline. Controls retain regressions: Linux four-worker half-precision wide-range is 0.947x previous/current; Windows four-worker tiny uint8 is 0.942x. Windows one-worker wide-range uint64/float32/float64 remain 0.535x/0.635x/0.512x original/current, and Linux float64 wide-range remains 0.914x. Their general sorting cost is an outstanding bottleneck; aggregate gains do not establish a per-case win.

| Host | Dtype | Case | Workers | Original ms | Current ms | Original / current | Previous / current |
|---|---|---|---:|---:|---:|---:|---:|
| Linux | uint8 | quantized | 1 | 13.70932 | 1.23334 | 11.116x | 3.727x |
| Linux | uint8 | quantized | 4 | 13.73914 | 0.89535 | 15.345x | 4.419x |
| Linux | uint8 | fortran | 1 | 13.38535 | 1.97519 | 6.777x | 2.686x |
| Linux | uint8 | fortran | 4 | 13.43797 | 1.70015 | 7.904x | 2.760x |
| Linux | uint8 | strided | 1 | 6.50481 | 0.65107 | 9.991x | 2.470x |
| Linux | uint8 | strided | 4 | 6.52097 | 0.65378 | 9.974x | 2.470x |
| Linux | uint8 | constant | 1 | 4.01624 | 0.03138 | 127.982x | 1.015x |
| Linux | uint8 | constant | 4 | 4.06358 | 0.03122 | 130.139x | 0.998x |
| Linux | uint8 | tiny | 1 | 0.02021 | 0.00344 | 5.871x | 0.953x |
| Linux | uint8 | tiny | 4 | 0.02021 | 0.00304 | 6.640x | 1.098x |
| Linux | uint8 | monotone | 1 | 6.69848 | 1.24061 | 5.399x | 3.884x |
| Linux | uint8 | monotone | 4 | 7.44823 | 0.89389 | 8.332x | 4.338x |
| Linux | uint8 | wide-range | 1 | 12.93126 | 1.24784 | 10.363x | 3.826x |
| Linux | uint8 | wide-range | 4 | 12.99990 | 0.92919 | 13.991x | 4.296x |
| Linux | uint64 | quantized | 1 | 6.13769 | 1.36635 | 4.492x | 2.548x |
| Linux | uint64 | quantized | 4 | 6.02600 | 0.90727 | 6.642x | 2.613x |
| Linux | uint64 | fortran | 1 | 7.16749 | 3.66083 | 1.958x | 1.946x |
| Linux | uint64 | fortran | 4 | 7.09599 | 3.37240 | 2.104x | 1.890x |
| Linux | uint64 | strided | 1 | 3.04799 | 0.83347 | 3.657x | 2.214x |
| Linux | uint64 | strided | 4 | 3.04776 | 0.83562 | 3.647x | 2.198x |
| Linux | uint64 | constant | 1 | 2.38997 | 0.07251 | 32.962x | 0.987x |
| Linux | uint64 | constant | 4 | 2.36597 | 0.06652 | 35.568x | 1.024x |
| Linux | uint64 | tiny | 1 | 0.02002 | 0.00326 | 6.140x | 1.012x |
| Linux | uint64 | tiny | 4 | 0.02030 | 0.00330 | 6.162x | 1.034x |
| Linux | uint64 | monotone | 1 | 5.13845 | 1.50217 | 3.421x | 3.335x |
| Linux | uint64 | monotone | 4 | 5.18208 | 1.26208 | 4.106x | 3.185x |
| Linux | uint64 | wide-range | 1 | 13.09970 | 11.38003 | 1.151x | 0.985x |
| Linux | uint64 | wide-range | 4 | 13.07859 | 8.86828 | 1.475x | 1.028x |
| Linux | float16 | quantized | 1 | 17.24769 | 1.85807 | 9.283x | 3.134x |
| Linux | float16 | quantized | 4 | 17.17963 | 1.63457 | 10.510x | 2.295x |
| Linux | float16 | fortran | 1 | 17.28323 | 2.72561 | 6.341x | 2.740x |
| Linux | float16 | fortran | 4 | 17.30316 | 2.62199 | 6.599x | 2.098x |
| Linux | float16 | strided | 1 | 8.22570 | 1.07697 | 7.638x | 2.458x |
| Linux | float16 | strided | 4 | 8.21317 | 1.08130 | 7.596x | 2.438x |
| Linux | float16 | constant | 1 | 6.39320 | 0.11068 | 57.765x | 0.988x |
| Linux | float16 | constant | 4 | 6.37908 | 0.12534 | 50.893x | 0.966x |
| Linux | float16 | tiny | 1 | 0.01990 | 0.00401 | 4.959x | 1.125x |
| Linux | float16 | tiny | 4 | 0.02089 | 0.00457 | 4.573x | 0.951x |
| Linux | float16 | monotone | 1 | 14.29653 | 1.93823 | 7.376x | 3.112x |
| Linux | float16 | monotone | 4 | 14.41678 | 1.72949 | 8.336x | 2.142x |
| Linux | float16 | wide-range | 1 | 25.92330 | 9.13670 | 2.837x | 0.982x |
| Linux | float16 | wide-range | 4 | 25.97022 | 6.99598 | 3.712x | 0.947x |
| Linux | float32 | quantized | 1 | 5.69715 | 1.80533 | 3.156x | 3.202x |
| Linux | float32 | quantized | 4 | 5.73585 | 1.56021 | 3.676x | 2.500x |
| Linux | float32 | fortran | 1 | 6.29134 | 3.11685 | 2.018x | 2.510x |
| Linux | float32 | fortran | 4 | 6.28817 | 3.09432 | 2.032x | 1.947x |
| Linux | float32 | strided | 1 | 2.70511 | 0.93718 | 2.886x | 2.836x |
| Linux | float32 | strided | 4 | 2.67672 | 0.95657 | 2.798x | 2.807x |
| Linux | float32 | constant | 1 | 2.42037 | 0.04517 | 53.578x | 1.007x |
| Linux | float32 | constant | 4 | 2.38382 | 0.04478 | 53.228x | 1.012x |
| Linux | float32 | tiny | 1 | 0.02049 | 0.00352 | 5.819x | 1.051x |
| Linux | float32 | tiny | 4 | 0.02137 | 0.00377 | 5.662x | 1.016x |
| Linux | float32 | monotone | 1 | 5.44347 | 1.95235 | 2.788x | 3.476x |
| Linux | float32 | monotone | 4 | 5.40500 | 1.75576 | 3.078x | 2.645x |
| Linux | float32 | wide-range | 1 | 10.75089 | 10.03841 | 1.071x | 1.042x |
| Linux | float32 | wide-range | 4 | 10.94909 | 8.65121 | 1.266x | 0.963x |
| Linux | float64 | quantized | 1 | 5.86603 | 1.83014 | 3.205x | 3.248x |
| Linux | float64 | quantized | 4 | 5.74071 | 1.52363 | 3.768x | 2.418x |
| Linux | float64 | fortran | 1 | 6.80506 | 4.01903 | 1.693x | 2.495x |
| Linux | float64 | fortran | 4 | 7.05455 | 3.93664 | 1.792x | 1.985x |
| Linux | float64 | strided | 1 | 2.80737 | 1.07347 | 2.615x | 3.224x |
| Linux | float64 | strided | 4 | 2.83115 | 1.07989 | 2.622x | 3.250x |
| Linux | float64 | constant | 1 | 2.44530 | 0.06846 | 35.718x | 1.018x |
| Linux | float64 | constant | 4 | 2.48150 | 0.06768 | 36.667x | 1.010x |
| Linux | float64 | tiny | 1 | 0.02030 | 0.00350 | 5.804x | 1.000x |
| Linux | float64 | tiny | 4 | 0.02064 | 0.00342 | 6.037x | 1.023x |
| Linux | float64 | monotone | 1 | 5.44721 | 2.27357 | 2.396x | 3.240x |
| Linux | float64 | monotone | 4 | 5.47928 | 1.82930 | 2.995x | 2.583x |
| Linux | float64 | wide-range | 1 | 12.16499 | 13.31670 | 0.914x | 0.998x |
| Linux | float64 | wide-range | 4 | 12.16866 | 9.23723 | 1.317x | 0.980x |
| Windows | uint8 | quantized | 1 | 14.33860 | 2.26073 | 6.342x | 4.195x |
| Windows | uint8 | quantized | 4 | 14.39040 | 1.68340 | 8.548x | 3.151x |
| Windows | uint8 | fortran | 1 | 14.73880 | 2.98798 | 4.933x | 3.386x |
| Windows | uint8 | fortran | 4 | 14.80180 | 2.46540 | 6.004x | 2.512x |
| Windows | uint8 | strided | 1 | 7.16115 | 1.16270 | 6.159x | 4.079x |
| Windows | uint8 | strided | 4 | 7.15670 | 1.17373 | 6.097x | 4.054x |
| Windows | uint8 | constant | 1 | 5.44915 | 0.26804 | 20.329x | 0.988x |
| Windows | uint8 | constant | 4 | 5.52360 | 0.26513 | 20.834x | 1.000x |
| Windows | uint8 | tiny | 1 | 0.02067 | 0.00417 | 4.956x | 0.952x |
| Windows | uint8 | tiny | 4 | 0.02025 | 0.00422 | 4.796x | 0.942x |
| Windows | uint8 | monotone | 1 | 8.16890 | 2.23940 | 3.648x | 4.130x |
| Windows | uint8 | monotone | 4 | 8.17750 | 1.68939 | 4.841x | 3.019x |
| Windows | uint8 | wide-range | 1 | 14.31090 | 2.26730 | 6.312x | 4.302x |
| Windows | uint8 | wide-range | 4 | 14.29850 | 1.66647 | 8.580x | 3.167x |
| Windows | uint64 | quantized | 1 | 7.96990 | 2.36618 | 3.368x | 4.146x |
| Windows | uint64 | quantized | 4 | 7.99840 | 1.73891 | 4.600x | 3.082x |
| Windows | uint64 | fortran | 1 | 9.85840 | 6.16150 | 1.600x | 2.165x |
| Windows | uint64 | fortran | 4 | 9.86050 | 5.64265 | 1.747x | 1.647x |
| Windows | uint64 | strided | 1 | 4.16197 | 1.70534 | 2.441x | 3.185x |
| Windows | uint64 | strided | 4 | 4.12630 | 1.71127 | 2.411x | 3.173x |
| Windows | uint64 | constant | 1 | 4.19013 | 0.29834 | 14.045x | 1.036x |
| Windows | uint64 | constant | 4 | 4.25253 | 0.30218 | 14.073x | 1.042x |
| Windows | uint64 | tiny | 1 | 0.02066 | 0.00402 | 5.139x | 1.001x |
| Windows | uint64 | tiny | 4 | 0.02049 | 0.00405 | 5.064x | 0.977x |
| Windows | uint64 | monotone | 1 | 6.88880 | 2.33198 | 2.954x | 4.064x |
| Windows | uint64 | monotone | 4 | 6.93375 | 1.73782 | 3.990x | 3.025x |
| Windows | uint64 | wide-range | 1 | 14.37500 | 26.85930 | 0.535x | 1.071x |
| Windows | uint64 | wide-range | 4 | 14.51200 | 16.45770 | 0.882x | 0.995x |
| Windows | float16 | quantized | 1 | 22.62330 | 4.10690 | 5.509x | 3.106x |
| Windows | float16 | quantized | 4 | 22.65980 | 3.00922 | 7.530x | 2.206x |
| Windows | float16 | fortran | 1 | 23.01440 | 4.98485 | 4.617x | 2.744x |
| Windows | float16 | fortran | 4 | 23.07110 | 3.96800 | 5.814x | 1.910x |
| Windows | float16 | strided | 1 | 11.21370 | 2.36433 | 4.743x | 2.859x |
| Windows | float16 | strided | 4 | 11.15070 | 2.43197 | 4.585x | 2.790x |
| Windows | float16 | constant | 1 | 12.42320 | 0.53610 | 23.173x | 1.000x |
| Windows | float16 | constant | 4 | 12.39120 | 0.51813 | 23.915x | 0.984x |
| Windows | float16 | tiny | 1 | 0.02005 | 0.00543 | 3.693x | 0.999x |
| Windows | float16 | tiny | 4 | 0.01999 | 0.00522 | 3.830x | 0.992x |
| Windows | float16 | monotone | 1 | 20.75280 | 4.16450 | 4.983x | 3.064x |
| Windows | float16 | monotone | 4 | 20.64690 | 3.09837 | 6.664x | 2.143x |
| Windows | float16 | wide-range | 1 | 29.46820 | 14.69250 | 2.006x | 1.024x |
| Windows | float16 | wide-range | 4 | 29.47730 | 8.46110 | 3.484x | 1.009x |
| Windows | float32 | quantized | 1 | 7.35320 | 3.77850 | 1.946x | 3.631x |
| Windows | float32 | quantized | 4 | 7.25195 | 2.70826 | 2.678x | 2.545x |
| Windows | float32 | fortran | 1 | 8.32500 | 5.64385 | 1.475x | 2.761x |
| Windows | float32 | fortran | 4 | 8.42750 | 4.63383 | 1.819x | 1.953x |
| Windows | float32 | strided | 1 | 3.43065 | 2.22478 | 1.542x | 3.270x |
| Windows | float32 | strided | 4 | 3.33853 | 2.24435 | 1.488x | 3.255x |
| Windows | float32 | constant | 1 | 4.11727 | 0.29201 | 14.100x | 0.974x |
| Windows | float32 | constant | 4 | 3.98943 | 0.27515 | 14.499x | 1.013x |
| Windows | float32 | tiny | 1 | 0.02029 | 0.00425 | 4.772x | 0.980x |
| Windows | float32 | tiny | 4 | 0.02047 | 0.00424 | 4.824x | 1.029x |
| Windows | float32 | monotone | 1 | 6.78260 | 3.92157 | 1.730x | 3.228x |
| Windows | float32 | monotone | 4 | 6.79635 | 2.83270 | 2.399x | 2.326x |
| Windows | float32 | wide-range | 1 | 12.73570 | 20.06700 | 0.635x | 1.031x |
| Windows | float32 | wide-range | 4 | 12.75520 | 13.01590 | 0.980x | 1.004x |
| Windows | float64 | quantized | 1 | 7.96990 | 3.93420 | 2.026x | 4.603x |
| Windows | float64 | quantized | 4 | 7.85760 | 2.81340 | 2.793x | 3.145x |
| Windows | float64 | fortran | 1 | 9.78690 | 8.11410 | 1.206x | 2.805x |
| Windows | float64 | fortran | 4 | 9.95270 | 7.06570 | 1.409x | 1.850x |
| Windows | float64 | strided | 1 | 3.90380 | 2.49304 | 1.566x | 3.984x |
| Windows | float64 | strided | 4 | 3.85210 | 2.52366 | 1.526x | 3.935x |
| Windows | float64 | constant | 1 | 4.31357 | 0.30351 | 14.212x | 1.008x |
| Windows | float64 | constant | 4 | 4.30703 | 0.30094 | 14.312x | 1.022x |
| Windows | float64 | tiny | 1 | 0.02033 | 0.00419 | 4.853x | 0.971x |
| Windows | float64 | tiny | 4 | 0.02041 | 0.00411 | 4.961x | 1.007x |
| Windows | float64 | monotone | 1 | 7.15375 | 3.96223 | 1.805x | 3.555x |
| Windows | float64 | monotone | 4 | 7.10785 | 2.85018 | 2.494x | 2.425x |
| Windows | float64 | wide-range | 1 | 13.72830 | 26.83650 | 0.512x | 1.022x |
| Windows | float64 | wide-range | 4 | 13.56170 | 15.77290 | 0.860x | 1.025x |
