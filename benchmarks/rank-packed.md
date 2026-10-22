# Packed wide-range rank sorting

Retained measured source: `3ed8a8c`; product fingerprint `fb3cb8143c1dafe8b709c43ad67e9e59fe68350e82b3aed40a90c824d4953d3b`. The exact [lookup-table package](rank-tables.md) is the paired baseline. Reference: original scikit-image 0.26.0; NumPy 2.5.4.

The primary metric was declared before timing: one-worker latency across the five fixed wide-range dtype cases, with the other 30 cases retained as controls. Inputs, precision budgets, seven randomized paired samples, 15 ms target batches and 20 ms cooldown are unchanged. Both packages pass exact rank/value-byte and reconstruction checks before timing with fallback disabled and reference pools fixed to one. Allocation, normalization, dispatch and result destruction are included.

General sorting previously moved 16-byte key/position records and built a full array of group starts. Images with 32-bit positions now use an 8-byte record for relative keys that fit 32 bits, or an alignment-safe 12-byte record for full 64-bit keys. The first format uses four stable radix passes. A counted-segment pass allocates the representative vector exactly, then emits ranks without a full group-start array. Recorded segments remain valid if OpenMP changes its next team size. Larger position ranges keep the full-width representation. Public signatures, label widths and global worker limits are unchanged.

An initial candidate (`6597021`) improved wide-range cases but regressed several Linux dense-table controls; it was revised. Its new portable test also requested four workers from a build without OpenMP, so that test failed before sanitizer checks. The retained candidate keeps the portable test serial and places general sorting in a separate non-inlined helper. Its independent paired run meets the primary metric on both OSes and improves the whole corpus. This does not establish a direct first-candidate/retained timing ratio across runners.

[First-candidate exact packages and samples](https://github.com/bipxx/scikit-imagexx/actions/runs/39785049710) and [retained-candidate exact packages and samples](https://github.com/bipxx/scikit-imagexx/actions/runs/39786248642) are preserved. The first raw reports remain available as [Linux](rank-packed-first-linux.json) and [Windows](rank-packed-first-windows.json).

Retained validation: Windows passes all 6,946 scoped rank/API/metadata/histogram-matching cases in each SIMD mode; Linux/NumPy 1.24 passes 2,828 rank/histogram-matching cases in each mode. The 48 new cases cover the 32-bit relative-key limit, near-uint64-maximum offsets, label widths and repeated groups. Native release, dynamic teams, portable (OpenMP/AVX2 disabled), ASan/UBSan and dynamic-team sanitizer checks pass 2/2. Analytic native cases exercise both packed formats with repeated groups crossing worker segments. Full candidate CI [39786176567](https://github.com/bipxx/scikit-imagexx/actions/runs/39786176567) remains in progress; its minimum-runtime job passes.

| Host | Workers | Scope | Original / current | Previous native / current |
|---|---:|---|---:|---:|
| Linux | 1 | wide-range | 2.203x | 1.560x |
| Linux | 1 | whole corpus | 5.524x | 1.075x |
| Linux | 4 | wide-range | 2.556x | 1.486x |
| Linux | 4 | whole corpus | 5.328x | 1.069x |
| Windows | 1 | wide-range | 1.724x | 1.229x |
| Windows | 1 | whole corpus | 3.396x | 1.064x |
| Windows | 4 | wide-range | 2.186x | 1.219x |
| Windows | 4 | whole corpus | 3.841x | 1.065x |

One-worker wide-range float32 improves 2.50x on Linux and 1.61x on Windows over the exact table package; it measures 1.16x/1.10x original/current on those runners. General uint64 and float64 sorts still trail upstream: Linux wide-range cases are 0.887x/0.938x original/current and Windows 0.938x/0.809x. A Windows one-worker strided uint8 control is 0.947x previous/current. No retained Linux control is below 0.95x. These remain optimization targets, and every adverse sample is retained below. Absolute original/current ratios come from different GitHub runners; retention uses comparisons paired within the same job.

| Host | Dtype | Case | Workers | Original ms | Current ms | Original / current | Previous / current |
|---|---|---|---:|---:|---:|---:|---:|
| Linux | uint8 | quantized | 1 | 11.29040 | 0.99956 | 11.295x | 0.993x |
| Linux | uint8 | quantized | 4 | 12.20546 | 1.20100 | 10.163x | 1.023x |
| Linux | uint8 | fortran | 1 | 11.66692 | 1.51385 | 7.707x | 0.991x |
| Linux | uint8 | fortran | 4 | 11.60450 | 1.73185 | 6.701x | 0.999x |
| Linux | uint8 | strided | 1 | 5.48250 | 0.54302 | 10.096x | 1.001x |
| Linux | uint8 | strided | 4 | 5.49243 | 0.54432 | 10.090x | 0.995x |
| Linux | uint8 | constant | 1 | 3.17760 | 0.02073 | 153.319x | 0.997x |
| Linux | uint8 | constant | 4 | 3.21189 | 0.02118 | 151.624x | 0.974x |
| Linux | uint8 | tiny | 1 | 0.00875 | 0.00144 | 6.093x | 1.087x |
| Linux | uint8 | tiny | 4 | 0.00891 | 0.00163 | 5.459x | 1.008x |
| Linux | uint8 | monotone | 1 | 5.52308 | 1.03955 | 5.313x | 0.995x |
| Linux | uint8 | monotone | 4 | 5.64612 | 1.25089 | 4.514x | 1.002x |
| Linux | uint8 | wide-range | 1 | 11.46113 | 1.01381 | 11.305x | 0.990x |
| Linux | uint8 | wide-range | 4 | 11.38049 | 1.22298 | 9.306x | 0.989x |
| Linux | uint64 | quantized | 1 | 4.32659 | 1.04164 | 4.154x | 1.011x |
| Linux | uint64 | quantized | 4 | 4.38510 | 1.17393 | 3.735x | 1.002x |
| Linux | uint64 | fortran | 1 | 5.37441 | 2.67906 | 2.006x | 1.002x |
| Linux | uint64 | fortran | 4 | 5.07625 | 2.66227 | 1.907x | 1.012x |
| Linux | uint64 | strided | 1 | 1.98237 | 0.72864 | 2.721x | 1.004x |
| Linux | uint64 | strided | 4 | 1.95460 | 0.72591 | 2.693x | 1.000x |
| Linux | uint64 | constant | 1 | 2.02384 | 0.09821 | 20.607x | 0.998x |
| Linux | uint64 | constant | 4 | 2.24610 | 0.09923 | 22.635x | 0.999x |
| Linux | uint64 | tiny | 1 | 0.00921 | 0.00168 | 5.481x | 1.048x |
| Linux | uint64 | tiny | 4 | 0.00900 | 0.00157 | 5.749x | 1.079x |
| Linux | uint64 | monotone | 1 | 3.77687 | 1.08075 | 3.495x | 1.006x |
| Linux | uint64 | monotone | 4 | 3.83130 | 1.19966 | 3.194x | 0.984x |
| Linux | uint64 | wide-range | 1 | 7.17677 | 8.09387 | 0.887x | 1.497x |
| Linux | uint64 | wide-range | 4 | 8.30028 | 7.09812 | 1.169x | 1.417x |
| Linux | float16 | quantized | 1 | 14.30206 | 1.52605 | 9.372x | 0.990x |
| Linux | float16 | quantized | 4 | 16.34662 | 1.82974 | 8.934x | 1.006x |
| Linux | float16 | fortran | 1 | 16.47316 | 2.25025 | 7.321x | 0.995x |
| Linux | float16 | fortran | 4 | 14.61739 | 2.17005 | 6.736x | 1.011x |
| Linux | float16 | strided | 1 | 6.96901 | 0.84278 | 8.269x | 1.004x |
| Linux | float16 | strided | 4 | 6.93741 | 0.84613 | 8.199x | 1.005x |
| Linux | float16 | constant | 1 | 5.50424 | 0.12060 | 45.639x | 0.980x |
| Linux | float16 | constant | 4 | 5.05358 | 0.24232 | 20.855x | 1.002x |
| Linux | float16 | tiny | 1 | 0.00891 | 0.00194 | 4.586x | 1.109x |
| Linux | float16 | tiny | 4 | 0.00940 | 0.00201 | 4.669x | 1.053x |
| Linux | float16 | monotone | 1 | 12.79264 | 1.72367 | 7.422x | 1.008x |
| Linux | float16 | monotone | 4 | 11.51436 | 1.58701 | 7.255x | 0.999x |
| Linux | float16 | wide-range | 1 | 22.02066 | 4.63136 | 4.755x | 1.724x |
| Linux | float16 | wide-range | 4 | 22.12883 | 3.73881 | 5.919x | 1.693x |
| Linux | float32 | quantized | 1 | 3.84320 | 1.39485 | 2.755x | 0.999x |
| Linux | float32 | quantized | 4 | 3.84970 | 1.76383 | 2.183x | 0.980x |
| Linux | float32 | fortran | 1 | 3.99802 | 1.95345 | 2.047x | 0.990x |
| Linux | float32 | fortran | 4 | 4.03152 | 2.32486 | 1.734x | 0.995x |
| Linux | float32 | strided | 1 | 1.85742 | 0.78867 | 2.355x | 1.001x |
| Linux | float32 | strided | 4 | 1.86762 | 0.79165 | 2.359x | 0.999x |
| Linux | float32 | constant | 1 | 2.09423 | 0.04194 | 49.935x | 0.997x |
| Linux | float32 | constant | 4 | 2.11760 | 0.04204 | 50.366x | 1.062x |
| Linux | float32 | tiny | 1 | 0.01025 | 0.00177 | 5.797x | 1.077x |
| Linux | float32 | tiny | 4 | 0.00900 | 0.00156 | 5.773x | 1.091x |
| Linux | float32 | monotone | 1 | 3.91057 | 1.54441 | 2.532x | 0.985x |
| Linux | float32 | monotone | 4 | 3.89487 | 1.61163 | 2.417x | 1.002x |
| Linux | float32 | wide-range | 1 | 5.70941 | 4.91828 | 1.161x | 2.500x |
| Linux | float32 | wide-range | 4 | 6.46218 | 4.64251 | 1.392x | 2.128x |
| Linux | float64 | quantized | 1 | 5.18183 | 1.61801 | 3.203x | 1.006x |
| Linux | float64 | quantized | 4 | 5.31414 | 1.69508 | 3.135x | 1.034x |
| Linux | float64 | fortran | 1 | 5.99548 | 3.26253 | 1.838x | 0.996x |
| Linux | float64 | fortran | 4 | 6.14243 | 3.40041 | 1.806x | 0.998x |
| Linux | float64 | strided | 1 | 2.12659 | 0.96534 | 2.203x | 0.994x |
| Linux | float64 | strided | 4 | 2.14166 | 0.96260 | 2.225x | 0.994x |
| Linux | float64 | constant | 1 | 2.22567 | 0.09652 | 23.058x | 0.997x |
| Linux | float64 | constant | 4 | 2.20464 | 0.09765 | 22.577x | 0.988x |
| Linux | float64 | tiny | 1 | 0.01002 | 0.00155 | 6.452x | 1.114x |
| Linux | float64 | tiny | 4 | 0.01041 | 0.00153 | 6.786x | 1.086x |
| Linux | float64 | monotone | 1 | 4.37941 | 1.54134 | 2.841x | 0.965x |
| Linux | float64 | monotone | 4 | 4.39948 | 1.54401 | 2.849x | 0.999x |
| Linux | float64 | wide-range | 1 | 8.27741 | 8.82410 | 0.938x | 1.447x |
| Linux | float64 | wide-range | 4 | 8.54514 | 7.01579 | 1.218x | 1.434x |
| Windows | uint8 | quantized | 1 | 18.53910 | 3.39910 | 5.454x | 0.983x |
| Windows | uint8 | quantized | 4 | 18.51280 | 2.77930 | 6.661x | 0.982x |
| Windows | uint8 | fortran | 1 | 18.68160 | 3.91320 | 4.774x | 1.003x |
| Windows | uint8 | fortran | 4 | 18.61510 | 3.32880 | 5.592x | 0.995x |
| Windows | uint8 | strided | 1 | 8.95260 | 1.72286 | 5.196x | 0.947x |
| Windows | uint8 | strided | 4 | 9.07770 | 1.72884 | 5.251x | 0.954x |
| Windows | uint8 | constant | 1 | 6.88610 | 0.45965 | 14.981x | 1.012x |
| Windows | uint8 | constant | 4 | 6.89475 | 0.46366 | 14.870x | 1.003x |
| Windows | uint8 | tiny | 1 | 0.01553 | 0.00283 | 5.481x | 1.256x |
| Windows | uint8 | tiny | 4 | 0.01457 | 0.00280 | 5.200x | 1.237x |
| Windows | uint8 | monotone | 1 | 10.57070 | 3.32947 | 3.175x | 0.998x |
| Windows | uint8 | monotone | 4 | 10.69320 | 2.72790 | 3.920x | 0.995x |
| Windows | uint8 | wide-range | 1 | 18.52700 | 3.31350 | 5.591x | 0.992x |
| Windows | uint8 | wide-range | 4 | 18.37460 | 2.77617 | 6.619x | 0.981x |
| Windows | uint64 | quantized | 1 | 9.72230 | 3.41102 | 2.850x | 0.978x |
| Windows | uint64 | quantized | 4 | 9.83850 | 2.80114 | 3.512x | 0.966x |
| Windows | uint64 | fortran | 1 | 10.96080 | 5.56590 | 1.969x | 0.989x |
| Windows | uint64 | fortran | 4 | 10.87770 | 4.87855 | 2.230x | 0.977x |
| Windows | uint64 | strided | 1 | 5.03883 | 2.48896 | 2.024x | 0.992x |
| Windows | uint64 | strided | 4 | 5.06375 | 2.48810 | 2.035x | 0.995x |
| Windows | uint64 | constant | 1 | 5.79405 | 0.53760 | 10.778x | 0.996x |
| Windows | uint64 | constant | 4 | 5.78545 | 0.53314 | 10.852x | 1.004x |
| Windows | uint64 | tiny | 1 | 0.01554 | 0.00301 | 5.167x | 1.200x |
| Windows | uint64 | tiny | 4 | 0.01466 | 0.00290 | 5.051x | 1.219x |
| Windows | uint64 | monotone | 1 | 9.31260 | 3.45935 | 2.692x | 0.995x |
| Windows | uint64 | monotone | 4 | 9.11240 | 2.76928 | 3.291x | 1.005x |
| Windows | uint64 | wide-range | 1 | 17.85780 | 19.03100 | 0.938x | 1.166x |
| Windows | uint64 | wide-range | 4 | 18.11140 | 14.63970 | 1.237x | 1.233x |
| Windows | float16 | quantized | 1 | 25.89370 | 5.12440 | 5.053x | 1.028x |
| Windows | float16 | quantized | 4 | 26.09050 | 4.01903 | 6.492x | 1.022x |
| Windows | float16 | fortran | 1 | 26.82060 | 5.87135 | 4.568x | 1.020x |
| Windows | float16 | fortran | 4 | 26.66430 | 4.71563 | 5.654x | 1.028x |
| Windows | float16 | strided | 1 | 12.84230 | 2.97788 | 4.313x | 1.026x |
| Windows | float16 | strided | 4 | 12.67800 | 2.98252 | 4.251x | 1.018x |
| Windows | float16 | constant | 1 | 14.45340 | 0.90060 | 16.049x | 1.004x |
| Windows | float16 | constant | 4 | 14.03260 | 0.84041 | 16.697x | 0.999x |
| Windows | float16 | tiny | 1 | 0.01493 | 0.00394 | 3.786x | 1.211x |
| Windows | float16 | tiny | 4 | 0.01478 | 0.00387 | 3.824x | 1.231x |
| Windows | float16 | monotone | 1 | 23.99760 | 5.32640 | 4.505x | 1.031x |
| Windows | float16 | monotone | 4 | 23.87980 | 4.11000 | 5.810x | 1.023x |
| Windows | float16 | wide-range | 1 | 34.87400 | 10.64500 | 3.276x | 1.288x |
| Windows | float16 | wide-range | 4 | 34.89690 | 8.33890 | 4.185x | 1.186x |
| Windows | float32 | quantized | 1 | 8.27550 | 4.56500 | 1.813x | 1.032x |
| Windows | float32 | quantized | 4 | 8.22790 | 3.63255 | 2.265x | 1.025x |
| Windows | float32 | fortran | 1 | 9.05240 | 5.87925 | 1.540x | 1.022x |
| Windows | float32 | fortran | 4 | 8.94080 | 4.91570 | 1.819x | 1.018x |
| Windows | float32 | strided | 1 | 3.98520 | 2.82017 | 1.413x | 1.063x |
| Windows | float32 | strided | 4 | 3.82150 | 2.78118 | 1.374x | 1.029x |
| Windows | float32 | constant | 1 | 5.53675 | 0.49091 | 11.278x | 0.996x |
| Windows | float32 | constant | 4 | 5.44850 | 0.48791 | 11.167x | 1.000x |
| Windows | float32 | tiny | 1 | 0.01500 | 0.00307 | 4.887x | 1.218x |
| Windows | float32 | tiny | 4 | 0.01433 | 0.00294 | 4.868x | 1.274x |
| Windows | float32 | monotone | 1 | 8.33620 | 4.80515 | 1.735x | 1.007x |
| Windows | float32 | monotone | 4 | 8.36550 | 3.76808 | 2.220x | 1.026x |
| Windows | float32 | wide-range | 1 | 14.20740 | 12.96660 | 1.096x | 1.612x |
| Windows | float32 | wide-range | 4 | 14.13160 | 10.37000 | 1.363x | 1.541x |
| Windows | float64 | quantized | 1 | 9.21820 | 4.69693 | 1.963x | 1.016x |
| Windows | float64 | quantized | 4 | 9.14740 | 3.64905 | 2.507x | 1.022x |
| Windows | float64 | fortran | 1 | 10.12780 | 6.88000 | 1.472x | 1.024x |
| Windows | float64 | fortran | 4 | 10.25340 | 5.81765 | 1.762x | 1.026x |
| Windows | float64 | strided | 1 | 4.47417 | 3.22515 | 1.387x | 1.002x |
| Windows | float64 | strided | 4 | 4.49157 | 3.18600 | 1.410x | 1.026x |
| Windows | float64 | constant | 1 | 5.93835 | 0.53432 | 11.114x | 0.998x |
| Windows | float64 | constant | 4 | 5.94515 | 0.54127 | 10.984x | 0.997x |
| Windows | float64 | tiny | 1 | 0.01525 | 0.00310 | 4.923x | 1.192x |
| Windows | float64 | tiny | 4 | 0.01577 | 0.00301 | 5.237x | 1.214x |
| Windows | float64 | monotone | 1 | 9.07650 | 4.86550 | 1.865x | 1.004x |
| Windows | float64 | monotone | 4 | 9.00040 | 3.80900 | 2.363x | 1.013x |
| Windows | float64 | wide-range | 1 | 15.95930 | 19.73500 | 0.809x | 1.169x |
| Windows | float64 | wide-range | 4 | 16.00010 | 14.97130 | 1.069x | 1.219x |
