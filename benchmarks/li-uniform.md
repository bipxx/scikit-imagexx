# Li constant-image fast path

Measured source: `a2bd97a`; retained product fingerprint `4c0283fa95f728b4ba1dd11348be3b5ab63ca52d0eee103eeaf8100832d0ead0`. The subsequent merge changes tests, diagnostic tools and documentation, not that product fingerprint. Reference: scikit-image 0.26.0; NumPy 2.5.4.

The previous scan classified finite values, counted NaNs and computed bounds even when every pixel was equal. The retained fast path uses AVX2 equality scans for uint8/float32/float64 and returns the original first scalar for uniform data. A mismatch immediately continues through the existing algorithm. NaNs continue through the original filtering path; signed zeros and constant infinities retain their upstream handling. Portable builds use the same comparison semantics. Global optional worker limits and Python signatures are unchanged.

The fixed 25-case corpus is unchanged from the [initial Li checkpoint](li.md): seed 817, 512 by 513, layouts, constants, tiny inputs, tolerance/guess controls, continuous data and nonfinite inputs. Each OS ran one and four worker limits with seven randomized paired samples, 15 ms target batches and 20 ms cooldown. Both native packages pass parity before timing; fallback is disabled and original thread pools are fixed to one. [Exact current/baseline packages and raw results](https://github.com/bipxx/scikit-imagexx/actions/runs/39764952637) are archived unchanged.

Scoped validation: 2,914 passed on Windows and Linux/NumPy 1.24 in each SIMD mode; native normal and dynamic-team checks pass 2/2. Added 264 constant and late-mismatch parity cases cover SIMD boundaries, NaNs and long arrays. The old minimum-runtime CI gate also exposed an allocation-dependent upstream CIEDE2000 discrepancy, independently reproduced and addressed in the [angle-layout report](distance-layout.md); it does not change these Li numerical budgets or benchmark inputs.

| Host | One-thread original / current geometric mean | One-thread baseline / current geometric mean |
|---|---:|---:|
| Linux | 5.577x | 1.342x |
| Windows | 4.718x | 1.169x |

Constant cases improve 9.1-15.4x against the baseline on Linux and 1.9-10.5x on Windows with one worker. All three constant dtypes now beat scikit-image on both measured runners. This meets the fixed corpus geometric-mean metric on both OSes, so the candidate is retained.

Remaining regressions are retained: Linux float32 nonfinite is 0.859x baseline/current; several other Linux float32 workloads are 0.92-0.97x. Windows tiny float64 is 0.922x. These cases need follow-up profiling; the aggregate gain does not establish a per-case improvement.

| Host | Dtype | Case | Workers | Original ms | Current ms | Original / current | Baseline / current |
|---|---|---|---:|---:|---:|---:|---:|
| Linux | uint8 | quantized | 1 | 1.14903 | 0.09889 | 11.619x | 1.011x |
| Linux | uint8 | quantized | 4 | 1.15991 | 0.09657 | 12.012x | 1.018x |
| Linux | uint8 | fortran | 1 | 1.86876 | 0.46631 | 4.008x | 1.003x |
| Linux | uint8 | fortran | 4 | 1.87023 | 0.48953 | 3.820x | 1.006x |
| Linux | uint8 | strided | 1 | 0.86963 | 0.10946 | 7.944x | 0.988x |
| Linux | uint8 | strided | 4 | 0.83823 | 0.10765 | 7.787x | 0.997x |
| Linux | uint8 | constant | 1 | 0.15762 | 0.00939 | 16.793x | 15.416x |
| Linux | uint8 | constant | 4 | 0.15315 | 0.00868 | 17.647x | 11.050x |
| Linux | uint8 | tiny | 1 | 0.11605 | 0.00802 | 14.474x | 0.995x |
| Linux | uint8 | tiny | 4 | 0.12133 | 0.00886 | 13.702x | 0.968x |
| Linux | uint8 | explicit-tolerance | 1 | 1.18014 | 0.10006 | 11.795x | 1.001x |
| Linux | uint8 | explicit-tolerance | 4 | 1.15655 | 0.09907 | 11.675x | 0.990x |
| Linux | uint8 | numeric-guess | 1 | 0.83547 | 0.10478 | 7.974x | 1.019x |
| Linux | uint8 | numeric-guess | 4 | 0.84445 | 0.10714 | 7.882x | 1.015x |
| Linux | float32 | quantized | 1 | 23.31725 | 4.56685 | 5.106x | 0.922x |
| Linux | float32 | quantized | 4 | 23.69910 | 3.89566 | 6.083x | 0.972x |
| Linux | float32 | fortran | 1 | 24.39139 | 5.09088 | 4.791x | 0.959x |
| Linux | float32 | fortran | 4 | 24.73676 | 4.69646 | 5.267x | 0.943x |
| Linux | float32 | strided | 1 | 11.02936 | 2.28318 | 4.831x | 0.937x |
| Linux | float32 | strided | 4 | 11.11308 | 2.25897 | 4.920x | 0.958x |
| Linux | float32 | constant | 1 | 0.24378 | 0.02948 | 8.269x | 9.109x |
| Linux | float32 | constant | 4 | 0.24030 | 0.02769 | 8.679x | 9.767x |
| Linux | float32 | tiny | 1 | 0.08200 | 0.01462 | 5.608x | 1.015x |
| Linux | float32 | tiny | 4 | 0.07925 | 0.01590 | 4.983x | 0.923x |
| Linux | float32 | explicit-tolerance | 1 | 22.55899 | 1.70707 | 13.215x | 0.990x |
| Linux | float32 | explicit-tolerance | 4 | 22.66173 | 2.33865 | 9.690x | 1.006x |
| Linux | float32 | numeric-guess | 1 | 6.94649 | 3.58651 | 1.937x | 0.937x |
| Linux | float32 | numeric-guess | 4 | 7.09400 | 2.57210 | 2.758x | 0.939x |
| Linux | float32 | continuous | 1 | 39.96362 | 2.70838 | 14.756x | 0.972x |
| Linux | float32 | continuous | 4 | 39.96165 | 3.61011 | 11.069x | 0.995x |
| Linux | float32 | nonfinite | 1 | 22.46104 | 4.89270 | 4.591x | 0.859x |
| Linux | float32 | nonfinite | 4 | 22.30921 | 4.43713 | 5.028x | 0.958x |
| Linux | float64 | quantized | 1 | 25.45144 | 9.38912 | 2.711x | 1.108x |
| Linux | float64 | quantized | 4 | 25.69852 | 8.70396 | 2.953x | 1.065x |
| Linux | float64 | fortran | 1 | 26.81916 | 11.25298 | 2.383x | 1.059x |
| Linux | float64 | fortran | 4 | 26.90313 | 10.73407 | 2.506x | 1.067x |
| Linux | float64 | strided | 1 | 12.31816 | 3.81543 | 3.229x | 0.988x |
| Linux | float64 | strided | 4 | 12.36574 | 3.76271 | 3.286x | 0.989x |
| Linux | float64 | constant | 1 | 0.28316 | 0.04926 | 5.748x | 10.740x |
| Linux | float64 | constant | 4 | 0.30379 | 0.04987 | 6.092x | 10.602x |
| Linux | float64 | tiny | 1 | 0.10059 | 0.01540 | 6.532x | 0.962x |
| Linux | float64 | tiny | 4 | 0.10192 | 0.01502 | 6.784x | 1.004x |
| Linux | float64 | explicit-tolerance | 1 | 24.44662 | 6.21224 | 3.935x | 1.030x |
| Linux | float64 | explicit-tolerance | 4 | 24.48740 | 6.16875 | 3.970x | 1.073x |
| Linux | float64 | numeric-guess | 1 | 9.52618 | 8.24274 | 1.156x | 1.130x |
| Linux | float64 | numeric-guess | 4 | 9.69731 | 7.08016 | 1.370x | 1.117x |
| Linux | float64 | continuous | 1 | 44.42136 | 7.49736 | 5.925x | 1.095x |
| Linux | float64 | continuous | 4 | 44.75967 | 7.74381 | 5.780x | 1.168x |
| Linux | float64 | nonfinite | 1 | 24.73789 | 9.19386 | 2.691x | 1.105x |
| Linux | float64 | nonfinite | 4 | 24.77763 | 9.30134 | 2.664x | 1.085x |
| Windows | uint8 | quantized | 1 | 2.08735 | 0.09904 | 21.077x | 1.023x |
| Windows | uint8 | quantized | 4 | 2.12564 | 0.11237 | 18.916x | 1.027x |
| Windows | uint8 | fortran | 1 | 2.76705 | 0.47508 | 5.824x | 1.013x |
| Windows | uint8 | fortran | 4 | 2.75475 | 0.49818 | 5.530x | 1.007x |
| Windows | uint8 | strided | 1 | 1.38636 | 0.11758 | 11.791x | 1.003x |
| Windows | uint8 | strided | 4 | 1.39535 | 0.11790 | 11.836x | 0.996x |
| Windows | uint8 | constant | 1 | 0.24258 | 0.00982 | 24.707x | 10.456x |
| Windows | uint8 | constant | 4 | 0.25083 | 0.01031 | 24.319x | 10.239x |
| Windows | uint8 | tiny | 1 | 0.12923 | 0.00973 | 13.277x | 1.004x |
| Windows | uint8 | tiny | 4 | 0.12781 | 0.00946 | 13.516x | 1.087x |
| Windows | uint8 | explicit-tolerance | 1 | 2.11818 | 0.09984 | 21.215x | 1.013x |
| Windows | uint8 | explicit-tolerance | 4 | 2.12392 | 0.11440 | 18.566x | 0.997x |
| Windows | uint8 | numeric-guess | 1 | 1.64304 | 0.10746 | 15.289x | 0.985x |
| Windows | uint8 | numeric-guess | 4 | 1.70317 | 0.12602 | 13.515x | 1.026x |
| Windows | float32 | quantized | 1 | 24.88670 | 11.93480 | 2.085x | 1.021x |
| Windows | float32 | quantized | 4 | 24.87370 | 7.39260 | 3.365x | 1.016x |
| Windows | float32 | fortran | 1 | 25.89830 | 12.82500 | 2.019x | 1.030x |
| Windows | float32 | fortran | 4 | 25.73890 | 8.62240 | 2.985x | 1.008x |
| Windows | float32 | strided | 1 | 11.61180 | 5.83840 | 1.989x | 1.056x |
| Windows | float32 | strided | 4 | 11.62660 | 5.77525 | 2.013x | 1.022x |
| Windows | float32 | constant | 1 | 0.49097 | 0.03135 | 15.663x | 1.917x |
| Windows | float32 | constant | 4 | 0.48703 | 0.03190 | 15.267x | 1.939x |
| Windows | float32 | tiny | 1 | 0.08518 | 0.01452 | 5.865x | 0.999x |
| Windows | float32 | tiny | 4 | 0.08801 | 0.01509 | 5.832x | 1.032x |
| Windows | float32 | explicit-tolerance | 1 | 22.99210 | 2.57136 | 8.942x | 1.037x |
| Windows | float32 | explicit-tolerance | 4 | 23.04630 | 3.35937 | 6.860x | 0.985x |
| Windows | float32 | numeric-guess | 1 | 8.56500 | 10.64720 | 0.804x | 1.025x |
| Windows | float32 | numeric-guess | 4 | 8.50990 | 5.54595 | 1.534x | 1.032x |
| Windows | float32 | continuous | 1 | 40.76760 | 3.78907 | 10.759x | 1.061x |
| Windows | float32 | continuous | 4 | 40.60890 | 4.91935 | 8.255x | 1.014x |
| Windows | float32 | nonfinite | 1 | 24.11250 | 12.04850 | 2.001x | 1.031x |
| Windows | float32 | nonfinite | 4 | 23.65540 | 12.00360 | 1.971x | 1.023x |
| Windows | float64 | quantized | 1 | 28.55780 | 22.48920 | 1.270x | 1.002x |
| Windows | float64 | quantized | 4 | 28.84560 | 12.34260 | 2.337x | 1.006x |
| Windows | float64 | fortran | 1 | 30.23040 | 24.49720 | 1.234x | 0.994x |
| Windows | float64 | fortran | 4 | 30.17470 | 14.21490 | 2.123x | 0.992x |
| Windows | float64 | strided | 1 | 13.75700 | 11.95980 | 1.150x | 1.003x |
| Windows | float64 | strided | 4 | 13.46980 | 11.69540 | 1.152x | 0.994x |
| Windows | float64 | constant | 1 | 0.74657 | 0.05163 | 14.461x | 2.028x |
| Windows | float64 | constant | 4 | 0.73877 | 0.05259 | 14.046x | 1.980x |
| Windows | float64 | tiny | 1 | 0.10666 | 0.01572 | 6.784x | 0.922x |
| Windows | float64 | tiny | 4 | 0.10283 | 0.01475 | 6.971x | 0.968x |
| Windows | float64 | explicit-tolerance | 1 | 25.37450 | 4.68007 | 5.422x | 0.985x |
| Windows | float64 | explicit-tolerance | 4 | 25.78590 | 4.62173 | 5.579x | 1.007x |
| Windows | float64 | numeric-guess | 1 | 10.02390 | 19.98670 | 0.502x | 0.992x |
| Windows | float64 | numeric-guess | 4 | 10.24070 | 10.04010 | 1.020x | 0.981x |
| Windows | float64 | continuous | 1 | 47.61790 | 7.25920 | 6.560x | 1.010x |
| Windows | float64 | continuous | 4 | 48.08330 | 7.03640 | 6.834x | 0.997x |
| Windows | float64 | nonfinite | 1 | 26.53780 | 22.02120 | 1.205x | 0.987x |
| Windows | float64 | nonfinite | 4 | 27.23050 | 21.94200 | 1.241x | 0.995x |
