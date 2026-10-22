# Retained hysteresis row-span helper

The one-worker 2-D path floods horizontal runs instead of queueing every seed/pixel. An explicitly out-of-line helper keeps the larger traversal separate from count/fast-path dispatch. Face connectivity, label-zero behavior and optional global workers are preserved. Other ranks and the parallel component path retain their original traversal.

[Successful paired Linux/Windows run](https://github.com/bipxx/scikit-imagexx/actions/runs/39799426035) measured `d11d3ba`, product SHA256 `1a4b0ff8e6c55a64f627cf4bfc2e7eeca2ef6216bd8eaafba2fcef02f3d2bb26`, against scikit-image 0.26.0 and exact initial native package `fa6a453`. Both packages use NumPy 2.5.4, pass every corpus case exactly and disable fallback. Exact wheels and extracted current/baseline packages are preserved in the run artifacts with checked binary/source hashes.

The fixed 32-case corpus uses one/four workers, seven randomized samples, size 512 and seed 1934, with one reference thread. Timings include allocation, conversion and dispatch. All raw samples and slower cases are retained in JSON and the complete table below.

One-worker geometric means versus the exact initial native package improve **1.150x Linux / 1.136x Windows**; versus upstream they are **2.104x / 1.828x**. Snake masks improve 1.61–1.67x on Linux and 1.51–1.77x on Windows against the initial package. The [earlier inline trial](hysteresis-spans-trial.md) is preserved separately and is not the retained source. Its equal-threshold regression was larger.

Residual controls remain adverse: Windows float32 equal is **0.918x** baseline/current, Linux float64 NaN-background **0.950x**, and Linux float32 NaN-background **0.954x**. Several random, strided and Fortran cases still trail upstream. These are open optimization targets, not hidden by the aggregate. Layout normalization and irregular component traversal remain costs.

Validation: the paired jobs pass dedicated hysteresis/API parity on both OSes, and the minimum-NumPy full CI job passes. The exact measured Windows wheel passes 6,308 local scoped cases in each SIMD mode. The earlier retained compatibility source full CI passes all three jobs: Linux 50,845 tests and Windows 50,843 plus two skips in each SIMD mode, with normal/dynamic/portable native checks passing 2/2 on both OSes. Follow-up helper full CI remains in progress. The earlier span implementation also passed local minimum-runtime, native portable and ASan/UBSan checks; those earlier results are not a substitute for exact-source follow-up CI.

| OS | dtype | layout | workers | upstream ms | current ms | upstream/current | baseline/current |
|---|---|---|---:|---:|---:|---:|---:|
| Linux | uint8 | random | 1 | 5.31973 | 5.10971 | 1.041x | 1.251x |
| Linux | uint8 | random | 4 | 5.26106 | 3.66054 | 1.437x | 0.983x |
| Linux | uint8 | fortran | 1 | 7.53659 | 8.08033 | 0.933x | 0.988x |
| Linux | uint8 | fortran | 4 | 8.22786 | 6.02297 | 1.366x | 0.827x |
| Linux | uint8 | strided | 1 | 2.72834 | 2.65493 | 1.028x | 1.259x |
| Linux | uint8 | strided | 4 | 2.79989 | 2.68229 | 1.044x | 1.259x |
| Linux | uint8 | all-weak | 1 | 4.13183 | 0.99316 | 4.160x | 1.008x |
| Linux | uint8 | all-weak | 4 | 4.21859 | 0.79740 | 5.290x | 0.970x |
| Linux | uint8 | no-seeds | 1 | 5.16804 | 2.06945 | 2.497x | 0.973x |
| Linux | uint8 | no-seeds | 4 | 5.18642 | 1.15605 | 4.486x | 1.018x |
| Linux | uint8 | equal | 1 | 7.28346 | 2.05251 | 3.549x | 0.965x |
| Linux | uint8 | equal | 4 | 7.47904 | 1.19133 | 6.278x | 0.871x |
| Linux | uint8 | threshold-fields | 1 | 5.17516 | 4.86349 | 1.064x | 1.266x |
| Linux | uint8 | threshold-fields | 4 | 5.22898 | 3.54872 | 1.473x | 0.956x |
| Linux | uint8 | snake | 1 | 3.76295 | 0.97361 | 3.865x | 1.646x |
| Linux | uint8 | snake | 4 | 3.79663 | 1.62093 | 2.342x | 0.951x |
| Linux | uint8 | checkerboard | 1 | 7.21830 | 0.89439 | 8.071x | 1.161x |
| Linux | uint8 | checkerboard | 4 | 7.37653 | 1.48212 | 4.977x | 0.928x |
| Linux | uint8 | tiny | 1 | 0.11372 | 0.10228 | 1.112x | 0.991x |
| Linux | uint8 | tiny | 4 | 0.11185 | 0.10118 | 1.105x | 1.001x |
| Linux | float32 | random | 1 | 5.13296 | 5.05886 | 1.015x | 1.269x |
| Linux | float32 | random | 4 | 5.09310 | 3.56328 | 1.429x | 1.005x |
| Linux | float32 | fortran | 1 | 6.04955 | 6.58020 | 0.919x | 1.223x |
| Linux | float32 | fortran | 4 | 6.20706 | 5.38446 | 1.153x | 1.027x |
| Linux | float32 | strided | 1 | 2.76562 | 2.77100 | 0.998x | 1.240x |
| Linux | float32 | strided | 4 | 2.75832 | 2.74354 | 1.005x | 1.269x |
| Linux | float32 | all-weak | 1 | 4.20166 | 1.14968 | 3.655x | 1.014x |
| Linux | float32 | all-weak | 4 | 4.12461 | 0.83716 | 4.927x | 0.979x |
| Linux | float32 | no-seeds | 1 | 5.32935 | 2.10991 | 2.526x | 0.994x |
| Linux | float32 | no-seeds | 4 | 5.17251 | 1.08194 | 4.781x | 0.976x |
| Linux | float32 | equal | 1 | 7.31393 | 2.99409 | 2.443x | 0.969x |
| Linux | float32 | equal | 4 | 7.28903 | 3.10821 | 2.345x | 0.882x |
| Linux | float32 | threshold-fields | 1 | 5.34680 | 5.07274 | 1.054x | 1.258x |
| Linux | float32 | threshold-fields | 4 | 7.54065 | 4.21052 | 1.791x | 0.868x |
| Linux | float32 | snake | 1 | 3.81239 | 1.11252 | 3.427x | 1.614x |
| Linux | float32 | snake | 4 | 3.81939 | 1.67653 | 2.278x | 0.943x |
| Linux | float32 | checkerboard | 1 | 7.22132 | 1.00685 | 7.172x | 1.252x |
| Linux | float32 | checkerboard | 4 | 7.27501 | 1.57719 | 4.613x | 0.889x |
| Linux | float32 | tiny | 1 | 0.10389 | 0.08907 | 1.166x | 1.000x |
| Linux | float32 | tiny | 4 | 0.10632 | 0.09105 | 1.168x | 0.996x |
| Linux | float32 | nan-background | 1 | 3.59234 | 1.15284 | 3.116x | 0.954x |
| Linux | float32 | nan-background | 4 | 3.54746 | 0.80355 | 4.415x | 1.028x |
| Linux | float64 | random | 1 | 5.33369 | 5.17550 | 1.031x | 1.236x |
| Linux | float64 | random | 4 | 5.37959 | 3.67033 | 1.466x | 0.980x |
| Linux | float64 | fortran | 1 | 8.61508 | 7.83551 | 1.099x | 1.098x |
| Linux | float64 | fortran | 4 | 6.37571 | 5.83820 | 1.092x | 0.977x |
| Linux | float64 | strided | 1 | 2.81855 | 2.78789 | 1.011x | 1.277x |
| Linux | float64 | strided | 4 | 2.84941 | 2.82915 | 1.007x | 1.264x |
| Linux | float64 | all-weak | 1 | 4.23005 | 1.03827 | 4.074x | 1.031x |
| Linux | float64 | all-weak | 4 | 4.26327 | 0.75672 | 5.634x | 0.971x |
| Linux | float64 | no-seeds | 1 | 5.20947 | 1.99201 | 2.615x | 0.998x |
| Linux | float64 | no-seeds | 4 | 5.27432 | 0.97530 | 5.408x | 1.015x |
| Linux | float64 | equal | 1 | 8.01061 | 1.93560 | 4.139x | 1.059x |
| Linux | float64 | equal | 4 | 8.11797 | 0.95159 | 8.531x | 1.024x |
| Linux | float64 | threshold-fields | 1 | 5.23904 | 5.00268 | 1.047x | 1.281x |
| Linux | float64 | threshold-fields | 4 | 8.29432 | 4.34076 | 1.911x | 0.847x |
| Linux | float64 | snake | 1 | 3.68400 | 1.00466 | 3.667x | 1.671x |
| Linux | float64 | snake | 4 | 3.85149 | 1.57852 | 2.440x | 0.956x |
| Linux | float64 | checkerboard | 1 | 7.91104 | 0.77544 | 10.202x | 1.408x |
| Linux | float64 | checkerboard | 4 | 7.30425 | 1.42965 | 5.109x | 0.919x |
| Linux | float64 | tiny | 1 | 0.10413 | 0.08359 | 1.246x | 1.013x |
| Linux | float64 | tiny | 4 | 0.10869 | 0.08509 | 1.277x | 0.974x |
| Linux | float64 | nan-background | 1 | 3.48475 | 1.05240 | 3.311x | 0.950x |
| Linux | float64 | nan-background | 4 | 3.55637 | 0.69255 | 5.135x | 1.008x |
| Windows | uint8 | random | 1 | 6.85390 | 7.50050 | 0.914x | 1.118x |
| Windows | uint8 | random | 4 | 6.93025 | 5.01310 | 1.382x | 0.998x |
| Windows | uint8 | fortran | 1 | 8.13150 | 9.50530 | 0.855x | 1.123x |
| Windows | uint8 | fortran | 4 | 8.20470 | 7.47850 | 1.097x | 0.990x |
| Windows | uint8 | strided | 1 | 3.64627 | 3.53065 | 1.033x | 1.187x |
| Windows | uint8 | strided | 4 | 3.61613 | 3.51467 | 1.029x | 1.186x |
| Windows | uint8 | all-weak | 1 | 5.90885 | 2.42178 | 2.440x | 1.012x |
| Windows | uint8 | all-weak | 4 | 5.94740 | 1.63637 | 3.635x | 1.013x |
| Windows | uint8 | no-seeds | 1 | 6.97425 | 3.33045 | 2.094x | 0.995x |
| Windows | uint8 | no-seeds | 4 | 6.92065 | 2.33494 | 2.964x | 1.010x |
| Windows | uint8 | equal | 1 | 6.94665 | 2.70995 | 2.563x | 1.032x |
| Windows | uint8 | equal | 4 | 6.79485 | 1.73850 | 3.908x | 0.985x |
| Windows | uint8 | threshold-fields | 1 | 6.66015 | 6.49420 | 1.026x | 1.198x |
| Windows | uint8 | threshold-fields | 4 | 6.78415 | 4.40250 | 1.541x | 1.018x |
| Windows | uint8 | snake | 1 | 5.71540 | 1.91565 | 2.984x | 1.510x |
| Windows | uint8 | snake | 4 | 5.69465 | 2.89003 | 1.970x | 1.025x |
| Windows | uint8 | checkerboard | 1 | 6.79705 | 1.68570 | 4.032x | 1.128x |
| Windows | uint8 | checkerboard | 4 | 6.70280 | 2.50822 | 2.672x | 1.013x |
| Windows | uint8 | tiny | 1 | 0.10148 | 0.09457 | 1.073x | 1.000x |
| Windows | uint8 | tiny | 4 | 0.10155 | 0.09299 | 1.092x | 0.997x |
| Windows | float32 | random | 1 | 6.73630 | 6.92250 | 0.973x | 1.166x |
| Windows | float32 | random | 4 | 6.80675 | 4.79063 | 1.421x | 1.005x |
| Windows | float32 | fortran | 1 | 8.12060 | 9.48690 | 0.856x | 1.166x |
| Windows | float32 | fortran | 4 | 8.08420 | 7.95010 | 1.017x | 0.994x |
| Windows | float32 | strided | 1 | 3.63507 | 3.45570 | 1.052x | 1.187x |
| Windows | float32 | strided | 4 | 3.51463 | 3.39335 | 1.036x | 1.213x |
| Windows | float32 | all-weak | 1 | 5.85330 | 2.26095 | 2.589x | 1.020x |
| Windows | float32 | all-weak | 4 | 5.81840 | 1.53339 | 3.794x | 1.024x |
| Windows | float32 | no-seeds | 1 | 6.75610 | 2.46810 | 2.737x | 1.021x |
| Windows | float32 | no-seeds | 4 | 6.84735 | 1.58973 | 4.307x | 1.012x |
| Windows | float32 | equal | 1 | 6.84735 | 3.83823 | 1.784x | 0.918x |
| Windows | float32 | equal | 4 | 6.94705 | 3.66970 | 1.893x | 1.026x |
| Windows | float32 | threshold-fields | 1 | 6.95350 | 6.08885 | 1.142x | 1.272x |
| Windows | float32 | threshold-fields | 4 | 6.94260 | 4.57535 | 1.517x | 1.006x |
| Windows | float32 | snake | 1 | 5.59595 | 1.76007 | 3.179x | 1.539x |
| Windows | float32 | snake | 4 | 5.59995 | 2.76278 | 2.027x | 1.004x |
| Windows | float32 | checkerboard | 1 | 6.71070 | 1.52494 | 4.401x | 1.143x |
| Windows | float32 | checkerboard | 4 | 6.50520 | 2.43124 | 2.676x | 1.003x |
| Windows | float32 | tiny | 1 | 0.09769 | 0.08510 | 1.148x | 1.002x |
| Windows | float32 | tiny | 4 | 0.09795 | 0.08698 | 1.126x | 0.975x |
| Windows | float32 | nan-background | 1 | 5.33525 | 2.15206 | 2.479x | 1.058x |
| Windows | float32 | nan-background | 4 | 5.41195 | 1.49096 | 3.630x | 1.015x |
| Windows | float64 | random | 1 | 6.75245 | 6.08365 | 1.110x | 1.268x |
| Windows | float64 | random | 4 | 6.62530 | 4.21583 | 1.572x | 1.009x |
| Windows | float64 | fortran | 1 | 7.75580 | 9.93780 | 0.780x | 1.160x |
| Windows | float64 | fortran | 4 | 7.78580 | 8.59550 | 0.906x | 1.011x |
| Windows | float64 | strided | 1 | 3.60993 | 3.79367 | 0.952x | 1.172x |
| Windows | float64 | strided | 4 | 3.63793 | 3.74420 | 0.972x | 1.195x |
| Windows | float64 | all-weak | 1 | 5.72655 | 1.77773 | 3.221x | 1.024x |
| Windows | float64 | all-weak | 4 | 5.76370 | 1.01330 | 5.688x | 1.006x |
| Windows | float64 | no-seeds | 1 | 6.63975 | 2.02373 | 3.281x | 1.019x |
| Windows | float64 | no-seeds | 4 | 6.65395 | 1.08236 | 6.148x | 1.018x |
| Windows | float64 | equal | 1 | 6.51365 | 2.00470 | 3.249x | 1.026x |
| Windows | float64 | equal | 4 | 6.68135 | 1.06656 | 6.264x | 1.011x |
| Windows | float64 | threshold-fields | 1 | 7.37955 | 6.47845 | 1.139x | 1.267x |
| Windows | float64 | threshold-fields | 4 | 7.46420 | 4.69877 | 1.589x | 1.006x |
| Windows | float64 | snake | 1 | 5.54315 | 1.28276 | 4.321x | 1.772x |
| Windows | float64 | snake | 4 | 5.62930 | 2.25400 | 2.497x | 0.998x |
| Windows | float64 | checkerboard | 1 | 6.54275 | 1.03352 | 6.331x | 1.197x |
| Windows | float64 | checkerboard | 4 | 6.53275 | 1.86446 | 3.504x | 1.003x |
| Windows | float64 | tiny | 1 | 0.09571 | 0.07795 | 1.228x | 0.998x |
| Windows | float64 | tiny | 4 | 0.09638 | 0.07993 | 1.206x | 0.985x |
| Windows | float64 | nan-background | 1 | 5.32635 | 1.69648 | 3.140x | 1.034x |
| Windows | float64 | nan-background | 4 | 5.28165 | 0.97341 | 5.426x | 1.026x |
