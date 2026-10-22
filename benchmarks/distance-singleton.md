# CIEDE2000 singleton-stride verification

Measured source: `b0aca08`; product fingerprint `56cefa9d5a16c721373336f4f34781f4ba127a22f38715d0f8a7e6622dbe16f7`. Original reference: scikit-image 0.26.0 and NumPy 2.5.4.

A reversed one-pixel row must keep its negative channel-view iteration stride even though there is only one pixel. Scalar colors use the zero stride introduced by upstream unrolling. The binding now preserves both cases when selecting the native NumPy atan2 kernel. This fixes a stable mismatch that the strengthened [allocation-choice parity gate](distance-layout.md) correctly rejected. Numerical budgets and public signatures are unchanged.

[Four focused validation jobs](https://github.com/bipxx/scikit-imagexx/actions/runs/39768231545) each pass 392 tests: two replicas each of NumPy 1.24.4/Python 3.11 and NumPy 2.5.4/Python 3.13. [Full CI 39767948656](https://github.com/bipxx/scikit-imagexx/actions/runs/39767948656) passes all three jobs. Linux passes 44,419 tests and Windows passes 44,417 plus two skips in each SIMD mode. Normal, dynamic-team and portable native checks each pass 2/2 on both OSes. The minimum-NumPy job passes 3,306 focused CIEDE2000 cases and 872 Li cases in each SIMD mode, alongside its other selected suites. Exact wheels, raw smoke benchmarks and upstream allocation traces are retained.

[Paired isolated performance run](https://github.com/bipxx/scikit-imagexx/actions/runs/39768753243) passes on Windows and Linux, retaining the exact measured current and initial packages. It repeats the unchanged 40-comparison CIEDE2000 corpus with one/four worker limits, seven paired samples, 15 ms target batches, 20 ms cooldown, seed 386 and size 512 by 513. Reference thread pools use one thread; native fallback is disabled. Both packages pass each workload before timing.

| Host | One-worker original / current geometric mean | One-worker initial / current geometric mean |
|---|---:|---:|
| Linux | 2.260x | 0.980x |
| Windows | 2.469x | 0.971x |

The correction costs approximately 2.0% on Linux and 2.9% on Windows against the initial package by the one-worker geometric mean. It is retained for correctness. Every adverse case and raw sample remains below and in the accompanying JSON reports. These source-specific measurements do not establish a whole-library refresh.

| Host | Dtype | Layout | Options | Workers | Original ms | Current ms | Original / current | Initial / current |
|---|---|---|---|---:|---:|---:|---:|---:|
| Linux | float32 | contiguous | `{"channel_axis": -1.0}` | 1 | 58.76863 | 37.58040 | 1.564x | 1.004x |
| Linux | float32 | contiguous | `{"channel_axis": -1.0}` | 4 | 59.13612 | 12.24304 | 4.830x | 1.011x |
| Linux | float32 | contiguous | `{"channel_axis": -1.0, "kC": 0.5, "kL": 2.0}` | 1 | 58.67409 | 37.46825 | 1.566x | 1.006x |
| Linux | float32 | contiguous | `{"channel_axis": -1.0, "kC": 0.5, "kL": 2.0}` | 4 | 58.98050 | 12.31599 | 4.789x | 1.000x |
| Linux | float32 | channel-first | `{"channel_axis": 0.0}` | 1 | 58.40350 | 37.11693 | 1.574x | 1.003x |
| Linux | float32 | channel-first | `{"channel_axis": 0.0}` | 4 | 59.22683 | 12.04173 | 4.918x | 1.014x |
| Linux | float32 | channel-first | `{"channel_axis": 0.0, "kC": 0.5, "kL": 2.0}` | 1 | 58.54731 | 37.14225 | 1.576x | 1.001x |
| Linux | float32 | channel-first | `{"channel_axis": 0.0, "kC": 0.5, "kL": 2.0}` | 4 | 58.34806 | 11.90990 | 4.899x | 1.012x |
| Linux | float32 | strided | `{"channel_axis": -1.0}` | 1 | 31.02518 | 20.97567 | 1.479x | 1.001x |
| Linux | float32 | strided | `{"channel_axis": -1.0}` | 4 | 30.73182 | 21.02532 | 1.462x | 0.999x |
| Linux | float32 | strided | `{"channel_axis": -1.0, "kC": 0.5, "kL": 2.0}` | 1 | 30.63477 | 20.98936 | 1.460x | 1.001x |
| Linux | float32 | strided | `{"channel_axis": -1.0, "kC": 0.5, "kL": 2.0}` | 4 | 30.84319 | 21.01133 | 1.468x | 1.002x |
| Linux | float32 | scalar | `{"channel_axis": -1.0}` | 1 | 0.15985 | 0.03876 | 4.124x | 0.897x |
| Linux | float32 | scalar | `{"channel_axis": -1.0}` | 4 | 0.16071 | 0.03771 | 4.262x | 0.937x |
| Linux | float32 | scalar | `{"channel_axis": -1.0, "kC": 0.5, "kL": 2.0}` | 1 | 0.16441 | 0.03993 | 4.118x | 0.896x |
| Linux | float32 | scalar | `{"channel_axis": -1.0, "kC": 0.5, "kL": 2.0}` | 4 | 0.16156 | 0.03830 | 4.218x | 0.919x |
| Linux | float32 | weight-control | `{"kC": 0.0}` | 1 | 58.71484 | 37.63735 | 1.560x | 1.002x |
| Linux | float32 | weight-control | `{"kC": 0.0}` | 4 | 58.89448 | 12.21748 | 4.821x | 1.013x |
| Linux | float32 | weight-control | `{"kH": 0.0}` | 1 | 59.34103 | 37.56192 | 1.580x | 1.000x |
| Linux | float32 | weight-control | `{"kH": 0.0}` | 4 | 58.54225 | 12.27383 | 4.770x | 1.002x |
| Linux | float32 | weight-control | `{"kC": 0.699999988079071, "kL": 2.0}` | 1 | 58.66363 | 37.53926 | 1.563x | 1.005x |
| Linux | float32 | weight-control | `{"kC": 0.699999988079071, "kL": 2.0}` | 4 | 58.75727 | 12.24303 | 4.799x | 1.014x |
| Linux | float64 | contiguous | `{"channel_axis": -1.0}` | 1 | 92.12145 | 32.43616 | 2.840x | 0.998x |
| Linux | float64 | contiguous | `{"channel_axis": -1.0}` | 4 | 92.00091 | 11.61850 | 7.918x | 0.989x |
| Linux | float64 | contiguous | `{"channel_axis": -1.0, "kC": 0.5, "kL": 2.0}` | 1 | 92.06200 | 32.46979 | 2.835x | 0.995x |
| Linux | float64 | contiguous | `{"channel_axis": -1.0, "kC": 0.5, "kL": 2.0}` | 4 | 91.81166 | 11.63597 | 7.890x | 0.981x |
| Linux | float64 | channel-first | `{"channel_axis": 0.0}` | 1 | 91.29965 | 32.28543 | 2.828x | 1.001x |
| Linux | float64 | channel-first | `{"channel_axis": 0.0}` | 4 | 90.86013 | 11.52261 | 7.885x | 0.996x |
| Linux | float64 | channel-first | `{"channel_axis": 0.0, "kC": 0.5, "kL": 2.0}` | 1 | 90.73235 | 32.34224 | 2.805x | 0.995x |
| Linux | float64 | channel-first | `{"channel_axis": 0.0, "kC": 0.5, "kL": 2.0}` | 4 | 90.43462 | 11.49619 | 7.866x | 0.995x |
| Linux | float64 | strided | `{"channel_axis": -1.0}` | 1 | 47.71339 | 18.53169 | 2.575x | 0.994x |
| Linux | float64 | strided | `{"channel_axis": -1.0}` | 4 | 48.33113 | 18.45759 | 2.618x | 1.004x |
| Linux | float64 | strided | `{"channel_axis": -1.0, "kC": 0.5, "kL": 2.0}` | 1 | 48.76378 | 18.42698 | 2.646x | 1.002x |
| Linux | float64 | strided | `{"channel_axis": -1.0, "kC": 0.5, "kL": 2.0}` | 4 | 48.92532 | 18.51247 | 2.643x | 0.996x |
| Linux | float64 | scalar | `{"channel_axis": -1.0}` | 1 | 0.14914 | 0.03230 | 4.617x | 0.926x |
| Linux | float64 | scalar | `{"channel_axis": -1.0}` | 4 | 0.14913 | 0.03233 | 4.614x | 0.944x |
| Linux | float64 | scalar | `{"channel_axis": -1.0, "kC": 0.5, "kL": 2.0}` | 1 | 0.14887 | 0.03295 | 4.518x | 0.942x |
| Linux | float64 | scalar | `{"channel_axis": -1.0, "kC": 0.5, "kL": 2.0}` | 4 | 0.14922 | 0.03331 | 4.480x | 0.915x |
| Linux | float32 | Sharma-boundaries | `{}` | 1 | 0.49622 | 0.31847 | 1.558x | 0.944x |
| Linux | float32 | Sharma-boundaries | `{}` | 4 | 0.50391 | 0.32006 | 1.574x | 0.941x |
| Windows | float32 | contiguous | `{"channel_axis": -1.0}` | 1 | 71.81590 | 40.61640 | 1.768x | 1.004x |
| Windows | float32 | contiguous | `{"channel_axis": -1.0}` | 4 | 73.17260 | 14.80180 | 4.943x | 0.976x |
| Windows | float32 | contiguous | `{"channel_axis": -1.0, "kC": 0.5, "kL": 2.0}` | 1 | 71.78810 | 40.83580 | 1.758x | 0.997x |
| Windows | float32 | contiguous | `{"channel_axis": -1.0, "kC": 0.5, "kL": 2.0}` | 4 | 72.81560 | 14.17550 | 5.137x | 1.036x |
| Windows | float32 | channel-first | `{"channel_axis": 0.0}` | 1 | 71.83530 | 39.63880 | 1.812x | 1.002x |
| Windows | float32 | channel-first | `{"channel_axis": 0.0}` | 4 | 71.61480 | 13.67330 | 5.238x | 1.041x |
| Windows | float32 | channel-first | `{"channel_axis": 0.0, "kC": 0.5, "kL": 2.0}` | 1 | 72.74550 | 39.67540 | 1.834x | 1.004x |
| Windows | float32 | channel-first | `{"channel_axis": 0.0, "kC": 0.5, "kL": 2.0}` | 4 | 72.12780 | 14.51380 | 4.970x | 1.068x |
| Windows | float32 | strided | `{"channel_axis": -1.0}` | 1 | 35.41940 | 22.88230 | 1.548x | 1.000x |
| Windows | float32 | strided | `{"channel_axis": -1.0}` | 4 | 36.11310 | 22.94570 | 1.574x | 0.997x |
| Windows | float32 | strided | `{"channel_axis": -1.0, "kC": 0.5, "kL": 2.0}` | 1 | 35.23800 | 23.14250 | 1.523x | 0.994x |
| Windows | float32 | strided | `{"channel_axis": -1.0, "kC": 0.5, "kL": 2.0}` | 4 | 35.33840 | 23.01150 | 1.536x | 1.002x |
| Windows | float32 | scalar | `{"channel_axis": -1.0}` | 1 | 0.17655 | 0.04093 | 4.314x | 0.840x |
| Windows | float32 | scalar | `{"channel_axis": -1.0}` | 4 | 0.17661 | 0.03734 | 4.729x | 0.912x |
| Windows | float32 | scalar | `{"channel_axis": -1.0, "kC": 0.5, "kL": 2.0}` | 1 | 0.17817 | 0.03893 | 4.576x | 0.866x |
| Windows | float32 | scalar | `{"channel_axis": -1.0, "kC": 0.5, "kL": 2.0}` | 4 | 0.18024 | 0.04119 | 4.375x | 0.887x |
| Windows | float32 | weight-control | `{"kC": 0.0}` | 1 | 73.03610 | 40.57070 | 1.800x | 1.005x |
| Windows | float32 | weight-control | `{"kC": 0.0}` | 4 | 72.75090 | 13.80850 | 5.269x | 1.018x |
| Windows | float32 | weight-control | `{"kH": 0.0}` | 1 | 72.49890 | 40.60620 | 1.785x | 1.006x |
| Windows | float32 | weight-control | `{"kH": 0.0}` | 4 | 72.34920 | 13.83790 | 5.228x | 1.009x |
| Windows | float32 | weight-control | `{"kC": 0.699999988079071, "kL": 2.0}` | 1 | 73.08910 | 40.56610 | 1.802x | 1.004x |
| Windows | float32 | weight-control | `{"kC": 0.699999988079071, "kL": 2.0}` | 4 | 72.63540 | 13.80980 | 5.260x | 1.009x |
| Windows | float64 | contiguous | `{"channel_axis": -1.0}` | 1 | 102.38240 | 32.26420 | 3.173x | 1.005x |
| Windows | float64 | contiguous | `{"channel_axis": -1.0}` | 4 | 101.32120 | 11.06440 | 9.157x | 1.007x |
| Windows | float64 | contiguous | `{"channel_axis": -1.0, "kC": 0.5, "kL": 2.0}` | 1 | 101.84750 | 32.44840 | 3.139x | 1.001x |
| Windows | float64 | contiguous | `{"channel_axis": -1.0, "kC": 0.5, "kL": 2.0}` | 4 | 102.28300 | 11.08540 | 9.227x | 0.999x |
| Windows | float64 | channel-first | `{"channel_axis": 0.0}` | 1 | 99.89990 | 32.22520 | 3.100x | 1.001x |
| Windows | float64 | channel-first | `{"channel_axis": 0.0}` | 4 | 100.13210 | 10.95090 | 9.144x | 1.004x |
| Windows | float64 | channel-first | `{"channel_axis": 0.0, "kC": 0.5, "kL": 2.0}` | 1 | 100.24810 | 32.24690 | 3.109x | 1.000x |
| Windows | float64 | channel-first | `{"channel_axis": 0.0, "kC": 0.5, "kL": 2.0}` | 4 | 99.60570 | 11.06310 | 9.003x | 0.989x |
| Windows | float64 | strided | `{"channel_axis": -1.0}` | 1 | 55.65250 | 19.47750 | 2.857x | 1.000x |
| Windows | float64 | strided | `{"channel_axis": -1.0}` | 4 | 56.15250 | 19.39460 | 2.895x | 1.000x |
| Windows | float64 | strided | `{"channel_axis": -1.0, "kC": 0.5, "kL": 2.0}` | 1 | 56.01950 | 19.42020 | 2.885x | 1.007x |
| Windows | float64 | strided | `{"channel_axis": -1.0, "kC": 0.5, "kL": 2.0}` | 4 | 55.66170 | 19.28860 | 2.886x | 1.000x |
| Windows | float64 | scalar | `{"channel_axis": -1.0}` | 1 | 0.16208 | 0.03438 | 4.714x | 0.846x |
| Windows | float64 | scalar | `{"channel_axis": -1.0}` | 4 | 0.16594 | 0.03321 | 4.996x | 0.889x |
| Windows | float64 | scalar | `{"channel_axis": -1.0, "kC": 0.5, "kL": 2.0}` | 1 | 0.16075 | 0.03127 | 5.141x | 0.927x |
| Windows | float64 | scalar | `{"channel_axis": -1.0, "kC": 0.5, "kL": 2.0}` | 4 | 0.16199 | 0.03163 | 5.121x | 0.908x |
| Windows | float32 | Sharma-boundaries | `{}` | 1 | 0.48383 | 0.36544 | 1.324x | 0.953x |
| Windows | float32 | Sharma-boundaries | `{}` | 4 | 0.47907 | 0.36157 | 1.325x | 0.954x |
