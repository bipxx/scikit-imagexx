# CIEDE2000 runtime math parity

Retained product: `308e1f0ac41461662a46cbb22a2e1fb0202d0f28`, source fingerprint `7bdbb498c5999e364c0bdeaf9f4622a301923a5ecf105794055004f5382435fa`. Previous saved build: `bb0d364`. Reference: scikit-image 0.26.0.

NumPy 1.24 on AVX-512 hosts can use SVML rather than libm. One-ulp differences in atan2, hypot and power can cross the CIEDE2000 180-degree hue discontinuity, causing order-one output differences. The previous source reproduces the original six CI failures on those hosts. A source-derived SVML emulation reproduced all six locally; a wider 735-input neighborhood also exposed chroma-rounding failures on ordinary libm hosts.

A tiny import-time probe selects the double-angle provider. The native kernel includes an AVX2/FMA adaptation of NumPy/Intel's BSD-licensed SVML atan2 polynomial and a scalar implementation. A SIMD geometry check identifies nearly antipodal lanes for scalar refinement. On the NumPy SVML profile, refinement and scalar tails call the runtime's builtin floating C loops for power/hypot, without the GIL or Python calls. Buffers occupy separate cache lines because NumPy1's conservative overlap test otherwise selects its libm path for adjacent scalars. Image traversal, image output, ordinary lanes and optional global workers remain C++. The standalone C++ entry point retains its ABI and libm behavior; an additive compatibility entry point accepts optional thread-safe elementary-math hooks. Public Python signatures are unchanged.

Provider implementation references: [NumPy1 SIMD loops](https://github.com/numpy/numpy/blob/v1.24.4/numpy/core/src/umath/loops_umath_fp.dispatch.c.src), [overlap checks](https://github.com/numpy/numpy/blob/v1.24.4/numpy/core/src/umath/loops_utils.h.src), [native ufunc-loop API](https://numpy.org/doc/1.24/reference/c-api/ufunc.html).

All 3,005 focused cases pass on Windows/NumPy 2.5.4 and Linux/NumPy 1.24.4 in each SIMD mode. The 37 added cases cover captured SVML outputs, signed/broadcast/planar views, optional workers and neighboring floating representations around antipodal hues. Native address/undefined-behavior checks pass with AVX2, reduced dynamic OpenMP teams and portable builds without SIMD/OpenMP, including callback context restoration. An independent 53-case SVML-emulated subset passes with SIMD enabled and disabled. No upstream parity tolerances were widened.

On the SVML host original scikit-image itself disagrees with one four-decimal published Sharma value. Every public result still compares against the active upstream provider. A separate standalone C++ libm-profile check verifies all 34 published values; it does not assert that different provider branches are numerically interchangeable.

The actual [AVX-512/SVML confirmation](https://github.com/bipxx/scikit-imagexx/actions/runs/39753080008) passes the minimum-runtime suite in both SIMD modes and records `svml_double_angles=true`. Its exact measured wheel and every sample are retained. Full current-source [Windows/Linux CI](https://github.com/bipxx/scikit-imagexx/actions/runs/39752595311) remains in progress at publication; the earlier source run39749730076 passed all jobs but used an AVX2 minimum-runtime host and is not evidence for the SVML correction.

Performance uses the unchanged 20-workload corpus in `ciede2000.py`: seed386, size512, seven shuffled warmed repetitions, 15ms target batches and 20ms cooldowns. It includes grayscale-free Lab float32/float64 images, contiguous/planar/strided/scalar layouts, weights and Sharma controls. Calls include preparation, allocation and destruction. Original pools use one thread; C++ uses one/four global workers. Local timings never overlap builds/tests. Saved comparator binary/Python hashes and ISA are verified. CI timing is a VM observation, not a controlled cross-machine comparison. Ratios from different runs are not chained.

| Run | Workers | Previous/current geometric mean | Original/current | Faster than original |
|---|---:|---:|---:|---:|
| Windows/NumPy2.5.4 | 1 | 1.005× | 2.130× | 20/20 |
| Windows/NumPy2.5.4 | 4 | 0.995× | 4.364× | 20/20 |
| Linux/NumPy1.24.4 | 1 | 1.027× | 2.878× | 20/20 |
| Linux/NumPy1.24.4 | 4 | 1.014× | 5.732× | 20/20 |
| CI AVX2 | 1 | — | 2.777× | 20/20 |
| CI AVX-512/SVML | 1 | — | 2.338× | 19/20 |

This increment fixes provider parity; it is not a universal performance improvement. Every regression versus the saved previous build remains below and in the raw history. The geometry/refinement work adds numerical protection, and antipodal-heavy inputs can spend more time in scalar/native-math refinement. These costs and scalar-call variation remain optimization targets. API coverage remains partial.

All local variants and raw samples are preserved in `distance-provider-tuning.json`. Exact local intermediate/final packages, wheels and source archives, final CI wheels and validation logs are hash-verified under `build/checkpoints/distance-provider-validation`. The first historical CI report omitted its wheel; that archival limitation is explicit in the history.

| Run | Dtype | Layout | Options | Workers | Original ms | Current ms | Original/current | Previous/current |
|---|---|---|---|---:|---:|---:|---:|---:|
| Windows | float32 | contiguous | {"channel_axis": -1.0} | 1 | 40.4467 | 29.5230 | 1.370× | 1.025× |
| Windows | float32 | contiguous | {"channel_axis": -1.0} | 4 | 40.3515 | 7.9675 | 5.065× | 1.039× |
| Windows | float32 | contiguous | {"channel_axis": -1.0, "kC": 0.5, "kL": 2.0} | 1 | 41.5738 | 30.3775 | 1.369× | 1.022× |
| Windows | float32 | contiguous | {"channel_axis": -1.0, "kC": 0.5, "kL": 2.0} | 4 | 40.3843 | 8.5319 | 4.733× | 0.973× |
| Windows | float32 | channel-first | {"channel_axis": 0.0} | 1 | 41.7694 | 29.6445 | 1.409× | 0.971× |
| Windows | float32 | channel-first | {"channel_axis": 0.0} | 4 | 40.8149 | 7.9335 | 5.145× | 1.046× |
| Windows | float32 | channel-first | {"channel_axis": 0.0, "kC": 0.5, "kL": 2.0} | 1 | 40.1793 | 28.9399 | 1.388× | 0.968× |
| Windows | float32 | channel-first | {"channel_axis": 0.0, "kC": 0.5, "kL": 2.0} | 4 | 39.8072 | 7.8836 | 5.049× | 1.033× |
| Windows | float32 | strided | {"channel_axis": -1.0} | 1 | 22.5903 | 15.9924 | 1.413× | 1.051× |
| Windows | float32 | strided | {"channel_axis": -1.0} | 4 | 23.2498 | 16.3881 | 1.419× | 1.020× |
| Windows | float32 | strided | {"channel_axis": -1.0, "kC": 0.5, "kL": 2.0} | 1 | 20.0413 | 15.7156 | 1.275× | 1.034× |
| Windows | float32 | strided | {"channel_axis": -1.0, "kC": 0.5, "kL": 2.0} | 4 | 21.1916 | 15.9716 | 1.327× | 1.019× |
| Windows | float32 | scalar | {"channel_axis": -1.0} | 1 | 0.1217 | 0.0188 | 6.471× | 0.899× |
| Windows | float32 | scalar | {"channel_axis": -1.0} | 4 | 0.1029 | 0.0180 | 5.700× | 0.946× |
| Windows | float32 | scalar | {"channel_axis": -1.0, "kC": 0.5, "kL": 2.0} | 1 | 0.1263 | 0.0353 | 3.581× | 0.907× |
| Windows | float32 | scalar | {"channel_axis": -1.0, "kC": 0.5, "kL": 2.0} | 4 | 0.0991 | 0.0165 | 5.992× | 0.970× |
| Windows | float32 | weight-control | {"kC": 0.0} | 1 | 40.9084 | 29.8302 | 1.371× | 1.039× |
| Windows | float32 | weight-control | {"kC": 0.0} | 4 | 41.3894 | 8.2169 | 5.037× | 1.063× |
| Windows | float32 | weight-control | {"kH": 0.0} | 1 | 39.7884 | 29.9909 | 1.327× | 1.016× |
| Windows | float32 | weight-control | {"kH": 0.0} | 4 | 40.7876 | 8.3033 | 4.912× | 1.029× |
| Windows | float32 | weight-control | {"kC": 0.699999988079071, "kL": 2.0} | 1 | 41.2374 | 29.7557 | 1.386× | 1.033× |
| Windows | float32 | weight-control | {"kC": 0.699999988079071, "kL": 2.0} | 4 | 41.6980 | 8.1912 | 5.091× | 0.998× |
| Windows | float64 | contiguous | {"channel_axis": -1.0} | 1 | 62.8391 | 22.9382 | 2.739× | 0.977× |
| Windows | float64 | contiguous | {"channel_axis": -1.0} | 4 | 59.9824 | 6.2340 | 9.622× | 0.995× |
| Windows | float64 | contiguous | {"channel_axis": -1.0, "kC": 0.5, "kL": 2.0} | 1 | 58.1970 | 22.3560 | 2.603× | 0.991× |
| Windows | float64 | contiguous | {"channel_axis": -1.0, "kC": 0.5, "kL": 2.0} | 4 | 60.4909 | 6.5839 | 9.188× | 0.978× |
| Windows | float64 | channel-first | {"channel_axis": 0.0} | 1 | 57.8615 | 22.0073 | 2.629× | 1.016× |
| Windows | float64 | channel-first | {"channel_axis": 0.0} | 4 | 57.8454 | 6.1201 | 9.452× | 1.150× |
| Windows | float64 | channel-first | {"channel_axis": 0.0, "kC": 0.5, "kL": 2.0} | 1 | 61.4850 | 22.0972 | 2.782× | 0.988× |
| Windows | float64 | channel-first | {"channel_axis": 0.0, "kC": 0.5, "kL": 2.0} | 4 | 56.9025 | 6.2702 | 9.075× | 0.977× |
| Windows | float64 | strided | {"channel_axis": -1.0} | 1 | 31.0039 | 13.8706 | 2.235× | 0.975× |
| Windows | float64 | strided | {"channel_axis": -1.0} | 4 | 30.3790 | 13.2308 | 2.296× | 0.985× |
| Windows | float64 | strided | {"channel_axis": -1.0, "kC": 0.5, "kL": 2.0} | 1 | 29.7580 | 13.2353 | 2.248× | 0.966× |
| Windows | float64 | strided | {"channel_axis": -1.0, "kC": 0.5, "kL": 2.0} | 4 | 30.4430 | 13.2103 | 2.304× | 0.991× |
| Windows | float64 | scalar | {"channel_axis": -1.0} | 1 | 0.1108 | 0.0230 | 4.820× | 1.142× |
| Windows | float64 | scalar | {"channel_axis": -1.0} | 4 | 0.0981 | 0.0198 | 4.964× | 0.928× |
| Windows | float64 | scalar | {"channel_axis": -1.0, "kC": 0.5, "kL": 2.0} | 1 | 0.0860 | 0.0151 | 5.698× | 1.137× |
| Windows | float64 | scalar | {"channel_axis": -1.0, "kC": 0.5, "kL": 2.0} | 4 | 0.1091 | 0.0218 | 5.007× | 0.793× |
| Windows | float32 | Sharma-boundaries | {} | 1 | 0.2890 | 0.2168 | 1.333× | 0.973× |
| Windows | float32 | Sharma-boundaries | {} | 4 | 0.3096 | 0.2481 | 1.248× | 1.006× |
| Linux | float32 | contiguous | {"channel_axis": -1.0} | 1 | 55.9398 | 22.6515 | 2.470× | 1.041× |
| Linux | float32 | contiguous | {"channel_axis": -1.0} | 4 | 56.3807 | 6.0898 | 9.258× | 1.052× |
| Linux | float32 | contiguous | {"channel_axis": -1.0, "kC": 0.5, "kL": 2.0} | 1 | 56.8042 | 22.5871 | 2.515× | 1.044× |
| Linux | float32 | contiguous | {"channel_axis": -1.0, "kC": 0.5, "kL": 2.0} | 4 | 56.1425 | 7.1433 | 7.859× | 0.950× |
| Linux | float32 | channel-first | {"channel_axis": 0.0} | 1 | 56.3849 | 22.5598 | 2.499× | 0.992× |
| Linux | float32 | channel-first | {"channel_axis": 0.0} | 4 | 61.8714 | 7.6308 | 8.108× | 0.839× |
| Linux | float32 | channel-first | {"channel_axis": 0.0, "kC": 0.5, "kL": 2.0} | 1 | 57.5813 | 22.3853 | 2.572× | 0.988× |
| Linux | float32 | channel-first | {"channel_axis": 0.0, "kC": 0.5, "kL": 2.0} | 4 | 54.7006 | 5.9112 | 9.254× | 1.024× |
| Linux | float32 | strided | {"channel_axis": -1.0} | 1 | 28.3263 | 12.4616 | 2.273× | 1.032× |
| Linux | float32 | strided | {"channel_axis": -1.0} | 4 | 28.8593 | 12.3039 | 2.346× | 1.055× |
| Linux | float32 | strided | {"channel_axis": -1.0, "kC": 0.5, "kL": 2.0} | 1 | 28.4027 | 12.6010 | 2.254× | 1.041× |
| Linux | float32 | strided | {"channel_axis": -1.0, "kC": 0.5, "kL": 2.0} | 4 | 29.0002 | 12.1406 | 2.389× | 1.077× |
| Linux | float32 | scalar | {"channel_axis": -1.0} | 1 | 0.0878 | 0.0215 | 4.077× | 1.050× |
| Linux | float32 | scalar | {"channel_axis": -1.0} | 4 | 0.0788 | 0.0281 | 2.804× | 0.868× |
| Linux | float32 | scalar | {"channel_axis": -1.0, "kC": 0.5, "kL": 2.0} | 1 | 0.0902 | 0.0208 | 4.349× | 1.025× |
| Linux | float32 | scalar | {"channel_axis": -1.0, "kC": 0.5, "kL": 2.0} | 4 | 0.0958 | 0.0247 | 3.879× | 1.353× |
| Linux | float32 | weight-control | {"kC": 0.0} | 1 | 64.9343 | 22.8225 | 2.845× | 1.057× |
| Linux | float32 | weight-control | {"kC": 0.0} | 4 | 56.4080 | 6.1542 | 9.166× | 1.034× |
| Linux | float32 | weight-control | {"kH": 0.0} | 1 | 53.3558 | 20.5861 | 2.592× | 1.081× |
| Linux | float32 | weight-control | {"kH": 0.0} | 4 | 51.5561 | 6.1125 | 8.435× | 0.974× |
| Linux | float32 | weight-control | {"kC": 0.699999988079071, "kL": 2.0} | 1 | 52.6305 | 20.9210 | 2.516× | 1.067× |
| Linux | float32 | weight-control | {"kC": 0.699999988079071, "kL": 2.0} | 4 | 54.5830 | 5.6621 | 9.640× | 1.043× |
| Linux | float64 | contiguous | {"channel_axis": -1.0} | 1 | 59.5302 | 20.2281 | 2.943× | 1.003× |
| Linux | float64 | contiguous | {"channel_axis": -1.0} | 4 | 58.5994 | 5.5373 | 10.583× | 1.036× |
| Linux | float64 | contiguous | {"channel_axis": -1.0, "kC": 0.5, "kL": 2.0} | 1 | 59.2941 | 20.2339 | 2.930× | 1.015× |
| Linux | float64 | contiguous | {"channel_axis": -1.0, "kC": 0.5, "kL": 2.0} | 4 | 64.9580 | 6.3305 | 10.261× | 0.966× |
| Linux | float64 | channel-first | {"channel_axis": 0.0} | 1 | 62.6031 | 21.5016 | 2.912× | 1.010× |
| Linux | float64 | channel-first | {"channel_axis": 0.0} | 4 | 63.4869 | 6.2325 | 10.186× | 1.017× |
| Linux | float64 | channel-first | {"channel_axis": 0.0, "kC": 0.5, "kL": 2.0} | 1 | 63.1441 | 22.0168 | 2.868× | 0.998× |
| Linux | float64 | channel-first | {"channel_axis": 0.0, "kC": 0.5, "kL": 2.0} | 4 | 62.1090 | 5.9772 | 10.391× | 1.000× |
| Linux | float64 | strided | {"channel_axis": -1.0} | 1 | 32.7558 | 11.8769 | 2.758× | 1.007× |
| Linux | float64 | strided | {"channel_axis": -1.0} | 4 | 32.2126 | 12.1569 | 2.650× | 0.992× |
| Linux | float64 | strided | {"channel_axis": -1.0, "kC": 0.5, "kL": 2.0} | 1 | 34.6783 | 11.7650 | 2.948× | 0.999× |
| Linux | float64 | strided | {"channel_axis": -1.0, "kC": 0.5, "kL": 2.0} | 4 | 31.4425 | 11.4716 | 2.741× | 1.021× |
| Linux | float64 | scalar | {"channel_axis": -1.0} | 1 | 0.0814 | 0.0186 | 4.370× | 0.847× |
| Linux | float64 | scalar | {"channel_axis": -1.0} | 4 | 0.1041 | 0.0185 | 5.614× | 1.129× |
| Linux | float64 | scalar | {"channel_axis": -1.0, "kC": 0.5, "kL": 2.0} | 1 | 0.0825 | 0.0179 | 4.621× | 1.188× |
| Linux | float64 | scalar | {"channel_axis": -1.0, "kC": 0.5, "kL": 2.0} | 4 | 0.0931 | 0.0139 | 6.673× | 0.930× |
| Linux | float32 | Sharma-boundaries | {} | 1 | 0.3448 | 0.1808 | 1.906× | 1.082× |
| Linux | float32 | Sharma-boundaries | {} | 4 | 0.3465 | 0.1834 | 1.890× | 1.004× |
| CI SVML | float32 | contiguous | {"channel_axis": -1.0} | 1 | 24.3646 | 13.8320 | 1.761× | — |
| CI SVML | float32 | contiguous | {"channel_axis": -1.0, "kC": 0.5, "kL": 2.0} | 1 | 23.5773 | 13.7451 | 1.715× | — |
| CI SVML | float32 | channel-first | {"channel_axis": 0.0} | 1 | 23.8452 | 13.1659 | 1.811× | — |
| CI SVML | float32 | channel-first | {"channel_axis": 0.0, "kC": 0.5, "kL": 2.0} | 1 | 36.0135 | 13.0277 | 2.764× | — |
| CI SVML | float32 | strided | {"channel_axis": -1.0} | 1 | 13.0874 | 8.2414 | 1.588× | — |
| CI SVML | float32 | strided | {"channel_axis": -1.0, "kC": 0.5, "kL": 2.0} | 1 | 13.2666 | 8.2486 | 1.608× | — |
| CI SVML | float32 | scalar | {"channel_axis": -1.0} | 1 | 0.0753 | 0.0173 | 4.346× | — |
| CI SVML | float32 | scalar | {"channel_axis": -1.0, "kC": 0.5, "kL": 2.0} | 1 | 0.0750 | 0.0172 | 4.350× | — |
| CI SVML | float32 | weight-control | {"kC": 0.0} | 1 | 24.1425 | 13.7890 | 1.751× | — |
| CI SVML | float32 | weight-control | {"kH": 0.0} | 1 | 24.6264 | 13.8170 | 1.782× | — |
| CI SVML | float32 | weight-control | {"kC": 0.699999988079071, "kL": 2.0} | 1 | 24.2726 | 13.7989 | 1.759× | — |
| CI SVML | float64 | contiguous | {"channel_axis": -1.0} | 1 | 32.3177 | 12.7505 | 2.535× | — |
| CI SVML | float64 | contiguous | {"channel_axis": -1.0, "kC": 0.5, "kL": 2.0} | 1 | 32.5345 | 12.7173 | 2.558× | — |
| CI SVML | float64 | channel-first | {"channel_axis": 0.0} | 1 | 30.4206 | 12.6008 | 2.414× | — |
| CI SVML | float64 | channel-first | {"channel_axis": 0.0, "kC": 0.5, "kL": 2.0} | 1 | 30.3506 | 12.6525 | 2.399× | — |
| CI SVML | float64 | strided | {"channel_axis": -1.0} | 1 | 16.8693 | 7.6279 | 2.212× | — |
| CI SVML | float64 | strided | {"channel_axis": -1.0, "kC": 0.5, "kL": 2.0} | 1 | 17.3687 | 7.7174 | 2.251× | — |
| CI SVML | float64 | scalar | {"channel_axis": -1.0} | 1 | 0.0748 | 0.0131 | 5.711× | — |
| CI SVML | float64 | scalar | {"channel_axis": -1.0, "kC": 0.5, "kL": 2.0} | 1 | 0.0744 | 0.0131 | 5.680× | — |
| CI SVML | float32 | Sharma-boundaries | {} | 1 | 0.1837 | 0.1875 | 0.980× | — |
