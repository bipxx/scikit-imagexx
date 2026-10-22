# CIEDE2000 runtime angle dispatch by layout

Validation update: [CI 39763182640](https://github.com/bipxx/scikit-imagexx/actions/runs/39763182640) passes all three jobs: Linux 44,136 passed and Windows 44,134 passed plus two skipped in each SIMD mode, with native normal/dynamic/portable checks passing 2/2. A subsequent AVX-512/NumPy 1.24 runner in run 39764951364 found two length-128 float64 antipodal cases after this green run. The allocation diagnostics below explain that subsequent difference; the earlier green run alone did not settle it. Exact old/new measured packages and all samples below are retained unchanged.

[Allocation diagnostics on two AVX-512 runners](https://github.com/bipxx/scikit-imagexx/actions/runs/39766158961) subsequently reproduce two or three different upstream arrays for identical input arrays. The original function varies by up to 6.252 units; native output stays invariant. Traces show NumPy selecting libm when disjoint strided input/output buffers trigger its conservative overlap test. NumPy 1.24 computes those bounds using stride times length with inclusive endpoints ([overlap helper](https://github.com/numpy/numpy/blob/v1.24.4/numpy/core/src/umath/loops_utils.h.src#L19), [math dispatcher](https://github.com/numpy/numpy/blob/v1.24.4/numpy/core/src/umath/loops_umath_fp.dispatch.c.src#L254)).

The antipodal parity gate now keeps the same numerical budgets and first compares normally. Only after a mismatch does it perturb temporary allocator occupancy while keeping the original arguments unchanged. Acceptance requires demonstrated upstream instability and a matching whole array actually returned by upstream. It never combines elementwise alternatives or broadens tolerances. Control tests reject stable wrong results and mixtures of valid arrays. Single-call equality at these allocation-dependent upstream discontinuities is not guaranteed; the native result must be an observed upstream choice. Other parity tests remain unchanged.

That gate subsequently caught a genuine stable NumPy 2.5 mismatch for reversed one-pixel rows. The binding had discarded the singleton axis's negative stride. Source `6ee9efc` preserves that stride, including zero strides for scalar-color unrolling, and adds singleton/broadcast regressions. [Four focused jobs](https://github.com/bipxx/scikit-imagexx/actions/runs/39768231545), covering two replicas each of NumPy 1.24.4 and 2.5.4, pass all 392 cases. The [singleton-stride checkpoint](distance-singleton.md) passes full CI and paired timing on both OSes. The measurements below retain the earlier exact source checkpoint.

Source: `a259fb7`; product fingerprint `fca68cf1daecbc13fea97919edad6233049211926d488263e5d045398b84596a`. Reference: scikit-image 0.26.0, NumPy 2.5.4.

NumPy 2.5 can select its X86_V4 atan2 kernel for positive strides and a different path for negative strides. The former fixed SVML probe did not identify this newer kernel. Antipodal hues amplify a few angle ULPs into the opposite hue branch. [The diagnostic run](https://github.com/bipxx/scikit-imagexx/actions/runs/39762548860) preserves the exact initial package and intermediate results: chroma, seventh powers and scale matched; the angle values changed with layout.

The C++ scalar refinement and tails now call the selected native NumPy atan2 loop with each original input’s signed inner stride. Calls use the native C loop without the GIL; no Python image arithmetic or upstream fallback runs. Existing SIMD image processing and global optional workers remain. The explicit legacy SVML test profile remains separate. Added 336 parity cases cover vector boundaries, mixed layouts, positive and negative strides and nonlinear views, with unchanged numerical budgets. Each scoped Windows and Linux/NumPy 1.24 SIMD mode passed 3,287 cases. Native normal/dynamic teams, portable and ASan/UBSan checks passed 2/2.

[Paired isolated measurements and exact wheels](https://github.com/bipxx/scikit-imagexx/actions/runs/39763752736) compare the preserved initial package with the corrected package in the same process. Each OS completed 40 comparisons, including 20 one-worker cases. Corpus: seed 386, 512 by 513, float32/float64 layouts and scalar weights, scalar controls, and Sharma boundaries; seven randomized paired samples, 15 ms target batches, 20 ms cooldown; original thread pools fixed to one. Every current and baseline workload passed parity before timing.

| Host | Original / current, one-worker geometric mean | Baseline / current, one-worker geometric mean |
|---|---:|---:|
| Linux | 2.259x | 0.984x |
| Windows | 2.869x | 0.983x |

This is a correctness correction. It adds scalar/refinement overhead: Linux scalar controls regress roughly 5–10% against the prior package; ordinary image cases stay near baseline. All observed regressions and every raw sample remain in the JSON reports. These measurements do not establish a whole-library performance checkpoint.

| Host | Dtype | Layout | Options | Workers | Original ms | Current ms | Original / current | Baseline / current |
|---|---|---|---|---:|---:|---:|---:|---:|
| Linux | float32 | contiguous | `{"channel_axis": -1.0}` | 1 | 58.47345 | 37.84448 | 1.545x | 0.996x |
| Linux | float32 | contiguous | `{"channel_axis": -1.0}` | 4 | 58.32128 | 12.57866 | 4.637x | 1.008x |
| Linux | float32 | contiguous | `{"channel_axis": -1.0, "kC": 0.5, "kL": 2.0}` | 1 | 58.38538 | 37.81021 | 1.544x | 0.998x |
| Linux | float32 | contiguous | `{"channel_axis": -1.0, "kC": 0.5, "kL": 2.0}` | 4 | 58.49801 | 12.38188 | 4.724x | 1.011x |
| Linux | float32 | channel-first | `{"channel_axis": 0.0}` | 1 | 57.93928 | 37.45378 | 1.547x | 0.996x |
| Linux | float32 | channel-first | `{"channel_axis": 0.0}` | 4 | 57.90452 | 12.00880 | 4.822x | 1.006x |
| Linux | float32 | channel-first | `{"channel_axis": 0.0, "kC": 0.5, "kL": 2.0}` | 1 | 58.19042 | 37.48566 | 1.552x | 0.994x |
| Linux | float32 | channel-first | `{"channel_axis": 0.0, "kC": 0.5, "kL": 2.0}` | 4 | 58.25934 | 12.13675 | 4.800x | 0.994x |
| Linux | float32 | strided | `{"channel_axis": -1.0}` | 1 | 30.46609 | 21.02283 | 1.449x | 0.999x |
| Linux | float32 | strided | `{"channel_axis": -1.0}` | 4 | 30.52945 | 21.15296 | 1.443x | 0.991x |
| Linux | float32 | strided | `{"channel_axis": -1.0, "kC": 0.5, "kL": 2.0}` | 1 | 30.44957 | 21.13752 | 1.441x | 0.996x |
| Linux | float32 | strided | `{"channel_axis": -1.0, "kC": 0.5, "kL": 2.0}` | 4 | 30.45249 | 21.21830 | 1.435x | 0.991x |
| Linux | float32 | scalar | `{"channel_axis": -1.0}` | 1 | 0.16327 | 0.03760 | 4.342x | 0.938x |
| Linux | float32 | scalar | `{"channel_axis": -1.0}` | 4 | 0.16368 | 0.03814 | 4.291x | 0.914x |
| Linux | float32 | scalar | `{"channel_axis": -1.0, "kC": 0.5, "kL": 2.0}` | 1 | 0.16388 | 0.03840 | 4.267x | 0.932x |
| Linux | float32 | scalar | `{"channel_axis": -1.0, "kC": 0.5, "kL": 2.0}` | 4 | 0.16407 | 0.03852 | 4.259x | 0.925x |
| Linux | float32 | weight-control | `{"kC": 0.0}` | 1 | 59.08088 | 37.82234 | 1.562x | 0.997x |
| Linux | float32 | weight-control | `{"kC": 0.0}` | 4 | 58.32202 | 12.36768 | 4.716x | 1.009x |
| Linux | float32 | weight-control | `{"kH": 0.0}` | 1 | 58.65636 | 37.81111 | 1.551x | 1.000x |
| Linux | float32 | weight-control | `{"kH": 0.0}` | 4 | 58.33068 | 12.51430 | 4.661x | 0.990x |
| Linux | float32 | weight-control | `{"kC": 0.699999988079071, "kL": 2.0}` | 1 | 58.83506 | 37.84595 | 1.555x | 0.997x |
| Linux | float32 | weight-control | `{"kC": 0.699999988079071, "kL": 2.0}` | 4 | 58.29961 | 12.32255 | 4.731x | 1.007x |
| Linux | float64 | contiguous | `{"channel_axis": -1.0}` | 1 | 90.84651 | 32.28526 | 2.814x | 1.006x |
| Linux | float64 | contiguous | `{"channel_axis": -1.0}` | 4 | 91.58113 | 11.51459 | 7.953x | 0.986x |
| Linux | float64 | contiguous | `{"channel_axis": -1.0, "kC": 0.5, "kL": 2.0}` | 1 | 90.85119 | 32.29182 | 2.813x | 1.006x |
| Linux | float64 | contiguous | `{"channel_axis": -1.0, "kC": 0.5, "kL": 2.0}` | 4 | 90.56044 | 11.45908 | 7.903x | 0.998x |
| Linux | float64 | channel-first | `{"channel_axis": 0.0}` | 1 | 90.07485 | 32.13936 | 2.803x | 1.006x |
| Linux | float64 | channel-first | `{"channel_axis": 0.0}` | 4 | 90.13552 | 11.43195 | 7.885x | 0.988x |
| Linux | float64 | channel-first | `{"channel_axis": 0.0, "kC": 0.5, "kL": 2.0}` | 1 | 90.13428 | 32.15951 | 2.803x | 1.006x |
| Linux | float64 | channel-first | `{"channel_axis": 0.0, "kC": 0.5, "kL": 2.0}` | 4 | 90.31782 | 11.45972 | 7.881x | 0.993x |
| Linux | float64 | strided | `{"channel_axis": -1.0}` | 1 | 47.40305 | 18.35138 | 2.583x | 1.006x |
| Linux | float64 | strided | `{"channel_axis": -1.0}` | 4 | 47.50166 | 18.42050 | 2.579x | 1.005x |
| Linux | float64 | strided | `{"channel_axis": -1.0, "kC": 0.5, "kL": 2.0}` | 1 | 48.17985 | 18.30866 | 2.632x | 1.005x |
| Linux | float64 | strided | `{"channel_axis": -1.0, "kC": 0.5, "kL": 2.0}` | 4 | 47.66108 | 18.36269 | 2.596x | 1.009x |
| Linux | float64 | scalar | `{"channel_axis": -1.0}` | 1 | 0.15000 | 0.03196 | 4.693x | 0.946x |
| Linux | float64 | scalar | `{"channel_axis": -1.0}` | 4 | 0.14813 | 0.03169 | 4.674x | 0.972x |
| Linux | float64 | scalar | `{"channel_axis": -1.0, "kC": 0.5, "kL": 2.0}` | 1 | 0.14963 | 0.03255 | 4.597x | 0.904x |
| Linux | float64 | scalar | `{"channel_axis": -1.0, "kC": 0.5, "kL": 2.0}` | 4 | 0.14934 | 0.03210 | 4.652x | 0.939x |
| Linux | float32 | Sharma-boundaries | `{}` | 1 | 0.49636 | 0.31450 | 1.578x | 0.951x |
| Linux | float32 | Sharma-boundaries | `{}` | 4 | 0.49908 | 0.31552 | 1.582x | 0.943x |
| Windows | float32 | contiguous | `{"channel_axis": -1.0}` | 1 | 97.45810 | 46.39600 | 2.101x | 1.001x |
| Windows | float32 | contiguous | `{"channel_axis": -1.0}` | 4 | 98.28500 | 15.83040 | 6.209x | 0.997x |
| Windows | float32 | contiguous | `{"channel_axis": -1.0, "kC": 0.5, "kL": 2.0}` | 1 | 97.65810 | 46.17070 | 2.115x | 1.005x |
| Windows | float32 | contiguous | `{"channel_axis": -1.0, "kC": 0.5, "kL": 2.0}` | 4 | 97.74130 | 15.90350 | 6.146x | 0.993x |
| Windows | float32 | channel-first | `{"channel_axis": 0.0}` | 1 | 97.02280 | 45.69050 | 2.123x | 0.999x |
| Windows | float32 | channel-first | `{"channel_axis": 0.0}` | 4 | 98.71920 | 15.65740 | 6.305x | 1.008x |
| Windows | float32 | channel-first | `{"channel_axis": 0.0, "kC": 0.5, "kL": 2.0}` | 1 | 96.90580 | 45.55660 | 2.127x | 1.001x |
| Windows | float32 | channel-first | `{"channel_axis": 0.0, "kC": 0.5, "kL": 2.0}` | 4 | 97.72980 | 16.10040 | 6.070x | 0.982x |
| Windows | float32 | strided | `{"channel_axis": -1.0}` | 1 | 45.63510 | 25.44760 | 1.793x | 1.002x |
| Windows | float32 | strided | `{"channel_axis": -1.0}` | 4 | 45.83780 | 25.53570 | 1.795x | 1.003x |
| Windows | float32 | strided | `{"channel_axis": -1.0, "kC": 0.5, "kL": 2.0}` | 1 | 45.81560 | 25.52390 | 1.795x | 1.006x |
| Windows | float32 | strided | `{"channel_axis": -1.0, "kC": 0.5, "kL": 2.0}` | 4 | 45.90890 | 25.52580 | 1.799x | 0.999x |
| Windows | float32 | scalar | `{"channel_axis": -1.0}` | 1 | 0.13414 | 0.02735 | 4.904x | 0.921x |
| Windows | float32 | scalar | `{"channel_axis": -1.0}` | 4 | 0.13505 | 0.02837 | 4.760x | 0.896x |
| Windows | float32 | scalar | `{"channel_axis": -1.0, "kC": 0.5, "kL": 2.0}` | 1 | 0.13370 | 0.02753 | 4.857x | 0.898x |
| Windows | float32 | scalar | `{"channel_axis": -1.0, "kC": 0.5, "kL": 2.0}` | 4 | 0.13303 | 0.02632 | 5.055x | 0.948x |
| Windows | float32 | weight-control | `{"kC": 0.0}` | 1 | 99.26060 | 46.25890 | 2.146x | 1.006x |
| Windows | float32 | weight-control | `{"kC": 0.0}` | 4 | 98.75570 | 16.05110 | 6.153x | 0.979x |
| Windows | float32 | weight-control | `{"kH": 0.0}` | 1 | 99.09260 | 46.23410 | 2.143x | 1.003x |
| Windows | float32 | weight-control | `{"kH": 0.0}` | 4 | 99.09140 | 15.86280 | 6.247x | 0.994x |
| Windows | float32 | weight-control | `{"kC": 0.699999988079071, "kL": 2.0}` | 1 | 99.06060 | 46.36760 | 2.136x | 1.000x |
| Windows | float32 | weight-control | `{"kC": 0.699999988079071, "kL": 2.0}` | 4 | 99.03330 | 15.85240 | 6.247x | 0.999x |
| Windows | float64 | contiguous | `{"channel_axis": -1.0}` | 1 | 138.34930 | 34.72920 | 3.984x | 1.001x |
| Windows | float64 | contiguous | `{"channel_axis": -1.0}` | 4 | 139.15470 | 12.45520 | 11.172x | 0.997x |
| Windows | float64 | contiguous | `{"channel_axis": -1.0, "kC": 0.5, "kL": 2.0}` | 1 | 138.01600 | 35.00600 | 3.943x | 0.992x |
| Windows | float64 | contiguous | `{"channel_axis": -1.0, "kC": 0.5, "kL": 2.0}` | 4 | 139.52650 | 12.44130 | 11.215x | 0.996x |
| Windows | float64 | channel-first | `{"channel_axis": 0.0}` | 1 | 136.43470 | 34.35890 | 3.971x | 1.001x |
| Windows | float64 | channel-first | `{"channel_axis": 0.0}` | 4 | 136.48840 | 12.32150 | 11.077x | 1.004x |
| Windows | float64 | channel-first | `{"channel_axis": 0.0, "kC": 0.5, "kL": 2.0}` | 1 | 137.45170 | 34.46540 | 3.988x | 1.000x |
| Windows | float64 | channel-first | `{"channel_axis": 0.0, "kC": 0.5, "kL": 2.0}` | 4 | 137.00240 | 12.24340 | 11.190x | 0.998x |
| Windows | float64 | strided | `{"channel_axis": -1.0}` | 1 | 74.41270 | 21.12200 | 3.523x | 0.998x |
| Windows | float64 | strided | `{"channel_axis": -1.0}` | 4 | 74.73140 | 21.17680 | 3.529x | 0.999x |
| Windows | float64 | strided | `{"channel_axis": -1.0, "kC": 0.5, "kL": 2.0}` | 1 | 74.95250 | 21.35020 | 3.511x | 0.996x |
| Windows | float64 | strided | `{"channel_axis": -1.0, "kC": 0.5, "kL": 2.0}` | 4 | 74.16830 | 21.16290 | 3.505x | 1.004x |
| Windows | float64 | scalar | `{"channel_axis": -1.0}` | 1 | 0.12127 | 0.02264 | 5.356x | 0.950x |
| Windows | float64 | scalar | `{"channel_axis": -1.0}` | 4 | 0.12087 | 0.02323 | 5.204x | 0.908x |
| Windows | float64 | scalar | `{"channel_axis": -1.0, "kC": 0.5, "kL": 2.0}` | 1 | 0.11970 | 0.02323 | 5.152x | 0.939x |
| Windows | float64 | scalar | `{"channel_axis": -1.0, "kC": 0.5, "kL": 2.0}` | 4 | 0.11997 | 0.02299 | 5.219x | 0.929x |
| Windows | float32 | Sharma-boundaries | `{}` | 1 | 0.40252 | 0.36203 | 1.112x | 0.960x |
| Windows | float32 | Sharma-boundaries | `{}` | 4 | 0.40114 | 0.36177 | 1.109x | 0.963x |
