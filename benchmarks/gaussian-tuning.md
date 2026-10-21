# Gaussian interior-span experiment

The fixed workload contains 225 Gaussian, difference-of-Gaussians and unsharp-mask calls at size 512, including all twelve real dtypes, layouts, boundaries, channels and output precision. Both libraries use one thread. Each case checks numerical, shape, dtype and warning parity with fallback disabled before timing. Seven warmed samples include Python dispatch, conversion, copies, allocation and destruction.

Only the convolution interior plan changes. Away from an axis boundary, neighbor addresses are fixed linear offsets. Processing several adjacent axis coordinates in one contiguous SIMD span avoids scalar three-pixel calls for channel-last images. Border mapping and the ordered double-precision coefficient sum remain unchanged.

| Observation | Source revision | Native latency geometric mean, ms | Faster than scikit-image |
|---|---|---:|---:|
| Baseline | `3bc3bd3` | 2.141694 | 217/225 |
| Optimized | `7681dcf` | 2.040915 | 224/225 |
| Confirmation | `7681dcf` | 2.041301 | 224/225 |

Keep: confirmation reduces the fixed aggregate native latency by 4.69%. The aggregate combines different operation sizes; per-case results and raw samples remain in the [JSON archive](gaussian-tuning.json).

| Channel-last call | Dtype | Baseline vs scikit-image | Confirmation vs scikit-image | Baseline C++ / confirmed C++ |
|---|---|---:|---:|---:|
| `filters.gaussian` | uint8 | 0.762x | 2.589x | 3.850x |
| `filters.difference_of_gaussians` | uint8 | 0.713x | 2.430x | 4.013x |
| `filters.unsharp_mask` | uint8 | 1.280x | 2.843x | 2.393x |
| `filters.gaussian` | float32 | 0.492x | 2.977x | 7.694x |
| `filters.difference_of_gaussians` | float32 | 0.534x | 2.854x | 7.312x |
| `filters.unsharp_mask` | float32 | 0.685x | 4.430x | 9.230x |
| `filters.gaussian` | float64 | 0.608x | 3.359x | 3.914x |
| `filters.difference_of_gaussians` | float64 | 0.727x | 2.006x | 2.263x |
| `filters.unsharp_mask` | float64 | 1.188x | 3.283x | 2.066x |

The optimized first pass has one slower float64 Fortran Gaussian case (0.663x), which measures 1.54x in confirmation. Confirmation retains a uint8 single-column Gaussian at 0.9994x. Ordinary system load and CPU scheduling affect individual timings; all observations are retained.

Validation: baseline full Windows suite passed 29,341 cases in each SIMD mode. After the loop change, 6,492 targeted Gaussian, signature, alias and dtype-metadata cases passed in each SIMD mode on both Windows and Linux. Native and AVX2/OpenMP-disabled C++ checks passed on both systems; Linux AddressSanitizer and UndefinedBehaviorSanitizer checks passed. These tests include large channel-last arrays, all five borders and one/four-worker execution. The complete [Windows/Linux CI run](https://github.com/bipxx/scikit-imagexx/actions/runs/39640257292) subsequently passed 29,425 tests in each SIMD mode on each system, together with native and portable C++ checks (source revision `f1fa942`, with the same native implementation as `7681dcf`).

Convolution follows the symmetric double-accumulation convention in [SciPy ndimage](https://github.com/scipy/scipy/blob/v1.18.1/scipy/ndimage/src/ni_filters.c), with its notice shipped in `third_party/scipy-ndimage-LICENSE.txt`. Coefficient metadata preserves typed sigma arithmetic. Image filtering, normalization, subtraction and sharpening run in C++.

This archive preserves a fixed experiment. The [current verification report](STATUS.md) is rebuilt separately and may have different timings. Reproduce with `python benchmarks/gaussian.py --threads 1 --repeat 7`; builds and tests must finish before measuring.

For explicit output buffers, the initial and first optimized snapshots record `max_abs_error` after timing reuses the shared buffer. Parity assertions occurred before timing in all snapshots. Confirmation captures that diagnostic before timing. Workloads and measured calls are unchanged.
