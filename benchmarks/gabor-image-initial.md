# Gabor image filtering

Source `2244f39` implements `filters.gabor` with fused real/imaginary convolution,
one prepared input, SIMD output pixels and optional global workers. Accumulation
keeps ndimage's reversed row-major order and independent epsilon cutoffs. Integer
inputs use libm kernel values with SIMD coordinate arithmetic: the approximate
transcendental path's coefficient ULPs can otherwise change integer casts.

[Run 39863482919](https://github.com/bipxx/scikit-imagexx/actions/runs/39863482919)
measures 72 fixed cases per OS: all 12 real dtypes (bool, every integer width and
float16/32/64), each with contiguous, strided, Fortran, tiny, large-kernel and
constant-boundary inputs. Standard dimensions are 192x193; seed7041 and all
workload definitions are preserved. Worker caps are one/four, seven raw samples
are retained, reference pools are fixed to one and native fallback is disabled.
Allocation, kernel generation, input preparation, Python dispatch and output
destruction are included. Downloaded package/source/binary manifests agree.

| OS | one-worker upstream/native geometric mean | minimum one-worker speedup | cases slower than upstream |
|---|---:|---:|---:|
| Linux | 8.977176x | 4.964256x | 0 / 72 |
| Windows | 9.006282x | 5.026933x | 0 / 72 |

Raw reports: [Linux](gabor-image-initial-linux.json),
[Windows](gabor-image-initial-windows.json). All 288 timed rows pass parity.
Integer/bool result bytes, including full 64-bit magnitudes, match exactly.
Float32 outputs (including promoted half input) are exact in this corpus;
the maximum float64 absolute error is 8.327e-17. General floating budgets remain
2e-6 (float32) and 2e-13 (float64); these scoped results do not prove every
possible call or hardware combination.

The five lowest one-worker ratios on each OS are retained below. Small
constant-boundary kernels have the least work over which to amortize planning
and buffer preparation; they remain faster than the reference.

| OS | dtype | case | upstream/native |
|---|---|---|---:|
| Linux | uint32 | constant | 4.964256x |
| Linux | int64 | constant | 5.243074x |
| Linux | float16 | constant | 5.369494x |
| Linux | float64 | constant | 5.782494x |
| Linux | bool | constant | 5.789367x |
| Windows | int64 | constant | 5.026933x |
| Windows | float64 | constant | 5.075333x |
| Windows | float16 | constant | 5.145306x |
| Windows | int32 | constant | 5.271301x |
| Windows | uint8 | constant | 5.360100x |

The same revision also reruns the existing 36-case `gabor_kernel` corpus to
check the shared helper/provider changes. All one-worker controls beat upstream,
with geometric means 3.442961x Linux /3.648453x Windows and minima
3.119142x /3.218617x. Raw controls: [Linux](gabor-image-kernel-control-linux.json),
[Windows](gabor-image-kernel-control-windows.json). These are direct upstream
comparisons, not paired measurements against the prior native package.

Dedicated checks pass 1,364 image-filter tests plus 440 kernel tests per SIMD
mode on Windows and minimum NumPy, alongside 4,781 API/metadata/namespace checks
per runtime. All 72 workloads also pass untimed parity checks locally. Linux
normal/dynamic/portable/ASan+UBSan configurations pass 3/3 native tests; a separate
sanitized regression covers the zero-stride sentinel path. Independent relocated
installed C++ consumers pass 2/2 on Windows and Linux, including OpenMP-disabled
Linux, with Python discovery disabled.

Earlier failures and exact source/packages are preserved. Kernel provider
rounding was corrected without relaxing integer parity. The native zero-stride
path reproduces ndimage's coordinate-1 sentinel collision and its reused
interior-region offsets. Distant negative reflection periods that can address
before a nonzero-stride reference buffer retain explicit upstream fallback.

[Full source CI39863482376](https://github.com/bipxx/scikit-imagexx/actions/runs/39863482376)
has passed all three jobs: Windows, Linux, and minimum NumPy. Source `2244f39`
is retained after both performance and full CI validation; its exact packages,
source, failed probes, and raw samples remain preserved.
