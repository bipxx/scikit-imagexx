# Kitchen–Rosenfeld: initial measurements

Measured source **ee733e2506f53b859929d1a80e727aaefc5069a1** implements
`feature.corner_kitchen_rosenfeld` in C++17. It reuses native Sobel passes,
omits the unused mixed derivative, reuses scratch storage and fuses response
arithmetic with AVX2. Global parallel workers are optional and default to one.

The isolated [measurement](https://github.com/bipxx/scikit-imagexx/actions/runs/39883035493)
passed on both platforms against **scikit-image 0.26.0**. Exact source and
binary hashes and seven samples per implementation per row were verified.
Timings include allocation, input preparation and Python dispatch.

| One-worker result | Linux | Windows |
|---|---:|---:|
| Geometric-mean speedup | 7.043285× | 6.175798× |
| Slowest case speedup | 5.295321× | 3.294634× |
| Cases faster than upstream | 120/120 | 120/120 |

All 480 timed rows, including worker-cap-four controls, have zero numerical
absolute difference. Linux's slowest cases use strided float16/32 arrays;
Windows's slowest cases use boolean Fortran, reflect and broadcast inputs.
No adverse cases were removed.

The fixed corpus covers 12 real dtypes × 10 cases: contiguous, strided,
Fortran, tiny, broadcast, reflect, mirror, wrap, nearest and mixed border
modes with a nonzero constant. Images are 256×257, seed8311; worker caps are
1 and 4. Reference thread pools are limited to one.

Native scope includes ordinary real 2-D arrays/memmaps, boolean and all integer
widths, float16/32/64, ndimage border modes and grid aliases. Integer values
are cast without normalization, as upstream requires. Float32 intermediate
rounding, zero denominator masks, shape/layout/memmap behavior are preserved.
Nonfinite/extreme inputs, unsupported protocols and non-default underflow
policies use explicit whole-function fallback. This is partial API coverage.

Production packages pass 6,620 shared/dedicated/metadata/namespace checks per
runtime. Combined corner/tensor suites pass 2,140 checks per SIMD mode on both
runtimes, including minimum NumPy1.24. Fixed-corpus untimed parity checks pass
240 cases per runtime. Native tests pass 7/7 on Windows and Linux with
OpenMP/AVX2 enabled and disabled; relocated SDK consumers pass 6/6 on all
three configurations. ASan/UBSan pass 7/7 normally and with constrained
OpenMP teams. No local performance timings were run.

Raw samples: [Linux](kitchen-initial-linux.json),
[Windows](kitchen-initial-windows.json). Source archives, exact packages,
manifests and validation logs remain in ignored build checkpoints.
The [full CI](https://github.com/bipxx/scikit-imagexx/actions/runs/39883029028)
is still running; retention awaits all three jobs.
