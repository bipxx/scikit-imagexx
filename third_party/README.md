# Upstream attribution

The compatibility kernels follow the behavior and numerical operation order of
scikit-image 0.26.0. The accompanying license file retains that release's upstream
copyright and license notices. It is included in Python wheels and standalone
C++ installations.

The CIEDE2000 numerical fixture in `tests/data/ciede2000-sharma.txt` comes from
[Gaurav Sharma's supplementary test data](https://hajim.rochester.edu/ece/sites/gsharma/ciede2000/dataNprograms/ciede2000testdata.txt)
for Sharma, Wu and Dalal's implementation notes, DOI 10.1002/col.20070.

The float32 logarithm in `src/threshold.cpp` follows the range reduction,
rational coefficients and fused evaluation used in NumPy's
[`loops_exponent_log.dispatch.c.src`](https://github.com/numpy/numpy/blob/v2.4.0/numpy/_core/src/umath/loops_exponent_log.dispatch.c.src)
and [`npy_simd_data.h`](https://github.com/numpy/numpy/blob/v2.4.0/numpy/_core/src/umath/npy_simd_data.h).
The included `numpy-LICENSE.txt` preserves NumPy's BSD notice. This rounding is
observable in Yen's first-maximum threshold selection. The standalone C++ path
uses scalar `std::fma` without requiring a hardware FMA instruction; Python uses
fixed metadata probes to select this approximation or platform libm according
to the installed NumPy provider. Unrecognized providers require explicit fallback.

Yen's compatibility logarithm has an optional eight-lane AVX2/FMA path in
`src/simd_threshold_fma.cpp`. It is dispatched only after the runtime AVX2/OS
checks and a separate FMA CPU-feature check. CPUs without FMA and builds with
SIMD disabled use scalar `std::fma`; the library has no hardware FMA requirement.
Only this translation unit uses the additional `-mfma` flag on GCC/Clang.
