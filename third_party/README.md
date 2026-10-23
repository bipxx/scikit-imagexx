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

Median selection models the first-pivot Hoare comparison order and numeric cast
stages in SciPy's `ndimage/src/ni_filters.c` (v1.18.1). The existing
`scipy-ndimage-LICENSE.txt` retains its BSD notice. The generalized median
kernel and SIMD networks are implemented in C++ without Python image arithmetic.

Multi-Otsu float32 moment initialization, objective accumulation order and first-maximum selection follow scikit-image 0.26.0 (`filters/_multiotsu.pyx`), covered by `scikit-image-LICENSE.txt`. Native candidate scoring, deterministic worker partitioning and bounded table storage are implemented in this repository.

CLAHE clipping and redistribution, 14-bit quantization and interpolation order follow scikit-image 0.26.0 (`exposure/_adapthist.py`), adapted there from Karel Zuiderveld, Graphics Gems IV and relicensed with permission under the Modified BSD license. See `scikit-image-LICENSE.txt`. This implementation uses unpadded tile access, native histogram rows and runtime-dispatched AVX2 quantization/interpolation.

The private resize/rescale spline foundation in `src/spline.cpp` adapts the B-spline weights, poles and coefficient recurrences from [SciPy v1.18.1 `ni_splines.c`](https://github.com/scipy/scipy/blob/v1.18.1/scipy/ndimage/src/ni_splines.c). `scipy-ndimage-LICENSE.txt` retains the BSD notice. Native axis traversal, scratch ownership and optional global worker partitioning are implemented here. An AVX2 kernel advances four independent coefficient lines together without gather/scatter scratch; SIMD/scalar outputs are checked bitwise. Legacy reflected initialization reproduces the short-axis endpoint aliasing fixed in [SciPy 1.18.0, gh-24615](https://github.com/scipy/scipy/pull/24615); the modern default uses the corrected recurrence. This foundation does not yet register native resize/rescale coverage.
