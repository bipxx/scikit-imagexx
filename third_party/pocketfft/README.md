# Pinned PocketFFT
Upstream https://github.com/scipy/pocketfft commit9367142748fcc9696a1c9e5a99b76ed9897c9daa.
Original pocketfft_hdronly.h (SHA256 50f4ddbe60432325f560c706b6f1c49f9e0949fc7f20ae6499f2299e37d9790c) and LICENSE.md are exact upstream bytes.
Compiled pocketfft_openmp.h adds the SKIMAGEXX_POCKETFFT_OPENMP threading branch and the MSVC x64 angle correction described below. The original upstream header remains unchanged.
The adapter uses an explicit global-worker snapshot and actual OpenMP team, restores thread-local context and propagates failures after joining. No private pool. Portable builds disable multithreading. Adapter plan cache is explicitly16, matching SciPy1.15.3 meson.build: https://raw.githubusercontent.com/scipy/scipy/v1.15.3/scipy/fft/_pocketfft/meson.build . The upstream mutex-protected LRU retains at most16 plans per type/namespace, keyed by length; this is not a byte cap. No custom cache or thread pool is added. Cache0 evidence is retained separately; no speed claim follows from this setting.
Baseline disables vectors. Separate AVX2 instantiation is selected by existing has_avx2 policy; compiler header vectors are not guaranteed on MSVC.
BSD-3-Clause notices are retained in both headers and LICENSE.md.

The compiled header uses
`src/fft_angle.hpp` for double twiddle angles on MSVC x64 in nearest rounding.
It reconstructs the source literal's 64-significand quarter-pi division and
final binary64 rounding using integer arithmetic. Other modes/platforms retain
the existing expression. The exact upstream header is untouched.

The bounded same-host Windows run40685077347 reproduced all8 baseline inverse-LPI
failures and passed all8 after this correction, with identical forward inputs.
Both normal/scalar modes passed80 frozen LPI specifications and87 Butterworth
specifications with workers1/4; the forward F/G arrays became bitwise equal to
the pinned reference. Frozen allowances were unchanged. This supports the
merged correction; full whole-product qualification is still pending and no
performance claim follows from this correctness run.

The standalone `fft_angle_native` CTest reads the committed363 pre-candidate
reference words in `tests/data/fft-angle-nearest.txt`, including uint64 denominator
boundaries. It checks exact results in four rounding modes and preserves FP flags.
