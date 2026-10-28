# Pinned PocketFFT
Upstream https://github.com/scipy/pocketfft commit9367142748fcc9696a1c9e5a99b76ed9897c9daa.
Original pocketfft_hdronly.h (SHA256 50f4ddbe60432325f560c706b6f1c49f9e0949fc7f20ae6499f2299e37d9790c) and LICENSE.md are exact upstream bytes.
Compiled pocketfft_openmp.h adds only the SKIMAGEXX_POCKETFFT_OPENMP threading branch; inverse replacement restores the original header. FFT math is unchanged.
The adapter uses an explicit global-worker snapshot and actual OpenMP team, restores thread-local context and propagates failures after joining. No private pool. Portable builds disable multithreading. Adapter plan cache is explicitly16, matching SciPy1.15.3 meson.build: https://raw.githubusercontent.com/scipy/scipy/v1.15.3/scipy/fft/_pocketfft/meson.build . The upstream mutex-protected LRU retains at most16 plans per type/namespace, keyed by length; this is not a byte cap. No custom cache or thread pool is added. Cache0 evidence is retained separately; no speed claim follows from this setting.
Baseline disables vectors. Separate AVX2 instantiation is selected by existing has_avx2 policy; compiler header vectors are not guaranteed on MSVC.
BSD-3-Clause notices are retained in both headers and LICENSE.md.

Private, unaccepted angle experiment: the compiled header additionally uses
`src/fft_angle.hpp` for double twiddle angles on MSVC x64 in nearest rounding.
It reconstructs the source literal's 64-significand quarter-pi division and
final binary64 rounding using integer arithmetic. Other modes/platforms retain
the existing expression. The exact upstream header is untouched. This branch
is an experimental overlay, not whole-product acceptance or a performance claim.
