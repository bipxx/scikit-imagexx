#pragma once

namespace skimagexx::detail {

// Private default-order stage. Caller validates native aligned row-major 4x4
// input and disjoint writable output of 16 elements. Mass is in [1e-6,1e6];
// every input is zero or has magnitude in [1e-12,1e12], and all are finite.
// float32_power selects NumPy2's float32 scalar power (float input only).
// All final divisions are double, followed by the output dtype conversion.
// No callbacks, allocation, thread creation, or worker-state changes occur here.
void moments_normalize_default(const float* input16, bool float32_power, float* out16);
void moments_normalize_default(const double* input16, bool float32_power, double* out16);

}  // namespace skimagexx::detail

#ifdef SKIMAGEXX_AVX2
namespace skimagexx::simd {
// Seven degree-indexed denominators; entries 0 and 1 are exactly one.
void moments_normalize_default(const float* input16, const double* denominators7, float* out16);
void moments_normalize_default(const double* input16, const double* denominators7, double* out16);
}  // namespace skimagexx::simd
#endif
