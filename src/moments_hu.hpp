#pragma once

#include <cstddef>

namespace skimagexx::detail {

// Private leaf: strides are in elements. The caller provides aligned, live
// storage for logical indices [0, 3] x [0, 3], with representable offsets,
// and seven writable output elements disjoint from the input. Negative and
// zero input strides are allowed. Python dtype/shape/storage checks belong at
// the binding boundary. The seven outputs use the source Cython operation order.
void moments_hu(const float* data, std::ptrdiff_t row_stride,
                std::ptrdiff_t col_stride, float* out7);
void moments_hu(const double* data, std::ptrdiff_t row_stride,
                std::ptrdiff_t col_stride, double* out7);

}  // namespace skimagexx::detail
