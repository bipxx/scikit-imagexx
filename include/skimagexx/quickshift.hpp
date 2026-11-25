#pragma once

#include <cstddef>

namespace skimagexx {

// Contiguous, finite, preprocessed image in row/column/channel order.
// noise and labels have height * width entries and must not alias image.
// The kernel size must be >= 1, and max_distance must be finite and nonnegative.
template<class T>
void quickshift_into(const T* image, const T* noise, std::ptrdiff_t* labels,
                     std::ptrdiff_t height, std::ptrdiff_t width, std::ptrdiff_t channels,
                     T kernel_size, T max_distance);

extern template void quickshift_into(const float*, const float*, std::ptrdiff_t*,
                                    std::ptrdiff_t, std::ptrdiff_t, std::ptrdiff_t, float, float);
extern template void quickshift_into(const double*, const double*, std::ptrdiff_t*,
                                    std::ptrdiff_t, std::ptrdiff_t, std::ptrdiff_t, double, double);

} // namespace skimagexx
