#pragma once

#include <cstddef>

namespace skimagexx {

// Prepared constant-border bilateral stencil for float or double. The input
// is C-order HWC with either one or three channels and two spatial dimensions
// in [2, 512]. The caller supplies aligned, complete and disjoint buffers,
// normalized finite input/cval and the exact input maximum in [1/256, 1].
// Odd window sizes are in [5, 25]. Color LUT bins are in [4, 65536].
// The spatial LUT has (window + 1)^2 elements and uses upstream's flattened
// indexing with stride window. Call under round-to-nearest.
template<class T>
void denoise_bilateral_constant_into(
    const T* image, std::size_t image_count, std::size_t height, std::size_t width,
    std::size_t channels, double maximum, std::size_t window, double cval,
    const T* color_lut, std::size_t bins, const T* spatial_lut, std::size_t spatial_count,
    T* result, std::size_t result_count);

}  // namespace skimagexx
