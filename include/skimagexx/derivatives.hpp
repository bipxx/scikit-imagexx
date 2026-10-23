#pragma once
#include "gaussian_nd.hpp"
#include <array>
namespace skimagexx {
// Floating contiguous aligned buffers of height*width elements, no overlap.
// Caller-provided scratch avoids repeated allocations for derivative chains.
// This is scipy.ndimage's unnormalized Sobel operator, including per-axis
// rounding. It does not normalize image values or construct a magnitude.
void sobel_derivative_2d_into(const void* input,void* output,void* scratch,
    std::size_t height,std::size_t width,bool single,std::size_t axis,
    const std::array<GaussianBorder,2>& borders,double cval=0);
}
