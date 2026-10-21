#pragma once
#include "convert.hpp"
#include <array>
#include <optional>

namespace skimagexx {
enum class PerceptualTransform { rgb2hsv, hsv2rgb, xyz2lab, lab2xyz, rgb2lab, lab2rgb, lab2lch, lch2lab,
                                 xyz2luv, luv2xyz, rgb2luv, luv2rgb };
struct PerceptualInfo {
    std::size_t negative_z = 0;
    bool unsupported = false; // HSV nonfinite input, RGB->HSV nonfinite intermediates, or huge hue cast.
};
// Three-channel pixels (Lab/LCh may have extra channels, preserved after dtype
// normalization). Output is contiguous float32 for float32 input, float64 otherwise.
// Strides count input elements. Negative/zero strides are supported. Input and
// output must not overlap. HSV pixels flagged unsupported are zero-filled.
// Optional Luv u'/v' white metadata preserves a caller's dot-product rounding;
// omitted metadata is calculated from the supplied XYZ white point in C++.
PerceptualInfo perceptual_into(const void* input, void* output, std::size_t count,
    PixelType input_type, PerceptualTransform transform,
    std::array<double,3> white = {0.95047,1.0,1.08883}, std::size_t channels = 3,
    std::ptrdiff_t pixel_stride = 3, std::ptrdiff_t channel_stride = 1,
    std::optional<std::array<double,2>> uv_reference = std::nullopt);
// NumPy's one-pixel negative-Z case promotes float32 Lab intermediates to
// float64 before applying the inverse curve. Only lab2xyz/lab2rgb are accepted.
void lab_scalar_negative_z_into(const float* input, double* output,
    PerceptualTransform transform, std::array<double,3> white, std::ptrdiff_t channel_stride = 1);
} // namespace skimagexx
