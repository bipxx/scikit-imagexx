#pragma once
#include "convert.hpp"

namespace skimagexx {
enum class ColorTransform {
    rgb2yuv, yuv2rgb, rgb2yiq, yiq2rgb, rgb2ypbpr, ypbpr2rgb,
    rgb2ycbcr, ycbcr2rgb, rgb2ydbdr, ydbdr2rgb, rgb2rgbcie, rgbcie2rgb,
    rgb2xyz, xyz2rgb
};
// Contiguous triples; count is the number of pixels. Input/output must not
// overlap. Output contains 3*count float32 elements for float32 input and
// float64 elements otherwise. Empty buffers are allowed. Uses global workers.
// Boolean input is supported except inverse YCbCr (upstream rejects subtraction).
void color_transform_into(const void* input, void* output, std::size_t count,
                          PixelType input_type, ColorTransform transform);
// Floating strided buffers. Strides count input elements, not bytes; negative
// and zero strides are supported. Input must cover every addressed element.
// Output is contiguous triples and must not overlap input.
template<class T> void color_transform_strided_into(const T* input, T* output,
    std::size_t count, std::ptrdiff_t pixel_stride, std::ptrdiff_t channel_stride,
    ColorTransform transform);
} // namespace skimagexx
