#pragma once
#include "convert.hpp"

namespace skimagexx {
// Raw Lab values, without integer intensity normalization. The first three
// channels participate; strides are in input elements and may be negative/zero.
// Output is float32 when both inputs are float32, float64 otherwise. Buffers
// must not overlap. count is the number of broadcasted pixel pairs.
void delta_e76_into(const void* first,const void* second,void* output,std::size_t count,
    PixelType first_type,PixelType second_type,std::ptrdiff_t first_pixels=3,
    std::ptrdiff_t first_channels=1,std::ptrdiff_t second_pixels=3,std::ptrdiff_t second_channels=1);
}
