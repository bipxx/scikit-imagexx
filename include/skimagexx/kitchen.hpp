#pragma once
#include "derivatives.hpp"
namespace skimagexx {
// Real aligned contiguous input; float32 returns float32, other source types
// return float64. Integers are cast without range normalization, matching the
// Kitchen-Rosenfeld API. Output does not overlap input. False means unsafe
// arithmetic/nonfinite input; discard output and use compatibility.
bool kitchen_rosenfeld_into(const void* image,void* output,PixelType source,
    std::size_t height,std::size_t width,const std::array<GaussianBorder,2>& borders,
    double cval=0);
}
