#pragma once
#include "convert.hpp"
#include <cstddef>
#include <vector>
namespace skimagexx {
enum class GaussianBorder {reflect,constant,nearest,mirror,wrap};
struct GaussianPass {
    std::size_t axis;
    std::vector<double> weights; // odd, symmetric, in increasing offset order
    GaussianBorder border=GaussianBorder::nearest;
    double cval=0;
};
// Native contiguous/aligned real buffers; output is float32 or float64 and does
// not overlap input. Passes retain axis order and round to output precision
// after each axis, including singleton axes. Empty shapes are permitted.
// The no-pass copy returns NumPy-compatible floating-cast warning metadata.
ConversionInfo gaussian_nd_into(const void*,void*,PixelType,PixelType,
    const std::vector<std::size_t>&,const std::vector<GaussianPass>&,bool preserve_range=false);
// Difference may reuse its first buffer; unsharp output is separate storage.
void gaussian_difference_into(const void*,const void*,void*,std::size_t,PixelType);
void gaussian_unsharp_into(const void*,const void*,void*,std::size_t,PixelType,PixelType,double,bool,bool,double);

}
