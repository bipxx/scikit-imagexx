#pragma once
#include "convert.hpp"

namespace skimagexx {
enum class StainTransform { separate, combine };
// Three-channel input; output has output_channels values per pixel. Matrix is
// output-major: weights[3*j+i] is the input-i/output-j coefficient. Buffers
// must be aligned, contiguous at the supplied element strides, and not overlap.
// Input normalization follows img_as_float. Float32 logarithms retain float32
// rounding before the selected scaling/accumulation precision. Defaults match
// NumPy 2.x with the standard float64 stain matrices.
void stain_transform_into(const void* input,void* output,std::size_t count,PixelType input_type,
    StainTransform transform,const double* weights,std::size_t output_channels=3,
    PixelType intermediate_type=PixelType::float64,PixelType output_type=PixelType::float64,
    std::ptrdiff_t pixel_stride=3,std::ptrdiff_t channel_stride=1,
    double log_adjust=13.815510557964274);
}
