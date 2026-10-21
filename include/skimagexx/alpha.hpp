#pragma once
#include "convert.hpp"
#include <cstddef>
namespace skimagexx {
// Row offsets and strides are bytes, including negative/zero strides.
// Output is contiguous and must not overlap inputs. Optional component-byte
// swapping follows NumPy stack dtype normalization without a temporary image.
// An optional item_bytes-long byte_order maps output bytes to source bytes,
// supporting mixed-endian structured records with unchanged field layout.
void gray_expand_rows_into(const void* image,const void* alpha,void* output,
    std::size_t rows,std::size_t columns,std::size_t item_bytes,int channels,
    const std::ptrdiff_t* image_offsets,const std::ptrdiff_t* alpha_offsets,
    std::ptrdiff_t image_step,std::ptrdiff_t alpha_step,std::size_t swap_component=0,const std::size_t* byte_order=nullptr);
// Fused normalized RGBA composition; float32 input yields float32, otherwise
// float64. Background contains three float64 metadata values.
void rgba2rgb_rows_into(const void* image,void* output,std::size_t rows,std::size_t columns,
    PixelType input_type,const double* background,const std::ptrdiff_t* row_offsets,
    std::ptrdiff_t pixel_step,std::ptrdiff_t channel_step);
}
