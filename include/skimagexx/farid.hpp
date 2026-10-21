#pragma once
#include "convert.hpp"
#include <cstddef>
#include <cstdint>
#include <vector>
namespace skimagexx {
enum class FaridBorder { reflect,constant,nearest,mirror,wrap };
// Caller-owned contiguous buffers, no overlap. Float32/half output is float32;
// other numeric input produces float64. Axes retain order and repetitions.
// An empty axis list produces zero. Negative axes retain upstream weighting.
// Directional mode writes one convolution directly (the farid_h/v contract).
void farid_into(const void*,void*,PixelType,const std::vector<std::size_t>&,
    const std::vector<std::ptrdiff_t>&,FaridBorder=FaridBorder::reflect,double=0,bool=false);
void farid_half_into(const std::uint16_t*,float*,const std::vector<std::size_t>&,
    const std::vector<std::ptrdiff_t>&,FaridBorder=FaridBorder::reflect,double=0,bool=false);
}
