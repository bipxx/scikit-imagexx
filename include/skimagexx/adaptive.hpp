#pragma once
#include "gaussian_nd.hpp"
namespace skimagexx {
enum class AdaptiveMethod {mean,median};
// Contiguous finite float32/64 input. Output has the same precision, with
// rounding after each mean axis. Buffers must not overlap. Window entries are
// positive; mean windows may be even. Borders follow GaussianBorder semantics.
// zero_strides records an originally fully broadcast input and reproduces
// SciPy's generic-rank offset sentinel behavior; input then has one value.
void adaptive_filter_into(const void*,void*,PixelType,const std::vector<std::size_t>&,
    const std::vector<std::size_t>&,const std::vector<GaussianBorder>&,AdaptiveMethod,double cval=0,bool zero_strides=false);
// Finite scalar subtraction in target precision. Identical buffers are
// permitted when types match; otherwise buffers must not overlap.
void adaptive_offset_into(const void*,void*,std::size_t,PixelType,PixelType,double);
}
