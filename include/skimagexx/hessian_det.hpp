#pragma once
#include "convert.hpp"
#include <cstddef>
#include <cstdint>
namespace skimagexx {
// Dense/aligned C buffers, no overlap. Input is normalized as img_as_float;
// approximate output is float64, including float32 input. Empty axes allowed.
// size is trunc(3*sigma), in [-3000,3000]. Returns false for unsafe arithmetic.
bool hessian_det_approx_into(const void*,PixelType,double*,std::size_t,std::size_t,std::int64_t);
// Symmetric fields in upper-triangle row order; output matches field precision.
// Per-pixel partial-pivot LU and signed log/exp determinant, 2..5 dimensions.
bool hessian_det_from_fields_into(const void* const*,void*,std::size_t,std::size_t,bool);
}
