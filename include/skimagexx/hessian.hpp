#pragma once
#include "gaussian_nd.hpp"
namespace skimagexx {
struct HessianPass { GaussianPass filter; bool derivative=false; };
// Independent contiguous/aligned buffers. Output list follows upper-triangle
// rc (or reversed 2-D xy) axis order. False requests whole-call fallback.
bool hessian_matrix_into(const void*,PixelType,void* const*,const std::vector<std::size_t>&,
 const std::vector<std::vector<HessianPass>>&,bool gaussian_derivatives,bool xy=false);
}
