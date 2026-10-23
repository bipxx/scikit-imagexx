#pragma once
#include "gaussian_nd.hpp"

namespace skimagexx {
// Contiguous aligned real image, 2..5 axes. Integer/bool images are normalized;
// float32 produces float32 fields, other inputs produce float64. Each output
// buffer has product(shape) elements and must not overlap input or other output
// buffers. Upper-triangle fields use row/column order, reversed for xy=true.
// Gaussian smoothing passes must contain normalized nonnegative coefficients.
// False means nonfinite/unsafe input: discard output and use compatibility.
// Optional parallelism follows the process-wide worker limit (default one).
bool structure_tensor_into(const void* image,PixelType source,void* const* output,
    const std::vector<std::size_t>& shape,const std::vector<GaussianPass>& smoothing,
    const std::vector<GaussianBorder>& borders,double cval=0,bool xy=false);
}
