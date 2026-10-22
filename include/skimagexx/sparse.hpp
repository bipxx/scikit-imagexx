#pragma once
#include "gaussian_nd.hpp"
#include <vector>
namespace skimagexx {
// Contiguous, aligned float32/64 buffers. Coordinates are in row-major order,
// start at the kernel corner, and refer to weights stored in the image type.
// Output does not overlap input. Each product rounds before the next addition.
// Returned pairs of flags describe multiply/add for every tap: overflow=1,
// underflow=2, invalid=4. Padding never performs floating arithmetic.
std::vector<unsigned> correlate_sparse_into(const void* input,void* output,PixelType,
    const std::vector<std::size_t>& shape,const std::vector<std::size_t>& kernel_shape,
    const std::vector<std::vector<std::size_t>>& coordinates,const void* weights,
    GaussianBorder border=GaussianBorder::reflect,bool valid=false);
}
