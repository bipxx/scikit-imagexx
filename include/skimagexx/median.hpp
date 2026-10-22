#pragma once
#include "gaussian_nd.hpp"
#include <cstdint>
#include <vector>
namespace skimagexx {
struct MedianInfo {bool unsupported_nan=false;};
// Contiguous native/aligned buffers; source and target must not overlap.
// Coordinates are row-major nonzero footprint positions. The rank is explicit
// because SciPy's fast 1-D path uses the full window even for sparse footprints.
// nd_rank reproduces source-typed rounding around double selection; fast_1d
// uses exact int64 working storage when the source is int64.
MedianInfo median_filter_into(const void*,void*,PixelType,PixelType,
    const std::vector<std::size_t>& shape,const std::vector<std::size_t>& window,
    const std::vector<std::vector<std::size_t>>& coordinates,std::size_t rank,
    GaussianBorder,double cval,std::int64_t integer_cval=0,bool nd_rank=true,
    bool fast_1d=false,bool zero_strides=false,bool source_cval=true);
}
