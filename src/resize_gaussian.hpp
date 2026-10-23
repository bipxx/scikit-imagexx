#pragma once
#include "skimagexx/gaussian_nd.hpp"
namespace skimagexx::detail {
// Raw-value ndimage Gaussian stage. Each axis stores back into the input
// dtype, including integer truncation. Source/output are aligned, native,
// contiguous and disjoint. No integer normalization is performed.
void gaussian_raw_into(const void*,void*,PixelType,
                       const std::vector<std::size_t>&,const std::vector<GaussianPass>&);
}
#ifdef SKIMAGEXX_AVX2
namespace skimagexx::simd {
std::size_t gaussian_raw_rows(const void* const*,const void* const*,const void*,void*,
                              std::size_t,std::size_t,const double*,double,double,PixelType);
}
#endif
