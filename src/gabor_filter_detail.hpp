#pragma once
#include <cstddef>

namespace skimagexx::detail {
struct GaborFilterTap {const double* row;double real,imaginary;};
}
namespace skimagexx::simd {
std::size_t gabor_filter_row(const detail::GaborFilterTap*,std::size_t,
    double*,double*,std::size_t,double);
std::size_t gabor_filter_store_float(const double*,float*,std::size_t);
std::size_t gabor_filter_widen_float(const float*,double*,std::size_t);
}
