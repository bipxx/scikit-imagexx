#pragma once
#include "skimagexx/gaussian_nd.hpp"
#include <array>
#include <vector>

namespace skimagexx::detail {
// Private resampling foundation. No Python resize/rescale native coverage is
// claimed until full pixel conversion, Gaussian, sampling and clipping pass.
std::array<double,6> spline_weights(double coordinate,int order);
// Contiguous float64 buffers, disjoint or identical. Axes run in input order;
// singleton axes are copies. Order 0/1 copies without prefiltering. Empty
// shapes are permitted. Global optional worker policy applies to long arrays.
// legacy_reflect reproduces the endpoint aliasing fixed in SciPy 1.18.0
// (scipy/scipy#24615). allow_simd=false permits exact independent validation.
void spline_filter_into(const double* input,double* output,
                        const std::vector<std::size_t>& shape,int order,GaussianBorder,
                        bool legacy_reflect=false,bool allow_simd=true);
}
#ifdef SKIMAGEXX_AVX2
namespace skimagexx::simd {
void spline_filter_four(double*,std::size_t length,std::size_t stride,
                        const double* poles,int npoles,GaussianBorder,bool legacy);
}
#endif
