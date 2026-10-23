#pragma once
#include "spline_detail.hpp"
#include <cstdint>

namespace skimagexx::detail {
struct ZoomPoint {
    std::array<std::ptrdiff_t,6> offsets{};
    std::array<double,6> weights{};
};
struct ZoomPlan {
    std::vector<std::vector<ZoomPoint>> axes;
    std::vector<std::size_t> output_strides;
    std::vector<std::uint8_t> footprint;
    std::size_t count=0,corners=1;
    int order=1;
    double cval=0;
};
struct ZoomOptions {
    int order=1;
    GaussianBorder border=GaussianBorder::mirror;
    double cval=0;
    bool legacy_reflect=false;
    bool identity_copy=false;
    bool allow_simd=true;
};
// Private float64 grid-mode zoom foundation. It includes spline coefficient
// preparation and nearest/grid-constant prepadding. Input/output are disjoint
// contiguous buffers (identical buffers allowed only for identity_copy).
// Public resize/rescale conversion, AA and clipping are separate stages.
void zoom_into(const double* input,double* output,
               const std::vector<std::size_t>& input_shape,
               const std::vector<std::size_t>& output_shape,const ZoomOptions&);
}
#ifdef SKIMAGEXX_AVX2
namespace skimagexx::simd {
std::size_t zoom_sample(const double*,double*,const detail::ZoomPlan&,
                        std::size_t begin,std::size_t end);
}
#endif
