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
// Native-endian aligned contiguous storage; source/output have the same type.
// Float32/64 support orders 0..5; integer/bool storage supports nearest only.
// A public resize wrapper converts integers to floating before higher orders.
void zoom_typed_into(const void* input,void* output,PixelType,
                     const std::vector<std::size_t>& input_shape,
                     const std::vector<std::size_t>& output_shape,const ZoomOptions&);
}
#ifdef SKIMAGEXX_AVX2
namespace skimagexx::simd {
std::size_t zoom_sample(const double*,double*,const detail::ZoomPlan&,
                        std::size_t begin,std::size_t end);
std::size_t zoom_sample_typed(const void*,void*,const detail::ZoomPlan&,
                             std::size_t begin,std::size_t end,PixelType source,PixelType target);
// Profile plans contain two axes of four paired query points, not a Cartesian grid.
bool profile_all_finite(const void*,std::size_t count,bool single);
std::size_t profile_sample_typed(const void*,void*,const detail::ZoomPlan&,
                                std::size_t lanes,PixelType);
}
#endif

namespace skimagexx::detail {
struct ProfileInput { const void* data;PixelType type;std::size_t height,width; };
// Private prepared plane: native aligned C bool/u8/f32/f64; coordinates are
// C double[2,count] (row plane, column plane). Output is disjoint same-type C.
// Order0/1 only. constant/wrap mean source grid-constant/grid-wrap.
// False discards the private output and requests the original primitive for
// nonfinite data/cval, unsupported FP state or unsafe coordinate conversion.
// No coordinate generation, channel stack, flip or reducer is performed here.
bool profile_sample(const ProfileInput&,const double* coordinates,std::size_t count,
                    void* output,int order,GaussianBorder,double cval,int workers);
}
