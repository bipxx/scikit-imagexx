#pragma once
#include "gaussian_nd.hpp"
#include <cstddef>

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
// Prepared C float64 center lines; fresh output[2,length,width]. False restores
// FP state and requests the unchanged perpendicular NumPy linspace/stack route.
bool profile_coordinates_into(const double* rows,const double* cols,std::size_t length,
    std::size_t width,double row_width,double col_width,double* output,int workers);
}
