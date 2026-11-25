#pragma once
#include <cstddef>
namespace skimagexx::simd {
struct HessianTap {
 std::ptrdiff_t left, right;
};
struct HessianKernel {
 const double* weights;
 const double* derivative_weights;
 std::size_t radius;
 std::size_t smooth_start, derivative_start;
 bool derivative;
};
std::size_t hessian_gradient(const void*,const void*,void*,std::size_t,bool,bool);
std::size_t hessian_convolve_direct(const void*,void*,void*,std::size_t,std::size_t,
 const HessianKernel&,bool,bool);
std::size_t hessian_convolve_indexed(const void*,const void*,void*,void*,std::size_t,
 const HessianTap*,const HessianKernel&,double,bool,bool);
}
