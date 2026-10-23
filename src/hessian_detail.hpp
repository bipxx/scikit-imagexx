#pragma once
#include <cstddef>
namespace skimagexx::simd {
std::size_t hessian_gradient(const void*,const void*,void*,std::size_t,bool,bool);
std::size_t hessian_convolve(const void* const*,const void* const*,const void*,void*,std::size_t,
 std::size_t,const double*,bool,double,bool);
}
