#pragma once
#include "skimagexx/gabor.hpp"
namespace skimagexx::simd {
std::size_t gabor_kernel_accurate_row(void*,bool,std::size_t,std::size_t,double,const GaborParameters&);
}
