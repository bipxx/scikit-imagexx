#pragma once
#include "skimagexx/frangi.hpp"
#include <cstddef>
namespace skimagexx::simd {
std::size_t frangi_prepare(const void*,void*,void*,void*,std::size_t,std::size_t,std::size_t,std::size_t,bool,double&,bool&);
std::size_t frangi_response(const void*,const void*,const void*,const void*,void*,std::size_t,bool,bool,const FrangiParameters&,bool&);
std::size_t ridge_hessian_finish(void*,std::size_t,bool);
std::size_t frangi_exp_arguments(double*,std::size_t,double,bool&);
std::size_t frangi_combine(const double*,const double*,const double*,const double*,double*,std::size_t,bool&);
}
