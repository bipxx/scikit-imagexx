#pragma once
#include <cstddef>
namespace skimagexx::simd {
std::size_t meijering_response(const void*,void*,std::size_t,std::size_t,std::size_t,std::size_t,bool,double,double&,bool&);
std::size_t meijering_normalize(void*,const void*,std::size_t,bool,double);
}
