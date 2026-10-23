#pragma once
#include <cstddef>
namespace skimagexx::simd {
std::size_t sato_negate(void*,std::size_t,bool);
std::size_t sato_response(const void*,const void*,const void*,void*,std::size_t,
                         bool,bool,bool,bool,double,bool&);
}
