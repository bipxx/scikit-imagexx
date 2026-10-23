#pragma once
#include <cstddef>
namespace skimagexx::simd {
std::size_t structure_validate(const void*,std::size_t,bool,double,bool&);
std::size_t structure_pass(const void*,const void*,const void*,void*,std::size_t,bool,bool,double);
std::size_t structure_product(const void*,const void*,void*,std::size_t,bool);
}
