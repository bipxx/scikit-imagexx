#pragma once
#include "skimagexx/compare_images.hpp"

namespace skimagexx::simd {
bool compare_finite(const void*,std::size_t,PixelType);
std::size_t compare_span(const void*,const void*,void*,std::size_t,
    PixelType,PixelType,PixelType,detail::CompareMethod,bool& all_finite);
}
