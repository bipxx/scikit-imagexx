#pragma once
#include "skimagexx/distance.hpp"
#include <cmath>
#include <stdexcept>
#include <type_traits>

namespace skimagexx::detail {
template<class T,class Input> T distance_read(const void* p,std::ptrdiff_t index) {
    if constexpr(std::is_same_v<Input,bool>) return T(static_cast<const unsigned char*>(p)[index]!=0);
    else return T(static_cast<const Input*>(p)[index]);
}
template<class T> using DistanceReader=T(*)(const void*,std::ptrdiff_t);
template<class T> DistanceReader<T> distance_reader(PixelType type) {
    switch(type) {
        case PixelType::boolean:return distance_read<T,bool>;
        case PixelType::uint8:return distance_read<T,std::uint8_t>;
        case PixelType::int8:return distance_read<T,std::int8_t>;
        case PixelType::uint16:return distance_read<T,std::uint16_t>;
        case PixelType::int16:return distance_read<T,std::int16_t>;
        case PixelType::uint32:return distance_read<T,std::uint32_t>;
        case PixelType::int32:return distance_read<T,std::int32_t>;
        case PixelType::uint64:return distance_read<T,std::uint64_t>;
        case PixelType::int64:return distance_read<T,std::int64_t>;
        case PixelType::float32:return distance_read<T,float>;
        case PixelType::float64:return distance_read<T,double>;
        default:throw std::invalid_argument("unsupported distance input dtype");
    }
}
template<class T> T distance_pixel(const void* a,const void* b,DistanceReader<T> ar,DistanceReader<T> br,
    std::ptrdiff_t ac,std::ptrdiff_t bc) {
    const T x=T(br(b,0)-ar(a,0)),y=T(br(b,bc)-ar(a,ac)),z=T(br(b,2*bc)-ar(a,2*ac));
    return std::sqrt(T(T(T(x*x)+T(y*y))+T(z*z)));
}
}
