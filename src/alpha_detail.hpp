#pragma once
#include "color_detail.hpp"
#include <cstring>
namespace skimagexx::detail {
template<class T> T alpha_load(const unsigned char* p) {T value;std::memcpy(&value,p,sizeof(T));return value;}
template<class Input,bool Boolean=false> ColorOutput<Input> alpha_normalized(const unsigned char* p) {
    if constexpr(Boolean)return alpha_load<std::uint8_t>(p)!=0 ? 1.0 : 0.0;
    else return color_value(alpha_load<Input>(p),0,false);
}
template<class T> T alpha_clip(T value) {return value<T(0) ? T(0) : value>T(1) ? T(1) : value;}
}
