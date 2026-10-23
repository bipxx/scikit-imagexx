#pragma once
#include "skimagexx/corner_response.hpp"
#include <cmath>
#include <limits>
#include <type_traits>
namespace skimagexx::detail {
template<class T> double corner_bound(){return std::sqrt(double(std::numeric_limits<T>::max()))/16;}
template<class T,class O> bool corner_pixel(T a,T b,T c,O& first,O& second,CornerResponse method,double parameter){
    const auto bound=corner_bound<T>();
    if(!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(c) || std::abs(double(a))>bound || std::abs(double(b))>bound || std::abs(double(c))>bound)return false;
    const T trace=T(a+c);
    if(method==CornerResponse::shi_tomasi){
        const T difference=T(a-c),square=T(b*b);
        const T root=std::sqrt(T(T(difference*difference)+T(T(4)*square)));
        first=O(T(T(trace-root)/T(2)));return std::isfinite(first);
    }
    const T determinant=T(T(a*c)-T(b*b));
    if(method==CornerResponse::harris){
        const T square=T(trace*trace);
        first=O(O(determinant)-O(O(parameter)*O(square)));
    }else if(method==CornerResponse::noble){
        const T numerator=T(T(2)*determinant);
        const O denominator=O(O(trace)+O(parameter));
        if(!std::isfinite(denominator) || denominator==O(0))return false;
        first=O(O(numerator)/denominator);
    }else{
        if(trace==T(0)){first=second=O(0);return true;}
        const T denominator=T(trace*trace);if(denominator==T(0))return false;
        first=O(T(determinant/trace));second=O(T(T(T(4)*determinant)/denominator));
        if(!std::isfinite(second))return false;
    }
    return std::isfinite(first);
}
}
namespace skimagexx::simd {
std::size_t corner_response(const void* const*,void*,void*,std::size_t,std::size_t,
    bool,bool,CornerResponse,double,bool&);
}
