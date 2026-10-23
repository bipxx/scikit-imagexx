#pragma once
// Defined counterparts of ndimage's raw line-buffer and interpolation casts.
// SciPy reference: ni_support.c / ni_interpolation.c, BSD notice retained in
// third_party/scipy-ndimage-LICENSE.txt. No out-of-range C++ FP-to-int cast.
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

namespace skimagexx::detail {
inline std::int32_t resize_truncate_i32(double value) {
    if(!(value>=-0x1p31 && value<0x1p31))return std::numeric_limits<std::int32_t>::min();
    return static_cast<std::int32_t>(value);
}
inline std::int64_t resize_truncate_i64(double value) {
    if(!(value>=-0x1p63 && value<0x1p63))return std::numeric_limits<std::int64_t>::min();
    return static_cast<std::int64_t>(value);
}
template<class T> T resize_raw_cast(double value) {
    if constexpr(std::is_floating_point_v<T>)return static_cast<T>(value);
    else if constexpr(sizeof(T)<4) {
        using U=std::make_unsigned_t<T>;
        const auto bits=static_cast<U>(resize_truncate_i32(value));
        T result;std::memcpy(&result,&bits,sizeof(T));return result;
    } else if constexpr(std::is_same_v<T,std::int32_t>)return resize_truncate_i32(value);
    else if constexpr(std::is_same_v<T,std::uint32_t>)return static_cast<T>(resize_truncate_i64(value));
    else if constexpr(std::is_same_v<T,std::int64_t>)return resize_truncate_i64(value);
    else {
        static_assert(std::is_same_v<T,std::uint64_t>);
        const bool upper=value>=0x1p63;
        const auto bits=static_cast<T>(resize_truncate_i64(upper ? value-0x1p63 : value));
        return upper ? bits^(std::uint64_t{1}<<63) : bits;
    }
}
template<class T,bool Boolean=false> T resize_sample_cast(double value) {
    if constexpr(std::is_floating_point_v<T>)return static_cast<T>(value);
    else if constexpr(Boolean)return resize_raw_cast<T>(value);
    else {
        if constexpr(std::is_unsigned_v<T>)value=value>0 ? value+.5 : 0.;
        else value=value>0 ? value+.5 : value-.5;
        // The sampler's integer source and destination have the same dtype.
        // Bounds are compared in double, including rounded 64-bit maxima.
        if(value>double(std::numeric_limits<T>::max()))value=double(std::numeric_limits<T>::max());
        if(value<double(std::numeric_limits<T>::min()))value=double(std::numeric_limits<T>::min());
        return resize_raw_cast<T>(value);
    }
}
} // namespace skimagexx::detail
