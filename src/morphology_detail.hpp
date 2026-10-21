#pragma once
#include "skimagexx/morphology.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <type_traits>

namespace skimagexx::detail {
// SciPy's generic grayscale buffers are double, including 64-bit integers.
// Make the observed x86 cast boundaries explicit rather than invoking C++ UB.
template<class T> T morphology_cast(double v) {
    if constexpr (std::is_floating_point_v<T>) return T(v);
    else if constexpr (std::is_same_v<T,std::int64_t>)
        return v >= 9223372036854775808.0 || v < -9223372036854775808.0 || !std::isfinite(v) ? std::numeric_limits<T>::min() : T(v);
    else if constexpr (std::is_same_v<T,std::uint64_t>) {
        if (!std::isfinite(v) || v >= 18446744073709551616.0) return 0;
        if (v < 0) {
            // SciPy's x86 min/max output casts convert through a signed register.
            return v >= -9223372036854775808.0 ? T(std::int64_t(v)) : T(1)<<63;
        }
        return T(v);
    }
    else if constexpr (std::is_same_v<T,std::int32_t>)
        return v > 2147483647.0 || v < -2147483648.0 || !std::isfinite(v) ? std::numeric_limits<T>::min() : T(v);
    else if constexpr (std::is_same_v<T,std::uint32_t>) {
        if (!std::isfinite(v) || v >= 18446744073709551616.0 || v < -9223372036854775808.0) return 0;
        double r = std::fmod(std::trunc(v),4294967296.0);
        if (r < 0) r += 4294967296.0;
        return T(r);
    } else {
        const auto integer = morphology_cast<std::int32_t>(v);
        using U = std::make_unsigned_t<T>;
        const auto bits = U(integer);
        T result; std::memcpy(&result,&bits,sizeof(T)); return result;
    }
}
template<class T> T morphology_round(T v) {
    if constexpr (std::is_integral_v<T> && sizeof(T) == 8) return morphology_cast<T>(double(v));
    else return v;
}
template<class T> T extrema(T a, T b, bool erode) { return erode ? (b < a ? b : a) : (b > a ? b : a); }
} // namespace skimagexx::detail
