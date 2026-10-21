#include "skimagexx/convert.hpp"
#include "skimagexx/image.hpp"
#include "execution.hpp"
#include "simd.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace skimagexx {
namespace {
template<class T> constexpr bool integer = std::is_integral_v<T> && !std::is_same_v<T,bool>;
template<class T> constexpr int bits = int(sizeof(T)*8);
template<class T> constexpr int value_bits = bits<T>-(std::is_signed_v<T> ? 1 : 0);
constexpr std::uint64_t mask(int n) { return n == 64 ? UINT64_MAX : (std::uint64_t{1} << n)-1; }
template<int n, int m> std::uint64_t expand(std::uint64_t v) {
    if constexpr (n == m) return v;
    else if constexpr (n > m) return v >> (n-m);
    else if constexpr (m % n == 0) return v*(mask(m)/mask(n));
    else {
        constexpr int extended = (m/n+1)*n;
        return (v*(mask(extended)/mask(n))) >> (extended-m);
    }
}
double even_round(double value) {
    const double low = std::floor(value), fraction = value-low;
    return low+(fraction > 0.5 || (fraction == 0.5 && std::fmod(low,2.0) != 0));
}
std::int64_t floor_shift(std::int64_t value, int shift) {
    const auto divisor = std::int64_t{1} << shift;
    return value/divisor-(value < 0 && value%divisor != 0);
}
template<class Input, class Output> Output pixel(Input value, bool no_scale) {
    if constexpr (std::is_same_v<Output,bool>) {
        if constexpr (std::is_floating_point_v<Input>) return value > Input(0.5);
        else return value > Input(double(std::numeric_limits<Input>::max())/2.0);
    } else if constexpr (std::is_same_v<Input,bool>) {
        return value ? (std::is_floating_point_v<Output> ? Output(1) : std::numeric_limits<Output>::max()) : Output(0);
    } else if constexpr (std::is_floating_point_v<Output>) {
        if constexpr (std::is_floating_point_v<Input>) return Output(value);
        else {
            using Arithmetic = std::conditional_t<sizeof(Input) <= sizeof(Output),Output,double>;
            const Arithmetic scaled = Arithmetic(value)*Arithmetic(1.0/double(std::numeric_limits<Input>::max()));
            return Output(std::is_signed_v<Input> ? std::max(scaled,Arithmetic(-1)) : scaled);
        }
    } else if constexpr (std::is_floating_point_v<Input>) {
        if (std::isnan(value)) return Output(0);
        using Arithmetic = Input;
        const Arithmetic scale = std::is_unsigned_v<Output> ? Arithmetic(std::numeric_limits<Output>::max()) :
            Arithmetic((double(std::numeric_limits<Output>::max())-double(std::numeric_limits<Output>::min()))/2.0);
        Arithmetic scaled = value*scale;
        if constexpr (std::is_signed_v<Output>) scaled -= Arithmetic(0.5);
        const double rounded = even_round(double(scaled));
        return Output(std::clamp(rounded,double(std::numeric_limits<Output>::min()),double(std::numeric_limits<Output>::max())));
    } else if constexpr (std::is_unsigned_v<Input>) {
        constexpr int n = bits<Input>, m = value_bits<Output>;
        if (no_scale) return Output(value);
        return Output(expand<n,m>(std::uint64_t(value)));
    } else if constexpr (std::is_unsigned_v<Output>) {
        constexpr int n = bits<Input>-1, m = bits<Output>;
        if (no_scale) {
            // Upstream casts to a signed intermediate *before* clipping. This
            // even preserves wraparound of large negative values in that branch.
            const auto intermediate = m < 16 ? std::int64_t(static_cast<std::int16_t>(value)) :
                                               std::int64_t(static_cast<std::int32_t>(value));
            return Output(std::max(intermediate,std::int64_t{0}));
        }
        if (value < 0) return Output(0);
        return Output(expand<n,m>(std::uint64_t(value)));
    } else {
        if constexpr (sizeof(Input) > sizeof(Output)) {
            if (no_scale) return Output(value);
            return Output(floor_shift(std::int64_t(value),bits<Input>-bits<Output>));
        } else if constexpr (sizeof(Input) == sizeof(Output)) return Output(value);
        else {
            // Signed expansion shifts the *entire* asymmetric integer range.
            const auto positive = std::uint64_t(std::int64_t(value)-std::int64_t(std::numeric_limits<Input>::min()));
            return Output(std::int64_t(expand<bits<Input>,bits<Output>>(positive))+
                          std::int64_t(std::numeric_limits<Output>::min()));
        }
    }
}

template<class T> constexpr PixelType type_id() {
    if constexpr (std::is_same_v<T,bool>) return PixelType::boolean;
    else if constexpr (std::is_same_v<T,float>) return PixelType::float32;
    else if constexpr (std::is_same_v<T,double>) return PixelType::float64;
    else if constexpr (std::is_same_v<T,std::uint8_t>) return PixelType::uint8;
    else if constexpr (std::is_same_v<T,std::int8_t>) return PixelType::int8;
    else if constexpr (std::is_same_v<T,std::uint16_t>) return PixelType::uint16;
    else if constexpr (std::is_same_v<T,std::int16_t>) return PixelType::int16;
    else if constexpr (std::is_same_v<T,std::uint32_t>) return PixelType::uint32;
    else if constexpr (std::is_same_v<T,std::int32_t>) return PixelType::int32;
    else if constexpr (std::is_same_v<T,std::uint64_t>) return PixelType::uint64;
    else return PixelType::int64;
}
template<class Input, class Output> ConversionInfo convert(const Input* p, Output* out, std::size_t n) {
    ConversionInfo info;
    if (!n) {
        if constexpr (std::is_floating_point_v<Input> && integer<Output>)
            throw std::invalid_argument("zero-size array to reduction operation minimum which has no identity");
        else if constexpr (integer<Input> && integer<Output> && value_bits<Input> > value_bits<Output>)
            throw std::invalid_argument("zero-size array to reduction operation maximum which has no identity");
        return info;
    }
    const int threads = detail::active_workers(n);
    // Global extrema are needed only by range validation and data-dependent
    // downcasting; ordinary casts and integer-to-float paths remain one-pass.
    if constexpr ((std::is_floating_point_v<Input> && integer<Output>) ||
                  (integer<Input> && integer<Output> && value_bits<Input> > value_bits<Output>)) {
        struct Bounds { Input low = std::numeric_limits<Input>::max(), high = std::numeric_limits<Input>::lowest(); bool nan = false; };
        std::vector<Bounds> partials(static_cast<std::size_t>(threads));
        detail::partition(n,threads,[&](auto begin, auto end, auto id) {
            auto& b = partials[id];
#ifdef SKIMAGEXX_AVX2
            if constexpr (integer<Input> && sizeof(Input) >= 4) {
                if (has_avx2()) { b.high = simd::maximum(p+begin,end-begin); return; }
            }
            if constexpr (std::is_floating_point_v<Input> || std::is_same_v<Input,std::uint16_t> || std::is_same_v<Input,std::int16_t>) {
                if (has_avx2()) {
                    double low, high;
                    simd::bounds(p+begin,end-begin,low,high);
                    b.low = Input(low); b.high = Input(high); b.nan = std::isnan(low);
                    return;
                }
            }
#endif
            for (auto i = begin; i < end; ++i) {
                b.low = std::min(b.low,p[i]); b.high = std::max(b.high,p[i]);
                if constexpr (std::is_floating_point_v<Input>) b.nan = b.nan || std::isnan(p[i]);
            }
        });
        Bounds b;
        for (const auto& item : partials) {
            b.low = std::min(b.low,item.low); b.high = std::max(b.high,item.high); b.nan = b.nan || item.nan;
        }
        if constexpr (std::is_floating_point_v<Input>) {
            if (!b.nan && (b.low < -1 || b.high > 1)) throw std::invalid_argument("Images of type float must be between -1 and 1.");
            info.nan_to_integer = b.nan;
        } else {
            constexpr int m = value_bits<Output>;
            // Signed -> unsigned _scale uses the full output bit count.
            if constexpr (std::is_signed_v<Input>) info.downcast_without_scaling = b.high < (std::int64_t{1} << m);
            else info.downcast_without_scaling = b.high < (std::uint64_t{1} << m);
            if constexpr (std::is_signed_v<Input>) info.maximum_signed = std::int64_t(b.high);
            else info.maximum_unsigned = std::uint64_t(b.high);
        }
    }
    constexpr bool float_cast = std::is_floating_point_v<Input> && std::is_floating_point_v<Output> && !std::is_same_v<Input,Output>;
    const bool track = float_cast || info.nan_to_integer;
    const auto slots = track ? static_cast<std::size_t>(threads) : 0;
    std::vector<unsigned char> overflows(slots), underflows(slots), invalids(slots);
    const bool vectorized = has_avx2();
    detail::partition(n,threads,[&](auto begin, auto end, auto id) {
        std::size_t consumed = 0;
        ConversionInfo status;
        auto& overflow = status.floating_overflow;
        auto& underflow = status.floating_underflow;
#ifdef SKIMAGEXX_AVX2
        if (vectorized && !info.nan_to_integer)
            consumed = simd::convert(p+begin,out+begin,end-begin,type_id<Input>(),type_id<Output>(),
                                     info.downcast_without_scaling,status);
#else
        (void)vectorized;
#endif
        for (auto i = begin+consumed; i < end; ++i) {
            if constexpr (std::is_floating_point_v<Input> && std::is_floating_point_v<Output> && !std::is_same_v<Input,Output>) {
                using Bits = std::conditional_t<sizeof(Input) == 4,std::uint32_t,std::uint64_t>;
                Bits raw;
                std::memcpy(&raw,p+i,sizeof(raw));
                constexpr std::uint64_t exponent = sizeof(Input) == 4 ? 0x7f800000ULL : 0x7ff0000000000000ULL;
                constexpr std::uint64_t mantissa = sizeof(Input) == 4 ? 0x007fffffULL : 0x000fffffffffffffULL;
                constexpr std::uint64_t quiet = sizeof(Input) == 4 ? 0x00400000ULL : 0x0008000000000000ULL;
                status.invalid_float_cast = status.invalid_float_cast || ((raw & exponent) == exponent && (raw & mantissa) != 0 && (raw & quiet) == 0);
            }
            if constexpr (std::is_same_v<Input,bool>)
                out[i] = pixel<Input,Output>(reinterpret_cast<const unsigned char*>(p)[i] != 0,info.downcast_without_scaling);
            else out[i] = pixel<Input,Output>(p[i],info.downcast_without_scaling);
            if constexpr (std::is_same_v<Input,double> && std::is_same_v<Output,float>) {
                overflow = overflow || (std::isfinite(p[i]) && std::isinf(out[i]));
                underflow = underflow || (std::isfinite(p[i]) && std::abs(double(out[i])) < std::numeric_limits<float>::min() && double(out[i]) != p[i]);
            }
            if constexpr (std::is_floating_point_v<Input> && integer<Output>) {
                if (info.nan_to_integer && std::isfinite(p[i])) {
                    const Input scale = std::is_unsigned_v<Output> ? Input(std::numeric_limits<Output>::max()) : Input(32767.5);
                    overflow = overflow || std::isinf(Input(p[i]*scale));
                }
            }
        }
        if (track) {
            overflows[id] = static_cast<unsigned char>(overflow);
            underflows[id] = static_cast<unsigned char>(underflow);
            invalids[id] = static_cast<unsigned char>(status.invalid_float_cast);
        }
    });
    info.floating_overflow = std::any_of(overflows.begin(),overflows.end(),[](auto flag) { return flag != 0; });
    info.floating_underflow = std::any_of(underflows.begin(),underflows.end(),[](auto flag) { return flag != 0; });
    info.invalid_float_cast = std::any_of(invalids.begin(),invalids.end(),[](auto flag) { return flag != 0; });
    return info;
}
template<class Input> ConversionInfo dispatch_output(const void* p, void* out, std::size_t n, PixelType target) {
    const auto* source = static_cast<const Input*>(p);
    switch (target) {
        case PixelType::boolean: return convert(source,static_cast<bool*>(out),n);
        case PixelType::uint8: return convert(source,static_cast<std::uint8_t*>(out),n);
        case PixelType::uint16: return convert(source,static_cast<std::uint16_t*>(out),n);
        case PixelType::int16: return convert(source,static_cast<std::int16_t*>(out),n);
        case PixelType::float32: return convert(source,static_cast<float*>(out),n);
        case PixelType::float64: return convert(source,static_cast<double*>(out),n);
        default: throw std::invalid_argument("unsupported output dtype");
    }
}
}

std::size_t pixel_size(PixelType type) {
    switch (type) {
        case PixelType::boolean: case PixelType::uint8: case PixelType::int8: return 1;
        case PixelType::uint16: case PixelType::int16: return 2;
        case PixelType::uint32: case PixelType::int32: case PixelType::float32: return 4;
        case PixelType::uint64: case PixelType::int64: case PixelType::float64: return 8;
        default: throw std::invalid_argument("unsupported pixel type");
    }
}
ConversionInfo convert_into(const void* p, void* out, std::size_t n, PixelType source, PixelType target) {
    if (n && (!p || !out)) throw std::invalid_argument("input and output must be nonnull");
    const auto limit = static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max());
    if (n > limit/std::max(pixel_size(source),pixel_size(target))) throw std::overflow_error("array size overflows");
    switch (source) {
        case PixelType::boolean: return dispatch_output<bool>(p,out,n,target);
        case PixelType::uint8: return dispatch_output<std::uint8_t>(p,out,n,target);
        case PixelType::int8: return dispatch_output<std::int8_t>(p,out,n,target);
        case PixelType::uint16: return dispatch_output<std::uint16_t>(p,out,n,target);
        case PixelType::int16: return dispatch_output<std::int16_t>(p,out,n,target);
        case PixelType::uint32: return dispatch_output<std::uint32_t>(p,out,n,target);
        case PixelType::int32: return dispatch_output<std::int32_t>(p,out,n,target);
        case PixelType::uint64: return dispatch_output<std::uint64_t>(p,out,n,target);
        case PixelType::int64: return dispatch_output<std::int64_t>(p,out,n,target);
        case PixelType::float32: return dispatch_output<float>(p,out,n,target);
        case PixelType::float64: return dispatch_output<double>(p,out,n,target);
        default: throw std::invalid_argument("unsupported input dtype");
    }
}
} // namespace skimagexx
