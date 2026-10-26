#pragma once
#include "skimagexx/color.hpp"
#include <array>
#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>

namespace skimagexx::detail {
// Native aligned C triples, disjoint output; global worker budget is reused.
// Returns standard divide/over/under/invalid bits and restores every worker FEnv.
unsigned color_transform_status_into(const float*,float*,std::size_t,ColorTransform);
unsigned color_transform_status_into(const double*,double*,std::size_t,ColorTransform);
struct ColorSpec {
    std::array<double,9> matrix;
    bool input_offset = false, output_offset = false;
    int transfer = 0; // 1: sRGB decode before matrix; 2: encode/clip after matrix.
};
const ColorSpec& color_spec(ColorTransform transform);
const std::array<double,256>& srgb_decode_table();
template<class T> using ColorOutput = std::conditional_t<std::is_same_v<T,float>,float,double>;
template<class T> ColorOutput<T> color_value(T value, int channel, bool offset) {
    if (offset) {
        const int amount = channel == 0 ? 16 : 128;
        if constexpr (std::is_floating_point_v<T>) value = T(value-T(amount));
        else if constexpr (!std::is_same_v<T,bool>) {
            using Unsigned = std::make_unsigned_t<T>;
            value = static_cast<T>(Unsigned(Unsigned(value)-Unsigned(amount)));
        }
    }
    if constexpr (std::is_floating_point_v<T> || std::is_same_v<T,bool>) return value;
    else {
        const double scaled = double(value)*(1.0/double(std::numeric_limits<T>::max()));
        return std::is_signed_v<T> ? std::max(scaled,-1.0) : scaled;
    }
}
template<class T> T decode_srgb(T value) {
    return value > T(0.04045) ? T(std::pow(T(T(value+T(0.055))/T(1.055)),T(2.4))) : T(value/T(12.92));
}
template<class T> T encode_srgb(T value) {
    value = value > T(0.0031308) ? T(T(T(1.055)*T(std::pow(value,T(1.0/2.4))))-T(0.055)) : T(value*T(12.92));
    // NumPy clipping chooses the bound on equality, including -0 -> +0.
    return value<=T(0) ? T(0) : value>=T(1) ? T(1) : value;
}
template<class Input> void color_pixel(const Input* source, ColorOutput<Input>* out, const ColorSpec& spec, std::ptrdiff_t channel_stride = 1) {
    using T = ColorOutput<Input>;
    T values[3];
    for (int j = 0; j < 3; ++j) {
        const auto offset = std::ptrdiff_t(j)*channel_stride;
        if constexpr (std::is_same_v<Input,bool>) values[j] = color_value(reinterpret_cast<const unsigned char*>(source)[offset] != 0,j,spec.input_offset);
        else values[j] = color_value(source[offset],j,spec.input_offset);
        if (spec.transfer == 1) {
            if constexpr (std::is_same_v<Input,std::uint8_t>) values[j] = srgb_decode_table()[source[offset]];
            else values[j] = decode_srgb(values[j]);
        }
    }
    for (int j = 0; j < 3; ++j) {
        const auto* m = spec.matrix.data()+3*j;
        T value = T(T(T(values[0]*T(m[0]))+T(values[1]*T(m[1])))+T(values[2]*T(m[2])));
        if (spec.output_offset) value = T(value+T(j == 0 ? 16 : 128));
        if (spec.transfer == 2) value = encode_srgb(value);
        out[j] = value;
    }
}
} // namespace skimagexx::detail
