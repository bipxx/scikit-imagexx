#pragma once
#include "skimagexx/perceptual.hpp"
#include "color_detail.hpp"
#include <cstring>

namespace skimagexx::detail {
struct PerceptualSpec {
    PerceptualTransform transform;
    std::array<double,3> white;
    std::size_t channels;
};
constexpr double pi = 3.141592653589793238462643383279502884;
template<class T> void xyz_to_lab(T* v, const std::array<double,3>& white) {
    for (int j = 0; j < 3; ++j) {
        const T scaled = T(v[j]/T(white[j]));
        v[j] = scaled > T(0.008856) ? T(std::cbrt(scaled)) : T(T(T(7.787)*scaled)+T(16.0/116.0));
    }
    const T x = v[0], y = v[1], z = v[2];
    v[0] = T(T(T(116)*y)-T(16)); v[1] = T(T(500)*T(x-y)); v[2] = T(T(200)*T(y-z));
}
template<class T> void lab_to_xyz(T* v, const std::array<double,3>& white, PerceptualInfo& info) {
    const T y = T(T(v[0]+T(16))/T(116));
    const T x = T(T(v[1]/T(500))+y);
    T z = T(y-T(v[2]/T(200)));
    if (z < 0) { ++info.negative_z; z = 0; }
    v[0] = x; v[1] = y; v[2] = z;
    for (int j = 0; j < 3; ++j) {
        const T value = v[j] > T(0.2068966) ? T(std::pow(v[j],T(3))) : T(T(v[j]-T(16.0/116.0))/T(7.787));
        // Upstream multiplies in-place by a float64 reference-white array.
        v[j] = T(double(value)*white[j]);
    }
}
template<class T> void rgb_to_hsv(T* v, PerceptualInfo& info) {
    if (!std::isfinite(v[0]) || !std::isfinite(v[1]) || !std::isfinite(v[2])) {
        info.unsupported = true; v[0] = v[1] = v[2] = 0; return;
    }
    // NumPy reductions select the last equal operand, including signed zero.
    int maximum = 0, minimum = 0;
    for (int j = 1; j < 3; ++j) { if (v[j] >= v[maximum]) maximum = j; if (v[j] <= v[minimum]) minimum = j; }
    using Bits = std::conditional_t<sizeof(T) == 4,std::uint32_t,std::uint64_t>;
    Bits value_bits;
    std::memcpy(&value_bits,v+maximum,sizeof(T));
    const T high = v[maximum], low = v[minimum];
    const T delta = T(high-low);
    T hue = 0, saturation = 0;
    if (delta != 0) {
        saturation = T(delta/high);
        if (v[0] == high) hue = T(T(v[1]-v[2])/delta);
        if (v[1] == high) hue = T(T(2)+T(T(v[2]-v[0])/delta));
        if (v[2] == high) hue = T(T(4)+T(T(v[0]-v[1])/delta));
        hue = T(hue/T(6)); hue = T(hue-T(std::floor(hue)));
    }
    if (!std::isfinite(delta) || !std::isfinite(hue) || !std::isfinite(saturation)) {
        info.unsupported = true; v[0] = v[1] = v[2] = 0; return;
    }
    v[0] = hue; v[1] = saturation;
    std::memcpy(v+2,&value_bits,sizeof(T));
}
template<class T> void hsv_to_rgb(T* v, PerceptualInfo& info) {
    const T h6 = T(v[0]*T(6)), hi = T(std::floor(h6));
    if (!std::isfinite(v[0]) || !std::isfinite(v[1]) || !std::isfinite(v[2]) ||
        !std::isfinite(hi) || double(hi) < -2147483648.0 || double(hi) >= 2147483648.0) {
        info.unsupported = true; v[0] = v[1] = v[2] = 0; return;
    }
    const T f = T(h6-hi), s = v[1], value = v[2];
    const T p = T(value*T(T(1)-s)), q = T(value*T(T(1)-T(f*s))),
            t = T(value*T(T(1)-T(T(T(1)-f)*s)));
    const auto sector = std::uint8_t(static_cast<std::int32_t>(hi))%6;
    const T choices[6][3] = {{value,t,p},{q,value,p},{p,value,t},{p,q,value},{t,p,value},{value,p,q}};
    for (int j = 0; j < 3; ++j) v[j] = choices[sector][j];
}
template<class Input> void perceptual_pixel(const Input* source, ColorOutput<Input>* out,
    const PerceptualSpec& spec, std::ptrdiff_t stride, PerceptualInfo& info) {
    using T = ColorOutput<Input>;
    using Op = PerceptualTransform;
    T v[3];
    for (std::size_t j = 0; j < spec.channels; ++j) {
        const auto offset = std::ptrdiff_t(j)*stride;
        T value;
        if constexpr (std::is_same_v<Input,bool>) value = reinterpret_cast<const unsigned char*>(source)[offset] != 0;
        else value = color_value(source[offset],0,false);
        if (j < 3) v[j] = value; else out[j] = value;
    }
    switch (spec.transform) {
        case Op::rgb2hsv: rgb_to_hsv(v,info); break;
        case Op::hsv2rgb: hsv_to_rgb(v,info); break;
        case Op::rgb2lab: color_pixel(source,v,color_spec(ColorTransform::rgb2xyz),stride); xyz_to_lab(v,spec.white); break;
        case Op::xyz2lab: xyz_to_lab(v,spec.white); break;
        case Op::lab2xyz: lab_to_xyz(v,spec.white,info); break;
        case Op::lab2rgb: {
            lab_to_xyz(v,spec.white,info);
            T rgb[3]; color_pixel(v,rgb,color_spec(ColorTransform::xyz2rgb));
            std::copy(rgb,rgb+3,v); break;
        }
        case Op::lab2lch: {
            const T a = v[1], b = v[2], theta = T(std::atan2(b,a));
            v[1] = T(std::hypot(a,b)); v[2] = T(double(theta)+(theta < 0 ? 2*pi : 0)); break;
        }
        case Op::lch2lab: {
            const T c = v[1], h = v[2];
            v[1] = T(c*T(std::cos(h))); v[2] = T(c*T(std::sin(h))); break;
        }
    }
    std::copy(v,v+3,out);
}
} // namespace skimagexx::detail
