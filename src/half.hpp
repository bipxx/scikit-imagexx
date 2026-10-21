#pragma once
#include <cstdint>
#include <cstring>

namespace skimagexx::detail {
// Portable binary16 storage conversion. AVX2 does not imply F16C support.
inline float decode_half(std::uint16_t value) {
    const auto sign=std::uint32_t(value&0x8000)<<16;
    std::uint32_t mantissa=value&1023,exponent=(value>>10)&31,bits=sign;
    if(exponent==31)bits|=0x7f800000U|(mantissa<<13);
    else if(exponent)bits|=((exponent+112)<<23)|(mantissa<<13);
    else if(mantissa) {
        int shift=0;while((mantissa&1024)==0) {mantissa<<=1;++shift;}
        bits|=(std::uint32_t(113-shift)<<23)|((mantissa&1023)<<13);
    }
    float result;std::memcpy(&result,&bits,sizeof(result));return result;
}

// Round to nearest, ties to even, independently of the host rounding mode.
// NaN sign and representable payload bits survive; never encode NaN as infinity.
inline std::uint16_t encode_half(float value) {
    std::uint32_t bits;std::memcpy(&bits,&value,sizeof(bits));
    const auto sign=std::uint16_t((bits>>16)&0x8000);
    const auto exponent=(bits>>23)&255, fraction=bits&0x7fffff;
    if(exponent==255) {
        const auto payload=std::uint16_t(fraction>>13);
        return std::uint16_t(sign|0x7c00|(fraction ? (payload ? payload : 1) : 0));
    }
    if(exponent>=143)return std::uint16_t(sign|0x7c00);
    if(exponent<113) {
        if(exponent<102)return sign;
        const auto mantissa=fraction|0x800000;
        const auto shift=126-exponent;
        const auto truncated=mantissa>>shift;
        const auto remainder=mantissa&((std::uint32_t(1)<<shift)-1);
        const auto halfway=std::uint32_t(1)<<(shift-1);
        const auto rounded=truncated+std::uint32_t(remainder>halfway || (remainder==halfway && (truncated&1)));
        return std::uint16_t(sign|rounded);
    }
    const auto magnitude=bits&0x7fffffff;
    const auto rounded=((magnitude+0xfff+((magnitude>>13)&1))>>13)-0x1c000;
    return std::uint16_t(sign|rounded);
}
}
