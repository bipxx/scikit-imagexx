#pragma once
#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <type_traits>
namespace skimagexx::simd::integer {
template<class T> __m256i broadcast(T value) {
    // Preserve unsigned high bits without an out-of-range signed cast.
    std::make_signed_t<T> bits;std::memcpy(&bits,&value,sizeof(T));
    if constexpr(sizeof(T)==1)return _mm256_set1_epi8(bits);
    else if constexpr(sizeof(T)==2)return _mm256_set1_epi16(bits);
    else if constexpr(sizeof(T)==4)return _mm256_set1_epi32(bits);
    else return _mm256_set1_epi64x(bits);
}
template<class T> __m256i minimum(__m256i a,__m256i b) {
    if constexpr(sizeof(T)==1) {if constexpr(std::is_signed_v<T>)return _mm256_min_epi8(a,b);else return _mm256_min_epu8(a,b);}
    else if constexpr(sizeof(T)==2) {if constexpr(std::is_signed_v<T>)return _mm256_min_epi16(a,b);else return _mm256_min_epu16(a,b);}
    else if constexpr(sizeof(T)==4) {if constexpr(std::is_signed_v<T>)return _mm256_min_epi32(a,b);else return _mm256_min_epu32(a,b);}
    else {
        const auto bias=std::is_signed_v<T> ? _mm256_setzero_si256() : _mm256_set1_epi64x(INT64_MIN);
        return _mm256_blendv_epi8(a,b,_mm256_cmpgt_epi64(_mm256_xor_si256(a,bias),_mm256_xor_si256(b,bias)));
    }
}
template<class T> __m256i maximum(__m256i a,__m256i b) {
    if constexpr(sizeof(T)==1) {if constexpr(std::is_signed_v<T>)return _mm256_max_epi8(a,b);else return _mm256_max_epu8(a,b);}
    else if constexpr(sizeof(T)==2) {if constexpr(std::is_signed_v<T>)return _mm256_max_epi16(a,b);else return _mm256_max_epu16(a,b);}
    else if constexpr(sizeof(T)==4) {if constexpr(std::is_signed_v<T>)return _mm256_max_epi32(a,b);else return _mm256_max_epu32(a,b);}
    else {
        const auto bias=std::is_signed_v<T> ? _mm256_setzero_si256() : _mm256_set1_epi64x(INT64_MIN);
        return _mm256_blendv_epi8(b,a,_mm256_cmpgt_epi64(_mm256_xor_si256(a,bias),_mm256_xor_si256(b,bias)));
    }
}
}
