#include "simd.hpp"
#include <immintrin.h>
#include <cstring>
#include <type_traits>
namespace skimagexx::simd {
namespace {
template<class T,bool Boolean=false> std::size_t cast_words(const void* source,std::uint64_t* output,std::size_t count) {
    const auto* input=static_cast<const T*>(source);std::size_t i=0;
    for(;i+4<=count;i+=4) {
        __m256i values;
        if constexpr(sizeof(T)==1) {
            std::uint32_t word;std::memcpy(&word,input+i,4);
            const auto bytes=_mm_cvtsi32_si128(int(word));
            values=std::is_signed_v<T> ? _mm256_cvtepi8_epi64(bytes) : _mm256_cvtepu8_epi64(bytes);
        } else if constexpr(sizeof(T)==2) {
            const auto words=_mm_loadl_epi64(reinterpret_cast<const __m128i*>(input+i));
            values=std::is_signed_v<T> ? _mm256_cvtepi16_epi64(words) : _mm256_cvtepu16_epi64(words);
        } else if constexpr(sizeof(T)==4) {
            const auto words=_mm_loadu_si128(reinterpret_cast<const __m128i*>(input+i));
            values=std::is_signed_v<T> ? _mm256_cvtepi32_epi64(words) : _mm256_cvtepu32_epi64(words);
        } else values=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(input+i));
        if constexpr(Boolean)values=_mm256_andnot_si256(_mm256_cmpeq_epi64(values,_mm256_setzero_si256()),_mm256_set1_epi64x(1));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(output+i),values);
    }
    return i;
}
}
std::size_t integral_cast_u64(const void* input,std::uint64_t* output,std::size_t count,PixelType type) {
#define RAW(tag,T) case PixelType::tag:return cast_words<T>(input,output,count);
    switch(type) {
    case PixelType::boolean:return cast_words<std::uint8_t,true>(input,output,count);
    RAW(uint8,std::uint8_t) RAW(int8,std::int8_t) RAW(uint16,std::uint16_t) RAW(int16,std::int16_t)
    RAW(uint32,std::uint32_t) RAW(int32,std::int32_t) RAW(uint64,std::uint64_t) RAW(int64,std::int64_t)
    default:return 0;
    }
#undef RAW
}
namespace {
inline __m256i prefix_four(__m256i value) {
    value=_mm256_add_epi64(value,_mm256_slli_si256(value,8));
    const auto low=_mm256_castsi256_si128(value);
    const auto cross=_mm256_inserti128_si256(_mm256_setzero_si256(),_mm_unpackhi_epi64(low,low),1);
    return _mm256_add_epi64(value,cross);
}
inline __m256i last_word(__m256i value) {return _mm256_permute4x64_epi64(value,_MM_SHUFFLE(3,3,3,3));}
}
std::size_t integral_prefix_u64(const std::uint64_t* input,std::uint64_t* output,std::size_t count) {
    auto carry=_mm256_setzero_si256();std::size_t i=0;
    for(;i+4<=count;i+=4) {
        const auto value=_mm256_add_epi64(prefix_four(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(input+i))),carry);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(output+i),value);carry=last_word(value);
    }
    return i;
}
std::size_t integral_add(const void* previous,void* current,std::size_t count,PixelType type) {
    std::size_t i=0;
    if(type==PixelType::float32) {
        const auto* before=static_cast<const float*>(previous);auto* values=static_cast<float*>(current);
        for(;i+8<=count;i+=8)_mm256_storeu_ps(values+i,_mm256_add_ps(_mm256_loadu_ps(before+i),_mm256_loadu_ps(values+i)));
    } else if(type==PixelType::float64) {
        const auto* before=static_cast<const double*>(previous);auto* values=static_cast<double*>(current);
        for(;i+4<=count;i+=4)_mm256_storeu_pd(values+i,_mm256_add_pd(_mm256_loadu_pd(before+i),_mm256_loadu_pd(values+i)));
        if(i+2<=count) {_mm_storeu_pd(values+i,_mm_add_pd(_mm_loadu_pd(before+i),_mm_loadu_pd(values+i)));i+=2;}
    } else {
        const auto* before=static_cast<const std::uint64_t*>(previous);auto* values=static_cast<std::uint64_t*>(current);
        for(;i+4<=count;i+=4)_mm256_storeu_si256(reinterpret_cast<__m256i*>(values+i),_mm256_add_epi64(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(before+i)),_mm256_loadu_si256(reinterpret_cast<const __m256i*>(values+i))));
    }
    return i;
}
}
