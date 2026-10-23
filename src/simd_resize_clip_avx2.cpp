#include "resize_clip.hpp"
#include <immintrin.h>
#include <cstring>
#include <type_traits>

namespace skimagexx::simd {
namespace {
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
template<class T> std::size_t integers(void* image,std::size_t count,const void* lower,const void* upper) {
    auto* data=static_cast<T*>(image);const auto lo=broadcast(*static_cast<const T*>(lower)),hi=broadcast(*static_cast<const T*>(upper));
    std::size_t i=0;
    for(;i+32/sizeof(T)<=count;i+=32/sizeof(T)) {
        auto value=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(data+i));
        value=minimum<T>(maximum<T>(value,lo),hi);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(data+i),value);
    }
    return i;
}
template<class T> std::size_t floating(void* image,std::size_t count,const void* lower,const void* upper,bool legacy) {
    auto* data=static_cast<T*>(image);std::size_t i=0;
    if constexpr(std::is_same_v<T,float>) {
        const auto low=_mm256_set1_ps(*static_cast<const T*>(lower)),high=_mm256_set1_ps(*static_cast<const T*>(upper));
        for(;i+8<=count;i+=8) {
            auto value=_mm256_loadu_ps(data+i);
            const auto below=legacy ? _mm256_cmp_ps(value,low,_CMP_LE_OQ) : _mm256_cmp_ps(value,low,_CMP_LT_OQ);
            value=_mm256_blendv_ps(value,low,below);
            const auto above=legacy ? _mm256_cmp_ps(value,high,_CMP_GE_OQ) : _mm256_cmp_ps(value,high,_CMP_GT_OQ);
            _mm256_storeu_ps(data+i,_mm256_blendv_ps(value,high,above));
        }
    } else {
        const auto low=_mm256_set1_pd(*static_cast<const T*>(lower)),high=_mm256_set1_pd(*static_cast<const T*>(upper));
        for(;i+4<=count;i+=4) {
            auto value=_mm256_loadu_pd(data+i);
            const auto below=legacy ? _mm256_cmp_pd(value,low,_CMP_LE_OQ) : _mm256_cmp_pd(value,low,_CMP_LT_OQ);
            value=_mm256_blendv_pd(value,low,below);
            const auto above=legacy ? _mm256_cmp_pd(value,high,_CMP_GE_OQ) : _mm256_cmp_pd(value,high,_CMP_GT_OQ);
            _mm256_storeu_pd(data+i,_mm256_blendv_pd(value,high,above));
        }
    }
    return i;
}
}
std::size_t resize_clip_nan(void* image,std::size_t count,PixelType type,const void* nan) {
    std::size_t i=0;
    if(type==PixelType::float32) {
        auto* data=static_cast<float*>(image);const auto bound=_mm256_set1_ps(*static_cast<const float*>(nan));
        for(;i+8<=count;i+=8) {
            const auto value=_mm256_loadu_ps(data+i);
            _mm256_storeu_ps(data+i,_mm256_blendv_ps(bound,value,_mm256_cmp_ps(value,value,_CMP_UNORD_Q)));
        }
    } else {
        auto* data=static_cast<double*>(image);const auto bound=_mm256_set1_pd(*static_cast<const double*>(nan));
        for(;i+4<=count;i+=4) {
            const auto value=_mm256_loadu_pd(data+i);
            _mm256_storeu_pd(data+i,_mm256_blendv_pd(bound,value,_mm256_cmp_pd(value,value,_CMP_UNORD_Q)));
        }
    }
    return i;
}
std::size_t resize_clip(void* image,std::size_t count,PixelType type,const void* lower,const void* upper,bool legacy) {
    switch(type) {
        case PixelType::boolean:return integers<std::uint8_t>(image,count,lower,upper);
        #define CLIP(tag,T) case PixelType::tag:return integers<T>(image,count,lower,upper);
        CLIP(uint8,std::uint8_t) CLIP(int8,std::int8_t) CLIP(uint16,std::uint16_t) CLIP(int16,std::int16_t)
        CLIP(uint32,std::uint32_t) CLIP(int32,std::int32_t) CLIP(uint64,std::uint64_t) CLIP(int64,std::int64_t)
        #undef CLIP
        case PixelType::float32:return floating<float>(image,count,lower,upper,legacy);
        case PixelType::float64:return floating<double>(image,count,lower,upper,legacy);
        default:return 0;
    }
}
}
