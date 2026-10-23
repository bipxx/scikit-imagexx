#include "resize_range.hpp"
#include "simd_integer_avx2.hpp"
#include <cmath>
#include <limits>

namespace skimagexx::simd {
namespace {
template<class T,bool Boolean=false> std::size_t integers(const void* input,std::size_t count,void* lower,void* upper,detail::ResizeRangeInfo& info) {
    constexpr auto lanes=32/sizeof(T);if(count<lanes)return 0;
    const auto* data=static_cast<const T*>(input);
    auto low=integer::broadcast(std::numeric_limits<T>::max()),high=integer::broadcast(std::numeric_limits<T>::lowest());
    std::size_t i=0;
    for(;i+lanes<=count;i+=lanes) {
        auto value=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(data+i));
        if constexpr(Boolean)value=_mm256_andnot_si256(_mm256_cmpeq_epi8(value,_mm256_setzero_si256()),_mm256_set1_epi8(1));
        low=integer::minimum<T>(low,value);high=integer::maximum<T>(high,value);
    }
    alignas(32) T lows[lanes],highs[lanes];_mm256_store_si256(reinterpret_cast<__m256i*>(lows),low);_mm256_store_si256(reinterpret_cast<__m256i*>(highs),high);
    T lo=lows[0],hi=highs[0];for(std::size_t lane=1;lane<lanes;++lane){if(lows[lane]<lo)lo=lows[lane];if(highs[lane]>hi)hi=highs[lane];}
    *static_cast<T*>(lower)=lo;*static_cast<T*>(upper)=hi;info.has_value=true;return i;
}
template<class T> std::size_t floating(const void* input,std::size_t count,void* lower,void* upper,detail::ResizeRangeInfo& info) {
    constexpr auto lanes=32/sizeof(T);if(count<lanes)return 0;
    const auto* data=static_cast<const T*>(input);std::size_t i=0;
    alignas(32) T lows[lanes],highs[lanes];
    if constexpr(std::is_same_v<T,float>) {
        const auto infinity=_mm256_set1_ps(std::numeric_limits<T>::infinity()),negative=_mm256_sub_ps(_mm256_setzero_ps(),infinity);
        auto low=infinity,high=negative;
        for(;i+lanes<=count;i+=lanes) {
            const auto value=_mm256_loadu_ps(data+i),nan=_mm256_cmp_ps(value,value,_CMP_UNORD_Q);
            const int nanbits=_mm256_movemask_ps(nan),zeros=_mm256_movemask_ps(_mm256_cmp_ps(value,_mm256_setzero_ps(),_CMP_EQ_OQ)),signs=_mm256_movemask_ps(value);
            info.has_nan|=nanbits!=0;info.has_value|=nanbits!=255;info.positive_zero|=(zeros&~signs)!=0;info.negative_zero|=(zeros&signs)!=0;
            low=_mm256_min_ps(low,_mm256_blendv_ps(value,infinity,nan));high=_mm256_max_ps(high,_mm256_blendv_ps(value,negative,nan));
        }
        _mm256_store_ps(lows,low);_mm256_store_ps(highs,high);
    } else {
        const auto infinity=_mm256_set1_pd(std::numeric_limits<T>::infinity()),negative=_mm256_sub_pd(_mm256_setzero_pd(),infinity);
        auto low=infinity,high=negative;
        for(;i+lanes<=count;i+=lanes) {
            const auto value=_mm256_loadu_pd(data+i),nan=_mm256_cmp_pd(value,value,_CMP_UNORD_Q);
            const int nanbits=_mm256_movemask_pd(nan),zeros=_mm256_movemask_pd(_mm256_cmp_pd(value,_mm256_setzero_pd(),_CMP_EQ_OQ)),signs=_mm256_movemask_pd(value);
            info.has_nan|=nanbits!=0;info.has_value|=nanbits!=15;info.positive_zero|=(zeros&~signs)!=0;info.negative_zero|=(zeros&signs)!=0;
            low=_mm256_min_pd(low,_mm256_blendv_pd(value,infinity,nan));high=_mm256_max_pd(high,_mm256_blendv_pd(value,negative,nan));
        }
        _mm256_store_pd(lows,low);_mm256_store_pd(highs,high);
    }
    T lo=lows[0],hi=highs[0];for(std::size_t lane=1;lane<lanes;++lane){if(lows[lane]<lo)lo=lows[lane];if(highs[lane]>hi)hi=highs[lane];}
    *static_cast<T*>(lower)=lo;*static_cast<T*>(upper)=hi;return i;
}
}
std::size_t resize_range(const void* input,std::size_t count,PixelType type,void* lower,void* upper,detail::ResizeRangeInfo& info) {
    switch(type) {
        case PixelType::boolean:return integers<std::uint8_t,true>(input,count,lower,upper,info);
        #define RANGE(tag,T) case PixelType::tag:return integers<T>(input,count,lower,upper,info);
        RANGE(uint8,std::uint8_t) RANGE(int8,std::int8_t) RANGE(uint16,std::uint16_t) RANGE(int16,std::int16_t)
        RANGE(uint32,std::uint32_t) RANGE(int32,std::int32_t) RANGE(uint64,std::uint64_t) RANGE(int64,std::int64_t)
        #undef RANGE
        case PixelType::float32:return floating<float>(input,count,lower,upper,info);
        case PixelType::float64:return floating<double>(input,count,lower,upper,info);
        default:return 0;
    }
}
}
