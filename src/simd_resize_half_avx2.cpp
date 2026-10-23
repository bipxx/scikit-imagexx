#include "resize_half.hpp"
#include "simd_integer_avx2.hpp"
#include <immintrin.h>

namespace skimagexx::simd {
namespace {
__m256i magnitude(__m256i value){return _mm256_and_si256(value,_mm256_set1_epi16(0x7fff));}
__m256i nan_mask(__m256i value){return _mm256_cmpgt_epi16(magnitude(value),_mm256_set1_epi16(0x7c00));}
__m256i zero_mask(__m256i value){return _mm256_cmpeq_epi16(magnitude(value),_mm256_setzero_si256());}
__m256i key(__m256i value){return _mm256_xor_si256(value,_mm256_or_si256(_mm256_srai_epi16(value,15),_mm256_set1_epi16(-32768)));}
__m256i greater_key(__m256i a,__m256i b){const auto bias=_mm256_set1_epi16(-32768);return _mm256_cmpgt_epi16(_mm256_xor_si256(a,bias),_mm256_xor_si256(b,bias));}
}
std::size_t half_copy(const void* input,std::uint16_t* output,std::size_t count,bool swap) {
    const auto* bytes=static_cast<const unsigned char*>(input);std::size_t i=0;
    const auto order=_mm256_setr_epi8(1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14,1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14);
    for(;i+16<=count;i+=16) {
        auto word=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(bytes+2*i));if(swap)word=_mm256_shuffle_epi8(word,order);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(output+i),word);
    }
    return i;
}
std::size_t half_range(const std::uint16_t* input,std::size_t count,std::uint16_t* lower,std::uint16_t* upper,detail::ResizeRangeInfo& info) {
    if(count<16)return 0;
    const auto all=_mm256_set1_epi16(-1),zero=_mm256_setzero_si256();auto low=all,high=zero;
    std::size_t i=0;
    for(;i+16<=count;i+=16) {
        const auto value=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(input+i)),nan=nan_mask(value),zeros=zero_mask(value),signs=_mm256_srai_epi16(value,15),keys=key(value);
        const auto bits=_mm256_movemask_epi8(nan);info.has_nan|=bits!=0;info.has_value|=bits!=-1;
        info.positive_zero|=_mm256_movemask_epi8(_mm256_andnot_si256(signs,zeros))!=0;
        info.negative_zero|=_mm256_movemask_epi8(_mm256_and_si256(signs,zeros))!=0;
        low=_mm256_min_epu16(low,_mm256_blendv_epi8(keys,all,nan));high=_mm256_max_epu16(high,_mm256_andnot_si256(nan,keys));
    }
    alignas(32) std::uint16_t lows[16],highs[16];_mm256_store_si256(reinterpret_cast<__m256i*>(lows),low);_mm256_store_si256(reinterpret_cast<__m256i*>(highs),high);
    auto lo=lows[0],hi=highs[0];for(int lane=1;lane<16;++lane){if(lows[lane]<lo)lo=lows[lane];if(highs[lane]>hi)hi=highs[lane];}
    *lower=detail::half_from_key(lo);*upper=detail::half_from_key(hi);return i;
}
std::size_t half_clip(std::uint16_t* data,std::size_t count,std::uint16_t lower,std::uint16_t upper,int policy) {
    const auto low=integer::broadcast(lower),high=integer::broadcast(upper);
    const auto low_key=key(low),high_key=key(high);const bool low_nan=detail::half_nan(lower),high_nan=detail::half_nan(upper);
    const bool fill=policy==1 && (low_nan||high_nan);const auto bound=low_nan ? low : high;
    std::size_t i=0;
    for(;i+16<=count;i+=16) {
        auto value=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(data+i));const auto nan=nan_mask(value);
        if(fill)value=_mm256_blendv_epi8(bound,value,nan);
        else {
            auto keys=key(value);
            if(!low_nan) {
                auto below=_mm256_andnot_si256(nan,greater_key(low_key,keys));
                if(detail::half_zero(lower))below=_mm256_andnot_si256(zero_mask(value),below);
                value=_mm256_blendv_epi8(value,low,below);keys=_mm256_blendv_epi8(keys,low_key,below);
            }
            if(!high_nan) {
                auto above=_mm256_andnot_si256(nan,greater_key(keys,high_key));
                if(detail::half_zero(upper))above=_mm256_andnot_si256(zero_mask(value),above);
                value=_mm256_blendv_epi8(value,high,above);
            }
        }
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(data+i),value);
    }
    return i;
}
}
