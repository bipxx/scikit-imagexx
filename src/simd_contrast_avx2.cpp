#include "simd.hpp"
#include <immintrin.h>
#include <cmath>
#include <limits>

namespace skimagexx::simd {
bool contrast_finite(const void* source,std::size_t count,bool single) {
    std::size_t i=0;
    if(single) {
        const auto* input=static_cast<const float*>(source);
        const auto bound=_mm256_set1_ps(std::numeric_limits<float>::max());
        const auto sign=_mm256_set1_ps(-0.f);
        for(;i+8<=count;i+=8) {
            const auto value=_mm256_andnot_ps(sign,_mm256_loadu_ps(input+i));
            if(_mm256_movemask_ps(_mm256_cmp_ps(value,bound,_CMP_NLE_UQ)))return false;
        }
        for(;i<count;++i)if(!std::isfinite(input[i]))return false;
    } else {
        const auto* input=static_cast<const double*>(source);
        const auto bound=_mm256_set1_pd(std::numeric_limits<double>::max());
        const auto sign=_mm256_set1_pd(-0.);
        for(;i+4<=count;i+=4) {
            const auto value=_mm256_andnot_pd(sign,_mm256_loadu_pd(input+i));
            if(_mm256_movemask_pd(_mm256_cmp_pd(value,bound,_CMP_NLE_UQ)))return false;
        }
        for(;i<count;++i)if(!std::isfinite(input[i]))return false;
    }
    return true;
}

unsigned char contrast_truth(const unsigned char* input,std::size_t count) {
    const auto zero=_mm256_setzero_si256();unsigned char found=0;std::size_t i=0;
    for(;i+32<=count && found!=3;i+=32) {
        const auto value=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(input+i));
        const auto bits=std::uint32_t(_mm256_movemask_epi8(_mm256_cmpeq_epi8(value,zero)));
        if(bits)found|=2;
        if(bits!=0xffffffffu)found|=1;
    }
    for(;i<count && found!=3;++i)found|=input[i] ? 1 : 2;
    return found;
}
}
