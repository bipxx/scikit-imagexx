#include "skimagexx/component_consumers.hpp"
#include <immintrin.h>
namespace skimagexx::simd {
std::size_t consumer_truth(const std::intptr_t* input,std::uint8_t* output,std::size_t count,bool invert) {
    if constexpr(sizeof(std::intptr_t)!=8)return 0;
    const auto zero=_mm256_setzero_si256();std::size_t i=0;
    for(;i+4<=count;i+=4) {
        const auto equal=_mm256_cmpeq_epi64(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(input+i)),zero);
        const auto bits=unsigned(_mm256_movemask_pd(_mm256_castsi256_pd(equal)));
        for(unsigned lane=0;lane<4;++lane)output[i+lane]=std::uint8_t(bool(bits&(1u<<lane))==invert);
    }return i;
}
std::size_t consumer_bool_write(const std::uint8_t* input,std::uint8_t* output,std::size_t count,bool invert) {
    const auto toggle=_mm256_set1_epi8(invert ? 1:0);std::size_t i=0;
    for(;i+32<=count;i+=32)_mm256_storeu_si256(reinterpret_cast<__m256i*>(output+i),
        _mm256_xor_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(input+i)),toggle));
    return i;
}
std::size_t consumer_or(const std::uint8_t* input,std::uint8_t* output,std::size_t count) {
    std::size_t i=0;
    for(;i+32<=count;i+=32)_mm256_storeu_si256(reinterpret_cast<__m256i*>(output+i),
        _mm256_or_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(input+i)),
                        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(output+i))));
    return i;
}
}
