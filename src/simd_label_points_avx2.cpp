#include "label_points.hpp"
#include <immintrin.h>

namespace skimagexx::simd {
void label_points_zero(std::uint64_t* output,std::size_t count) {
    const auto zero=_mm256_setzero_si256();
    std::size_t i=0;
    for(;count-i>=16;i+=16) {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(output+i),zero);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(output+i+4),zero);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(output+i+8),zero);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(output+i+12),zero);
    }
    for(;i<count;++i)output[i]=0;
}
}
