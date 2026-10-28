#include "level_set.hpp"
#include <immintrin.h>
#include <cstring>

namespace skimagexx::simd {
void level_set_checker_row(const std::int8_t* pattern,std::int8_t* output,
                            std::size_t count,bool invert) {
    const auto bit=_mm256_set1_epi8(std::int8_t(invert));std::size_t i=0;
    for(;i+32<=count;i+=32) {
        const auto value=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(pattern+i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(output+i),_mm256_xor_si256(value,bit));
    }
    for(;i<count;++i)output[i]=std::int8_t(pattern[i]^invert);
}
std::size_t level_set_disk_row(const double* last_axis,std::int8_t* output,
                               std::size_t count,double prefix,double radius) {
    const auto base=_mm256_set1_pd(prefix),r=_mm256_set1_pd(radius),zero=_mm256_setzero_pd();
    std::size_t i=0;
    for(;i+4<=count;i+=4) {
        const auto sum=_mm256_add_pd(base,_mm256_loadu_pd(last_axis+i));
        const auto distance=_mm256_sqrt_pd(sum);
        const auto phi=_mm256_sub_pd(r,distance);
        const unsigned bits=unsigned(_mm256_movemask_pd(_mm256_cmp_pd(phi,zero,_CMP_GT_OQ)));
        const std::int8_t packed[4]={std::int8_t(bits&1),std::int8_t((bits>>1)&1),
                                    std::int8_t((bits>>2)&1),std::int8_t((bits>>3)&1)};
        std::memcpy(output+i,packed,4);
    }
    return i;
}
}
