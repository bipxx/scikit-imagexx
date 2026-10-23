#include "skimagexx/footprint_dense.hpp"
#include <immintrin.h>
#include <cstdint>
#include <cstring>

namespace skimagexx::simd {
std::size_t footprint_dense_row(void* output,std::size_t itemsize,const void* zero,
                               const void* one,const double* axis,std::size_t width,
                               double middle,double outer,bool ball,double limit) {
    auto* bytes=static_cast<unsigned char*>(output);
    const auto m=_mm256_set1_pd(middle),o=_mm256_set1_pd(outer),bound=_mm256_set1_pd(limit);
    std::size_t column=0;
    for(;column+4<=width;column+=4) {
        auto cost=_mm256_add_pd(_mm256_loadu_pd(axis+column),m);
        if(ball)cost=_mm256_add_pd(cost,o);
        const auto comparison=_mm256_cmp_pd(cost,bound,_CMP_LE_OQ);
        const auto mask=_mm256_movemask_pd(comparison);
        if(itemsize==1) {
            const auto z=*static_cast<const unsigned char*>(zero),v=*static_cast<const unsigned char*>(one);
            const unsigned char packed[4]={(mask&1) ? v:z,(mask&2) ? v:z,(mask&4) ? v:z,(mask&8) ? v:z};
            std::memcpy(bytes+column,packed,4);
        } else if(itemsize==4) {
            std::int32_t z,v;std::memcpy(&z,zero,4);std::memcpy(&v,one,4);
            const auto choose=_mm_set_epi32(-(bool(mask&8)),-(bool(mask&4)),-(bool(mask&2)),-(bool(mask&1)));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(bytes+column*4),
                _mm_blendv_epi8(_mm_set1_epi32(z),_mm_set1_epi32(v),choose));
        } else if(itemsize==8) {
            std::int64_t z,v;std::memcpy(&z,zero,8);std::memcpy(&v,one,8);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(bytes+column*8),
                _mm256_blendv_epi8(_mm256_set1_epi64x(z),_mm256_set1_epi64x(v),_mm256_castpd_si256(comparison)));
        } else for(int lane=0;lane<4;++lane)
            std::memcpy(bytes+(column+std::size_t(lane))*itemsize,(mask&(1<<lane)) ? one:zero,itemsize);
    }
    return column;
}
}
