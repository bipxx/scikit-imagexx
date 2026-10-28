#include "unique_rows.hpp"
#include <immintrin.h>
#include <cstring>
#ifdef _MSC_VER
#include <intrin.h>
#endif

namespace skimagexx::simd {
int unique_rows_compare(const unsigned char* first,const unsigned char* second,
                        std::size_t count) {
    std::size_t i=0;
    for(;count-i>=32;i+=32) {
        const auto a=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(first+i));
        const auto b=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(second+i));
        const auto different=~unsigned(_mm256_movemask_epi8(_mm256_cmpeq_epi8(a,b)));
        if(different) {
#ifdef _MSC_VER
            unsigned long position;_BitScanForward(&position,different);
#else
            const auto position=unsigned(__builtin_ctz(different));
#endif
            return int(first[i+position])-int(second[i+position]);
        }
    }
    return std::memcmp(first+i,second+i,count-i);
}
}
