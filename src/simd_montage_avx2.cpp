#include "montage_simd.hpp"
#include <immintrin.h>
#include <cstring>

namespace skimagexx::simd {
void montage_stride2_words32(const unsigned char* source,unsigned char* output,
                            std::size_t width) {
    const auto even=_mm256_setr_epi32(0,2,4,6,0,2,4,6);
    std::size_t column=0;
    // Two full loads consume64 bytes. Nine remaining selected words span68
    // bytes, so no row padding or byte beyond the final selected word is read.
    for(;width-column>=9;column+=8) {
        const auto* from=source+column*8;
        const auto first=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(from));
        const auto second=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(from+32));
        const auto low=_mm256_permutevar8x32_epi32(first,even);
        const auto high=_mm256_permutevar8x32_epi32(second,even);
        const auto packed=_mm256_inserti128_si256(low,_mm256_castsi256_si128(high),1);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(output+column*4),packed);
    }
    for(;column<width;++column)std::memcpy(output+column*4,source+column*8,4);
}
}
