#include "simd.hpp"
#include <immintrin.h>
#include <cstring>

namespace skimagexx::simd {
// Caller bounds every unsigned key-low to the table extent (<65536), so the
// signed gather offsets are valid even when the original keys exceed INT64_MAX.
std::size_t label_lookup(const std::uint64_t* input,std::uint64_t* output,std::size_t count,
                         std::uint64_t low,const std::uint64_t* table) {
    long long low_bits;std::memcpy(&low_bits,&low,sizeof(low));
    const auto base=_mm256_set1_epi64x(low_bits);
    std::size_t i=0;
    for(;i+4<=count;i+=4) {
        const auto keys=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(input+i));
        const auto indices=_mm256_sub_epi64(keys,base);
        const auto ranks=_mm256_i64gather_epi64(reinterpret_cast<const long long*>(table),indices,8);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(output+i),ranks);
    }
    return i;
}
}
