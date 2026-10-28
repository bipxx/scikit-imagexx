#include "regular_seeds.hpp"
#include <immintrin.h>
#include <cstring>

namespace skimagexx::simd {
void regular_seeds_zero(void* output,std::size_t bytes) {
    auto* destination=static_cast<char*>(output);
    const auto zero=_mm256_setzero_si256();
    std::size_t i=0;
    for(;bytes-i>=32;i+=32)_mm256_storeu_si256(reinterpret_cast<__m256i*>(destination+i),zero);
    if(i<bytes)std::memset(destination+i,0,bytes-i);
}
void regular_seeds_u32_run(std::uint32_t* output,std::size_t count,std::uint64_t first) {
    auto value=_mm256_setr_epi32(int(std::uint32_t(first)),int(std::uint32_t(first+1)),int(std::uint32_t(first+2)),int(std::uint32_t(first+3)),int(std::uint32_t(first+4)),int(std::uint32_t(first+5)),int(std::uint32_t(first+6)),int(std::uint32_t(first+7)));
    const auto increment=_mm256_set1_epi32(8);std::size_t i=0;
    for(;count-i>=8;i+=8) {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(output+i),value);
        value=_mm256_add_epi32(value,increment);
    }
    for(;i<count;++i)output[i]=std::uint32_t(first+i);
}
void regular_seeds_u64_run(std::uint64_t* output,std::size_t count,std::uint64_t first) {
    auto value=_mm256_setr_epi64x(first,first+1,first+2,first+3);
    const auto increment=_mm256_set1_epi64x(4);std::size_t i=0;
    for(;count-i>=4;i+=4) {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(output+i),value);
        value=_mm256_add_epi64(value,increment);
    }
    for(;i<count;++i)output[i]=first+i;
}
}
