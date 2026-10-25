#include <immintrin.h>
#include <cstddef>
#include <cstdint>
namespace skimagexx::simd {
// Separate optional TU. Caller has authenticated spans and runtime AVX2.
// Boolean storage is nonzero truth, never raw-byte bitwise conjunction.
std::uint64_t coloc_bool_count(const unsigned char* a,const unsigned char* b,std::size_t n) {
    const auto zero=_mm256_setzero_si256();std::uint64_t count=0;std::size_t i=0;
    for(;n-i>=32;i+=32) {
        auto empty=_mm256_cmpeq_epi8(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a+i)),zero);
        if(b)empty=_mm256_or_si256(empty,_mm256_cmpeq_epi8(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b+i)),zero));
        auto bits=~std::uint32_t(_mm256_movemask_epi8(empty));
        // Avoid assuming a separate POPCNT CPU feature or compiler ISA flag.
        bits=bits-((bits>>1)&0x55555555u);bits=(bits&0x33333333u)+((bits>>2)&0x33333333u);
        count+=(((bits+(bits>>4))&0x0f0f0f0fu)*0x01010101u)>>24;
    }
    for(;i<n;++i)count+=(a[i]!=0)&&(!b || b[i]!=0);
    return count;
}
}
