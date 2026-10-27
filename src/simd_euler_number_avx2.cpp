#include "euler_number.hpp"
#include <immintrin.h>

namespace skimagexx::simd {
std::size_t euler_row(const std::uint8_t* current,const std::uint8_t* above,
                      const std::uint8_t* previous,const std::uint8_t* previous_above,
                      std::size_t width,std::ptrdiff_t* histogram){
    std::size_t x=1;alignas(32)std::uint8_t codes[32];
    for(;width-x>=32;x+=32){
        const auto a=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(current+x));
        const auto b=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(current+x-1));
        const auto c=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(above+x));
        const auto d=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(above+x-1));
        const auto e=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(previous+x));
        const auto f=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(previous+x-1));
        const auto g=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(previous_above+x));
        const auto h=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(previous_above+x-1));
        const auto front=_mm256_or_si256(_mm256_or_si256(a,_mm256_slli_epi16(b,2)),_mm256_or_si256(_mm256_slli_epi16(c,1),_mm256_slli_epi16(d,3)));
        const auto back=_mm256_or_si256(_mm256_or_si256(_mm256_slli_epi16(e,4),_mm256_slli_epi16(f,6)),_mm256_or_si256(_mm256_slli_epi16(g,5),_mm256_slli_epi16(h,7)));
        _mm256_store_si256(reinterpret_cast<__m256i*>(codes),_mm256_or_si256(front,back));
        for(auto value:codes)++histogram[value];
    }
    return x;
}
}
