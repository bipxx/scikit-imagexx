#include "perimeter.hpp"
#include <immintrin.h>

namespace skimagexx::simd {
std::size_t perimeter_border(const std::uint8_t* image,const std::uint8_t* eroded,
    std::uint8_t* border,std::size_t count){
    std::size_t i=0;
    for(;count-i>=32;i+=32){
        const auto a=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(image+i));
        const auto b=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(eroded+i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(border+i),_mm256_sub_epi8(a,b));
    }
    return i;
}
std::size_t perimeter_row(const std::uint8_t* previous,const std::uint8_t* current,
    const std::uint8_t* next,std::size_t width,std::ptrdiff_t* histogram){
    std::size_t x=0;alignas(32)std::uint8_t codes[32];
    for(;width-x>=32;x+=32){
        const auto load=[](const std::uint8_t* p){return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));};
        const auto* a=previous+x;const auto* b=current+x;const auto* c=next+x;
        auto edge=_mm256_add_epi8(_mm256_add_epi8(load(a),load(c)),_mm256_add_epi8(load(b-1),load(b+1)));
        auto corner=_mm256_add_epi8(_mm256_add_epi8(load(a-1),load(a+1)),_mm256_add_epi8(load(c-1),load(c+1)));
        edge=_mm256_add_epi8(edge,edge);
        const auto twice=_mm256_add_epi8(corner,corner);
        corner=_mm256_add_epi8(twice,twice);corner=_mm256_add_epi8(corner,corner);
        const auto value=_mm256_add_epi8(load(b),_mm256_add_epi8(edge,_mm256_add_epi8(corner,twice)));
        _mm256_store_si256(reinterpret_cast<__m256i*>(codes),value);
        for(auto code:codes)++histogram[code];
    }
    return x;
}
}
