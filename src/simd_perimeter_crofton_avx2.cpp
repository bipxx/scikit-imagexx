#include "perimeter_crofton.hpp"
#include <immintrin.h>
#include <cstring>

namespace skimagexx::simd {
std::size_t crofton_threshold(const void* input,PixelType type,std::uint8_t* output,std::size_t count){
    const auto zero=_mm256_setzero_si256(),ones=_mm256_set1_epi32(-1);
    const auto bytes=type==PixelType::boolean||type==PixelType::uint8||type==PixelType::int8?1u:
        type==PixelType::uint16||type==PixelType::int16?2u:
        type==PixelType::uint32||type==PixelType::int32||type==PixelType::float32?4u:8u;
    const auto lanes=32/bytes;const auto* data=static_cast<const unsigned char*>(input);std::size_t i=0;
    for(;count-i>=lanes;i+=lanes){
        const auto value=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(data+i*bytes));__m256i mask;
        switch(type){
        case PixelType::int8:mask=_mm256_cmpgt_epi8(value,zero);break;
        case PixelType::int16:mask=_mm256_cmpgt_epi16(value,zero);break;
        case PixelType::int32:mask=_mm256_cmpgt_epi32(value,zero);break;
        case PixelType::int64:mask=_mm256_cmpgt_epi64(value,zero);break;
        case PixelType::float32:
            mask=_mm256_and_si256(_mm256_cmpgt_epi32(value,zero),_mm256_xor_si256(ones,_mm256_cmpgt_epi32(value,_mm256_set1_epi32(0x7f800000))));break;
        case PixelType::float64:
            mask=_mm256_and_si256(_mm256_cmpgt_epi64(value,zero),_mm256_xor_si256(ones,_mm256_cmpgt_epi64(value,_mm256_set1_epi64x(INT64_C(0x7ff0000000000000)))));break;
        case PixelType::boolean:case PixelType::uint8:mask=_mm256_xor_si256(ones,_mm256_cmpeq_epi8(value,zero));break;
        case PixelType::uint16:mask=_mm256_xor_si256(ones,_mm256_cmpeq_epi16(value,zero));break;
        case PixelType::uint32:mask=_mm256_xor_si256(ones,_mm256_cmpeq_epi32(value,zero));break;
        case PixelType::uint64:mask=_mm256_xor_si256(ones,_mm256_cmpeq_epi64(value,zero));break;
        default:return i;
        }
        if(bytes==1)_mm256_storeu_si256(reinterpret_cast<__m256i*>(output+i),_mm256_and_si256(mask,_mm256_set1_epi8(1)));
        else if(bytes==4||bytes==8){
            const auto words=_mm_packs_epi32(_mm256_castsi256_si128(mask),_mm256_extracti128_si256(mask,1));
            const auto packed=_mm_and_si128(_mm_packs_epi16(words,words),_mm_set1_epi8(1));
            if(bytes==4)_mm_storel_epi64(reinterpret_cast<__m128i*>(output+i),packed);
            else{
                // Each 64-bit mask produced two identical bytes; keep one per lane.
                const auto distinct=_mm_shuffle_epi8(packed,_mm_setr_epi8(0,2,4,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1));
                const auto value=std::uint32_t(_mm_cvtsi128_si32(distinct));
                std::memcpy(output+i,&value,sizeof(value));
            }
        }
        else{
            const auto bits=std::uint32_t(_mm256_movemask_epi8(mask));
            for(std::size_t j=0;j<lanes;++j)output[i+j]=(bits>>(j*bytes))&1u;
        }
    }
    return i;
}
std::size_t crofton_row(const std::uint8_t* current,const std::uint8_t* previous,std::size_t width,std::ptrdiff_t* histogram){
    std::size_t x=1;alignas(32)std::uint8_t codes[32];
    for(;width-x>=32;x+=32){
        const auto a=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(current+x));
        const auto b=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(current+x-1));
        const auto c=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(previous+x));
        const auto d=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(previous+x-1));
        const auto code=_mm256_or_si256(_mm256_or_si256(a,_mm256_slli_epi16(b,2)),_mm256_or_si256(_mm256_slli_epi16(c,1),_mm256_slli_epi16(d,3)));
        _mm256_store_si256(reinterpret_cast<__m256i*>(codes),code);
        for(auto value:codes)++histogram[value];
    }
    return x;
}
}
