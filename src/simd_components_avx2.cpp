#include "skimagexx/components.hpp"
#include <immintrin.h>
#include <cstring>
#include <type_traits>
#ifdef _MSC_VER
#include <intrin.h>
#endif

namespace skimagexx::simd {
namespace {
unsigned first_zero(unsigned mask) {
#ifdef _MSC_VER
    unsigned long bit;
    _BitScanForward(&bit,~mask);
    return unsigned(bit);
#else
    return unsigned(__builtin_ctz(~mask));
#endif
}
template<class T> __m256i expand(const T* p,bool swap) {
    if constexpr(sizeof(T)==1) {
        std::int32_t word;std::memcpy(&word,p,4);const auto x=_mm_cvtsi32_si128(word);
        if constexpr(std::is_signed_v<T>)return _mm256_cvtepi8_epi64(x);else return _mm256_cvtepu8_epi64(x);
    } else if constexpr(sizeof(T)==2) {
        auto x=_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p));
        if(swap)x=_mm_shuffle_epi8(x,_mm_setr_epi8(1,0,3,2,5,4,7,6,8,9,10,11,12,13,14,15));
        if constexpr(std::is_signed_v<T>)return _mm256_cvtepi16_epi64(x);else return _mm256_cvtepu16_epi64(x);
    } else if constexpr(sizeof(T)==4) {
        auto x=_mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
        if(swap)x=_mm_shuffle_epi8(x,_mm_setr_epi8(3,2,1,0,7,6,5,4,11,10,9,8,15,14,13,12));
        if constexpr(std::is_signed_v<T>)return _mm256_cvtepi32_epi64(x);else return _mm256_cvtepu32_epi64(x);
    } else {
        auto x=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
        if(swap)x=_mm256_shuffle_epi8(x,_mm256_setr_epi8(7,6,5,4,3,2,1,0,15,14,13,12,11,10,9,8,
                                                       7,6,5,4,3,2,1,0,15,14,13,12,11,10,9,8));
        return x;
    }
}
template<class T> std::size_t prepare(const void* source,std::intptr_t* output,std::size_t count,bool swap) {
    if constexpr(sizeof(std::intptr_t)!=8)return 0;
    else {
        const auto* input=static_cast<const T*>(source);std::size_t i=0;
        for(;i+4<=count;i+=4)_mm256_storeu_si256(reinterpret_cast<__m256i*>(output+i),expand(input+i,swap));
        return i;
    }
}
template<class T> std::size_t prepare_float(const void* source,std::intptr_t* output,
                                            std::size_t count,bool swap) {
    if(sizeof(std::intptr_t)!=8 || swap)return 0;
    const auto* input=static_cast<const T*>(source);
    std::size_t i=0;
    for(;i+4<=count;i+=4) {
        __m256d value;
        if constexpr(std::is_same_v<T,float>) {
            const auto words=_mm_loadu_si128(reinterpret_cast<const __m128i*>(input+i));
            const auto magnitude=_mm_and_si128(words,_mm_set1_epi32(0x7fffffff));
            const auto valid=_mm_cmpgt_epi32(_mm_set1_epi32(0x4f000000),magnitude);
            if(_mm_movemask_ps(_mm_castsi128_ps(valid))!=15)break;
            value=_mm256_cvtps_pd(_mm_castsi128_ps(words));
        } else {
            value=_mm256_loadu_pd(input+i);
            const auto magnitude=_mm256_and_si256(_mm256_castpd_si256(value),_mm256_set1_epi64x(0x7fffffffffffffffLL));
            const auto valid=_mm256_cmpgt_epi64(_mm256_set1_epi64x(0x41e0000000000000LL),magnitude);
            if(_mm256_movemask_pd(_mm256_castsi256_pd(valid))!=15)break;
        }
        const auto converted=_mm256_cvtepi32_epi64(_mm256_cvttpd_epi32(value));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(output+i),converted);
    }
    return i;
}
}
std::size_t components_prepare_index(const void* input,std::intptr_t* output,std::size_t count,PixelType type,bool swap) {
    switch(type) {
        #define PREP(tag,T) case PixelType::tag:return prepare<T>(input,output,count,swap);
        PREP(uint8,std::uint8_t) PREP(int8,std::int8_t) PREP(uint16,std::uint16_t) PREP(int16,std::int16_t)
        PREP(uint32,std::uint32_t) PREP(int32,std::int32_t) PREP(uint64,std::uint64_t) PREP(int64,std::int64_t)
        #undef PREP
        case PixelType::float32:return prepare_float<float>(input,output,count,swap);
        case PixelType::float64:return prepare_float<double>(input,output,count,swap);
        default:return 0;
    }
}
std::size_t components_prepare_bool(const std::uint8_t* input,std::uint8_t* output,std::size_t count) {
    const auto zero=_mm256_setzero_si256(),one=_mm256_set1_epi8(1);std::size_t i=0;
    for(;i+32<=count;i+=32) {
        const auto value=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(input+i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(output+i),_mm256_andnot_si256(_mm256_cmpeq_epi8(value,zero),one));
    }
    return i;
}
std::size_t components_canonical_bool_prefix(const std::uint8_t* input,std::size_t count) {
    const auto noncanonical=_mm256_set1_epi8(-2);
    std::size_t i=0;
    for(;i+32<=count;i+=32) {
        const auto words=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(input+i));
        if(!_mm256_testz_si256(words,noncanonical))break;
    }
    return i;
}

std::size_t components_float_prefix(const void* input,std::size_t count,PixelType type,bool binary) {
    std::size_t i=0;
    if(type==PixelType::float32) {
        const auto* source=static_cast<const float*>(input);
        const auto signless=_mm256_set1_epi32(0x7fffffff);
        const auto one=_mm256_set1_epi32(0x3f800000);
        for(;i+8<=count;i+=8) {
            const auto words=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(source+i));
            const auto magnitude=_mm256_and_si256(words,signless);
            const auto valid=binary ? _mm256_or_si256(_mm256_cmpeq_epi32(words,one),
                _mm256_cmpeq_epi32(magnitude,_mm256_setzero_si256())) : _mm256_cmpgt_epi32(one,magnitude);
            if(unsigned(_mm256_movemask_epi8(valid))!=0xffffffffu)break;
        }
    } else {
        const auto* source=static_cast<const double*>(input);
        const auto signless=_mm256_set1_epi64x(0x7fffffffffffffffLL);
        const auto one=_mm256_set1_epi64x(0x3ff0000000000000LL);
        for(;i+4<=count;i+=4) {
            const auto words=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(source+i));
            const auto magnitude=_mm256_and_si256(words,signless);
            const auto valid=binary ? _mm256_or_si256(_mm256_cmpeq_epi64(words,one),
                _mm256_cmpeq_epi64(magnitude,_mm256_setzero_si256())) : _mm256_cmpgt_epi64(one,magnitude);
            if(unsigned(_mm256_movemask_epi8(valid))!=0xffffffffu)break;
        }
    }
    return i;
}

std::size_t components_equal_prefix(const void* input,std::size_t count,const void* scalar,std::size_t bytes) {
    // Callers check AVX2 availability and finish the unprocessed suffix.
    if((bytes!=1 && bytes!=2 && bytes!=4 && bytes!=8) || count<32/bytes)return 0;
    std::size_t i=0;
    if(bytes==1) {
        const auto value=_mm256_set1_epi8(char(*static_cast<const std::uint8_t*>(scalar)));
        for(;i+32<=count;i+=32) {
            const auto equal=_mm256_cmpeq_epi8(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(static_cast<const char*>(input)+i)),value);
            const auto mask=unsigned(_mm256_movemask_epi8(equal));if(mask!=0xffffffffu)return i+first_zero(mask);
        }
    } else if(bytes==2) {
        std::int16_t word;std::memcpy(&word,scalar,2);const auto value=_mm256_set1_epi16(word);
        for(;i+16<=count;i+=16) {
            const auto equal=_mm256_cmpeq_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(static_cast<const char*>(input)+i*2)),value);
            const auto mask=unsigned(_mm256_movemask_epi8(equal));if(mask!=0xffffffffu)return i+first_zero(mask)/2;
        }
    } else if(bytes==4) {
        std::int32_t word;std::memcpy(&word,scalar,4);const auto value=_mm256_set1_epi32(word);
        for(;i+8<=count;i+=8) {
            const auto equal=_mm256_cmpeq_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(static_cast<const char*>(input)+i*4)),value);
            const auto mask=unsigned(_mm256_movemask_ps(_mm256_castsi256_ps(equal)));if(mask!=255)return i+first_zero(mask);
        }
    } else {
        std::int64_t word;std::memcpy(&word,scalar,8);const auto value=_mm256_set1_epi64x(word);
        for(;i+4<=count;i+=4) {
            const auto equal=_mm256_cmpeq_epi64(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(static_cast<const char*>(input)+i*8)),value);
            const auto mask=unsigned(_mm256_movemask_pd(_mm256_castsi256_pd(equal)));if(mask!=15)return i+first_zero(mask);
        }
    }
    return i;
}
std::size_t components_fill(void* output,std::size_t count,const void* scalar,std::size_t bytes) {
    std::size_t i=0;
    if(bytes==4) {
        std::int32_t word;std::memcpy(&word,scalar,4);const auto value=_mm256_set1_epi32(word);
        for(;i+8<=count;i+=8)_mm256_storeu_si256(reinterpret_cast<__m256i*>(static_cast<char*>(output)+i*4),value);
    } else {
        std::int64_t word;std::memcpy(&word,scalar,8);const auto value=_mm256_set1_epi64x(word);
        for(;i+4<=count;i+=4)_mm256_storeu_si256(reinterpret_cast<__m256i*>(static_cast<char*>(output)+i*8),value);
    }
    return i;
}
}
