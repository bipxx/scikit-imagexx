#include "skimagexx/max_tree.hpp"
#include <immintrin.h>
#include <cstring>
#include <limits>
#include <type_traits>

namespace skimagexx::simd {
void max_tree_copy(const void* input,void* output,std::size_t bytes) {
    auto* dst=static_cast<unsigned char*>(output);const auto* src=static_cast<const unsigned char*>(input);
    std::size_t i=0;
    for(;bytes-i>=32;i+=32)_mm256_storeu_si256(reinterpret_cast<__m256i*>(dst+i),_mm256_loadu_si256(reinterpret_cast<const __m256i*>(src+i)));
    if(i<bytes)std::memcpy(dst+i,src+i,bytes-i);
}
template<class T> void integral_inverse(const void* input,void* output,std::size_t n) {
    const auto* src=static_cast<const T*>(input);auto* dst=static_cast<T*>(output);std::size_t i=0;
    constexpr auto lanes=32/sizeof(T);const auto ones=_mm256_set1_epi32(-1);
    for(;n-i>=lanes;i+=lanes) {
        auto value=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(src+i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst+i),_mm256_xor_si256(value,ones));
    }
    for(;i<n;++i) {
        if constexpr(std::is_unsigned_v<T>)dst[i]=T(std::numeric_limits<T>::max()-src[i]);
        else dst[i]=T(T(-1)-src[i]);
    }
}
void max_tree_invert(const void* input,void* output,std::size_t n,MaxTreeType type) {
    switch(type) {
#define INTEGER(name,T) case MaxTreeType::name:integral_inverse<T>(input,output,n);return;
        INTEGER(u8,std::uint8_t) INTEGER(u16,std::uint16_t) INTEGER(u32,std::uint32_t) INTEGER(u64,std::uint64_t)
        INTEGER(i8,std::int8_t) INTEGER(i16,std::int16_t) INTEGER(i32,std::int32_t) INTEGER(i64,std::int64_t)
#undef INTEGER
        case MaxTreeType::f32: {
            const auto* src=static_cast<const float*>(input);auto* dst=static_cast<float*>(output);std::size_t i=0;
            for(;n-i>=8;i+=8)_mm256_storeu_ps(dst+i,_mm256_sub_ps(_mm256_set1_ps(1.f),_mm256_loadu_ps(src+i)));
            for(;i<n;++i)dst[i]=1.f-src[i];
            return;
        }
        case MaxTreeType::f64: {
            const auto* src=static_cast<const double*>(input);auto* dst=static_cast<double*>(output);std::size_t i=0;
            for(;n-i>=4;i+=4)_mm256_storeu_pd(dst+i,_mm256_sub_pd(_mm256_set1_pd(1.),_mm256_loadu_pd(src+i)));
            for(;i<n;++i)dst[i]=1.-src[i];
            return;
        }
    }
}
} // namespace skimagexx::simd
