#include <cstddef>
#include <cstdint>
#include <immintrin.h>
namespace skimagexx::detail {
bool reconstruction_order_u8(const std::uint8_t* a,const std::uint8_t* b,std::size_t n,bool erosion) {
    std::size_t i=0;
    for(;i+32<=n;i+=32){auto x=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a+i));auto y=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b+i));auto m=_mm256_max_epu8(x,y);auto expected=erosion ? x : y;if(_mm256_movemask_epi8(_mm256_cmpeq_epi8(m,expected))!=-1)return false;}
    for(;i<n;++i)if(erosion ? a[i]<b[i] : a[i]>b[i])return false;
    return true;
}
bool reconstruction_order_u16(const std::uint16_t* a,const std::uint16_t* b,std::size_t n,bool erosion) {
    std::size_t i=0;
    for(;i+16<=n;i+=16){auto x=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a+i));auto y=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b+i));auto m=_mm256_max_epu16(x,y);auto expected=erosion ? x : y;if(_mm256_movemask_epi8(_mm256_cmpeq_epi16(m,expected))!=-1)return false;}
    for(;i<n;++i)if(erosion ? a[i]<b[i] : a[i]>b[i])return false;
    return true;
}
void reconstruction_copy_u8_f64(const std::uint8_t* a,double* out,std::size_t n) {
    std::size_t i=0;
    for(;i+8<=n;i+=8){auto x=_mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(a+i)));_mm256_storeu_pd(out+i,_mm256_cvtepi32_pd(_mm256_castsi256_si128(x)));_mm256_storeu_pd(out+i+4,_mm256_cvtepi32_pd(_mm256_extracti128_si256(x,1)));}
    for(;i<n;++i)out[i]=a[i];
}
void reconstruction_copy_u16_f64(const std::uint16_t* a,double* out,std::size_t n) {
    std::size_t i=0;
    for(;i+8<=n;i+=8){auto x=_mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(a+i)));_mm256_storeu_pd(out+i,_mm256_cvtepi32_pd(_mm256_castsi256_si128(x)));_mm256_storeu_pd(out+i+4,_mm256_cvtepi32_pd(_mm256_extracti128_si256(x,1)));}
    for(;i<n;++i)out[i]=a[i];
}
}
