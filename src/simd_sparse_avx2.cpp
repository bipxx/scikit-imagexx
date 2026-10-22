#include "simd.hpp"
#include <immintrin.h>
namespace skimagexx::simd {
std::size_t sparse_product(const void* input,void* output,std::size_t n,const void* weight,bool single) {
    std::size_t i=0;
    if(single){const auto* a=static_cast<const float*>(input);auto* b=static_cast<float*>(output);const auto w=_mm256_set1_ps(*static_cast<const float*>(weight));for(;i+8<=n;i+=8)_mm256_storeu_ps(b+i,_mm256_mul_ps(w,_mm256_loadu_ps(a+i)));}
    else {const auto* a=static_cast<const double*>(input);auto* b=static_cast<double*>(output);const auto w=_mm256_set1_pd(*static_cast<const double*>(weight));for(;i+4<=n;i+=4)_mm256_storeu_pd(b+i,_mm256_mul_pd(w,_mm256_loadu_pd(a+i)));}
    return i;
}
std::size_t sparse_add(const void* input,void* output,std::size_t n,bool single) {
    std::size_t i=0;
    if(single){const auto* a=static_cast<const float*>(input);auto* b=static_cast<float*>(output);for(;i+8<=n;i+=8)_mm256_storeu_ps(b+i,_mm256_add_ps(_mm256_loadu_ps(b+i),_mm256_loadu_ps(a+i)));}
    else {const auto* a=static_cast<const double*>(input);auto* b=static_cast<double*>(output);for(;i+4<=n;i+=4)_mm256_storeu_pd(b+i,_mm256_add_pd(_mm256_loadu_pd(b+i),_mm256_loadu_pd(a+i)));}
    return i;
}
}
