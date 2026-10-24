#include <immintrin.h>
#include <cstddef>
namespace skimagexx::simd {
void tv_bregman_copy4(const float* input,float* dx,float* dy,float* bx,float* by,std::size_t n) {
    std::size_t i=0;for(;i+8<=n;i+=8){const auto v=_mm256_loadu_ps(input+i);_mm256_storeu_ps(dx+i,v);_mm256_storeu_ps(dy+i,v);_mm256_storeu_ps(bx+i,v);_mm256_storeu_ps(by+i,v);}
    for(;i<n;++i){const float v=input[i];dx[i]=v;dy[i]=v;bx[i]=v;by[i]=v;}
}
void tv_bregman_copy4(const double* input,double* dx,double* dy,double* bx,double* by,std::size_t n) {
    std::size_t i=0;for(;i+4<=n;i+=4){const auto v=_mm256_loadu_pd(input+i);_mm256_storeu_pd(dx+i,v);_mm256_storeu_pd(dy+i,v);_mm256_storeu_pd(bx+i,v);_mm256_storeu_pd(by+i,v);}
    for(;i<n;++i){const double v=input[i];dx[i]=v;dy[i]=v;bx[i]=v;by[i]=v;}
}
} // namespace skimagexx::simd
