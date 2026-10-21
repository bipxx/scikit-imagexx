#include "simd.hpp"
#include <immintrin.h>
#include <algorithm>
namespace skimagexx::simd {
namespace {
template<class T> std::size_t run(const T* image,std::size_t count,const T* edges,std::size_t bins,std::uint64_t* out) {
    const __m256d low=_mm256_set1_pd(double(edges[0]));
    const __m256d scale=_mm256_set1_pd(double(bins)/(double(edges[bins])-double(edges[0])));
    alignas(16) int indices[4];std::size_t i=0;
    for(;i+4<=count;i+=4) {
        __m256d values;
        if constexpr(sizeof(T)==4)values=_mm256_cvtps_pd(_mm_loadu_ps(image+i));
        else values=_mm256_loadu_pd(image+i);
        const auto estimates=_mm256_mul_pd(_mm256_sub_pd(values,low),scale);
        _mm_store_si128(reinterpret_cast<__m128i*>(indices),_mm256_cvttpd_epi32(estimates));
        for(int j=0;j<4;++j) {
            const auto v=image[i+j];
            if(!(v>=edges[0] && v<=edges[bins]))continue;
            auto b=indices[j]<0 ? std::size_t(0) : std::min(std::size_t(indices[j]),bins-1);
            while(b && v<edges[b])--b;
            while(b+1<bins && v>=edges[b+1])++b;
            ++out[b];
        }
    }
    return i;
}
}
std::size_t histogram_edges(const float* p,std::size_t n,const float* e,std::size_t b,std::uint64_t* o) {return run(p,n,e,b,o);}
std::size_t histogram_edges(const double* p,std::size_t n,const double* e,std::size_t b,std::uint64_t* o) {return run(p,n,e,b,o);}
}
