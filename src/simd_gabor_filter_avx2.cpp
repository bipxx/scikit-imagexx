#include "gabor_filter_detail.hpp"
#include <immintrin.h>

namespace skimagexx::simd {
std::size_t gabor_filter_row(const detail::GaborFilterTap* taps,std::size_t count,
    double* real,double* imaginary,std::size_t width,double cval) {
    const auto constant=_mm256_set1_pd(cval);std::size_t x=0;
    for(;x+8<=width;x+=8){
        auto r0=_mm256_setzero_pd(),r1=r0,i0=r0,i1=r0;
        for(std::size_t t=0;t<count;++t){const auto& tap=taps[t];
            const auto a=tap.row ? _mm256_loadu_pd(tap.row+x) : constant;
            const auto b=tap.row ? _mm256_loadu_pd(tap.row+x+4) : constant;
            if(tap.real!=0){const auto w=_mm256_set1_pd(tap.real);
                r0=_mm256_add_pd(r0,_mm256_mul_pd(w,a));r1=_mm256_add_pd(r1,_mm256_mul_pd(w,b));}
            if(tap.imaginary!=0){const auto w=_mm256_set1_pd(tap.imaginary);
                i0=_mm256_add_pd(i0,_mm256_mul_pd(w,a));i1=_mm256_add_pd(i1,_mm256_mul_pd(w,b));}
        }
        _mm256_storeu_pd(real+x,r0);_mm256_storeu_pd(real+x+4,r1);
        _mm256_storeu_pd(imaginary+x,i0);_mm256_storeu_pd(imaginary+x+4,i1);
    }
    for(;x+4<=width;x+=4){auto r=_mm256_setzero_pd(),i=r;
        for(std::size_t t=0;t<count;++t){const auto& tap=taps[t];const auto value=tap.row ? _mm256_loadu_pd(tap.row+x) : constant;
            if(tap.real!=0)r=_mm256_add_pd(r,_mm256_mul_pd(_mm256_set1_pd(tap.real),value));
            if(tap.imaginary!=0)i=_mm256_add_pd(i,_mm256_mul_pd(_mm256_set1_pd(tap.imaginary),value));}
        _mm256_storeu_pd(real+x,r);_mm256_storeu_pd(imaginary+x,i);
    }
    return x;
}
std::size_t gabor_filter_store_float(const double* input,float* output,std::size_t n) {
    std::size_t i=0;for(;i+4<=n;i+=4)_mm_storeu_ps(output+i,_mm256_cvtpd_ps(_mm256_loadu_pd(input+i)));return i;
}
std::size_t gabor_filter_widen_float(const float* input,double* output,std::size_t n) {
    std::size_t i=0;for(;i+4<=n;i+=4)_mm256_storeu_pd(output+i,_mm256_cvtps_pd(_mm_loadu_ps(input+i)));return i;
}
}
