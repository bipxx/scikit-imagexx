#include "window_internal.hpp"
#include <immintrin.h>

namespace skimagexx::simd {
void window_sample(const double* window,std::size_t length,const float* coords,double* output,
                   std::size_t begin,std::size_t end,int order) {
    const auto zero=_mm256_setzero_pd(),one=_mm256_set1_pd(1.);
    const auto lower=_mm256_set1_pd(order==0 ? -.5 : -1.);
    const auto upper=_mm256_set1_pd(double(length));
    auto i=begin;
    for(;end-i>=4;i+=4) {
        auto x=_mm256_cvtps_pd(_mm_loadu_ps(coords+i));
        auto valid=_mm256_and_pd(_mm256_cmp_pd(x,lower,_CMP_GE_OQ),_mm256_cmp_pd(x,upper,_CMP_LT_OQ));
        // Normal bounds handling also prevents unsafe casts for unclaimed
        // nonfinite private inputs; no extra coordinate-validation pass.
        x=_mm256_blendv_pd(zero,x,valid);
        auto start=order==0 ? _mm256_floor_pd(_mm256_add_pd(x,_mm256_set1_pd(.5))) : _mm256_floor_pd(x);
        double indices[4],first[4]{},second[4]{};
        _mm256_storeu_pd(indices,start);
        const auto mask=_mm256_movemask_pd(valid);
        for(int lane=0;lane<4;++lane) if(mask&(1<<lane)) {
            const auto index=std::ptrdiff_t(indices[lane]);
            first[lane]=detail::window_tap(window,length,index);
            if(order)second[lane]=detail::window_tap(window,length,index+1);
        }
        auto a=_mm256_loadu_pd(first),b=_mm256_loadu_pd(second);
        if(order) {
            auto wa=_mm256_sub_pd(one,_mm256_sub_pd(x,start));
            auto wb=_mm256_sub_pd(one,wa);
            a=_mm256_mul_pd(a,wa);b=_mm256_mul_pd(b,wb);
        }
        auto total=_mm256_add_pd(zero,a);
        if(order)total=_mm256_add_pd(total,b);
        _mm256_storeu_pd(output+i,total);
    }
    for(;i<end;++i)output[i]=detail::window_sample_one(window,length,double(coords[i]),order);
}
}
