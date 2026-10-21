#include "simd.hpp"
#include "skimagexx/local_threshold.hpp"
#include <immintrin.h>
namespace skimagexx::simd {
namespace {
template<int Groups> void prefix_columns(double* sums,double* squares,std::size_t length,std::size_t stride) {
    __m256d mean[Groups],square[Groups];
    for(int group=0;group<Groups;++group) {mean[group]=_mm256_loadu_pd(sums+4*group);square[group]=_mm256_loadu_pd(squares+4*group);}
    for(std::size_t row=1;row<length;++row) {
        for(int group=0;group<Groups;++group) {
            const auto offset=row*stride+4*group;
            mean[group]=_mm256_add_pd(mean[group],_mm256_loadu_pd(sums+offset));
            square[group]=_mm256_add_pd(square[group],_mm256_loadu_pd(squares+offset));
            _mm256_storeu_pd(sums+offset,mean[group]);_mm256_storeu_pd(squares+offset,square[group]);
        }
    }
}
__m256d clipped_sqrt(__m256d variance) {
    const auto clipped=_mm256_max_pd(variance,_mm256_setzero_pd());
    return _mm256_sqrt_pd(_mm256_blendv_pd(clipped,variance,_mm256_cmp_pd(variance,variance,_CMP_UNORD_Q)));
}
__m128 clipped_sqrt(__m128 variance) {
    const auto clipped=_mm_max_ps(variance,_mm_setzero_ps());
    return _mm_sqrt_ps(_mm_blendv_ps(clipped,variance,_mm_cmpunord_ps(variance,variance)));
}
}
std::size_t local_prefix_columns(double* sums,double* squares,std::size_t length,std::size_t stride,std::size_t lanes) {
    if(lanes>=8) {prefix_columns<2>(sums,squares,length,stride);return 8;}
    if(lanes>=4) {prefix_columns<1>(sums,squares,length,stride);return 4;}
    return 0;
}
std::size_t local_threshold_rows(const double* const* means,const double* const* squares,const int* signs,
    std::size_t corners,void* output,std::size_t width,std::size_t area,bool single,bool wide_output,
    LocalThreshold method,const LocalThresholdParameters& p) {
    const auto zero=_mm256_setzero_pd(),ones=_mm256_set1_pd(1.0),divisor=_mm256_set1_pd(double(area));
    const auto sign=_mm256_set1_pd(-0.0);
    std::size_t x=0;
    for(;x+4<=width;x+=4) {
        auto sum=_mm256_loadu_pd(means[0]+x),square=_mm256_loadu_pd(squares[0]+x);
        if(signs[0]<0) {sum=_mm256_xor_pd(sum,sign);square=_mm256_xor_pd(square,sign);}
        for(std::size_t corner=1;corner<corners;++corner) {
            const auto a=_mm256_loadu_pd(means[corner]+x),b=_mm256_loadu_pd(squares[corner]+x);
            if(signs[corner]>0) {sum=_mm256_add_pd(sum,a);square=_mm256_add_pd(square,b);}
            else {sum=_mm256_sub_pd(sum,a);square=_mm256_sub_pd(square,b);}
        }
        if(!single) {
            const auto mean=_mm256_div_pd(sum,divisor),mean_square=_mm256_div_pd(square,divisor);
            const auto sd=clipped_sqrt(_mm256_sub_pd(mean_square,_mm256_mul_pd(mean,mean)));
            const auto k=_mm256_set1_pd(p.k);
            const auto value=method==LocalThreshold::niblack ? _mm256_sub_pd(mean,_mm256_mul_pd(k,sd)) :
                _mm256_mul_pd(mean,_mm256_add_pd(ones,_mm256_mul_pd(k,_mm256_sub_pd(_mm256_div_pd(sd,_mm256_set1_pd(p.r)),ones))));
            _mm256_storeu_pd(static_cast<double*>(output)+x,value);
            continue;
        }
        auto mean=_mm256_cvtpd_ps(sum),mean_square=_mm256_cvtpd_ps(square);
        if(p.divide_double) {
            mean=_mm256_cvtpd_ps(_mm256_div_pd(_mm256_cvtps_pd(mean),divisor));
            mean_square=_mm256_cvtpd_ps(_mm256_div_pd(_mm256_cvtps_pd(mean_square),divisor));
        } else {
            const auto divisor32=_mm_set1_ps(float(area));
            mean=_mm_div_ps(mean,divisor32);mean_square=_mm_div_ps(mean_square,divisor32);
        }
        const auto sd=clipped_sqrt(_mm_sub_ps(mean_square,_mm_mul_ps(mean,mean)));
        if(method==LocalThreshold::niblack && p.k_double) {
            const auto value=_mm256_sub_pd(_mm256_cvtps_pd(mean),_mm256_mul_pd(_mm256_set1_pd(p.k),_mm256_cvtps_pd(sd)));
            _mm256_storeu_pd(static_cast<double*>(output)+x,value);continue;
        }
        if(method==LocalThreshold::sauvola && (p.r_double || p.k_double)) {
            __m256d inner;
            if(p.r_double)inner=_mm256_sub_pd(_mm256_div_pd(_mm256_cvtps_pd(sd),_mm256_set1_pd(p.r)),ones);
            else inner=_mm256_cvtps_pd(_mm_sub_ps(_mm_div_ps(sd,_mm_set1_ps(float(p.r))),_mm_set1_ps(1.0f)));
            const auto value=_mm256_mul_pd(_mm256_cvtps_pd(mean),_mm256_add_pd(ones,_mm256_mul_pd(_mm256_set1_pd(p.k),inner)));
            _mm256_storeu_pd(static_cast<double*>(output)+x,value);continue;
        }
        const auto value=method==LocalThreshold::niblack ? _mm_sub_ps(mean,_mm_mul_ps(_mm_set1_ps(float(p.k)),sd)) :
            _mm_mul_ps(mean,_mm_add_ps(_mm_set1_ps(1.0f),_mm_mul_ps(_mm_set1_ps(float(p.k)),_mm_sub_ps(_mm_div_ps(sd,_mm_set1_ps(float(p.r))),_mm_set1_ps(1.0f)))));
        if(wide_output)_mm256_storeu_pd(static_cast<double*>(output)+x,_mm256_cvtps_pd(value));
        else _mm_storeu_ps(static_cast<float*>(output)+x,value);
    }
    (void)zero;
    return x;
}
}
