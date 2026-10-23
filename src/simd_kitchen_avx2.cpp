#include "kitchen_detail.hpp"
#include <immintrin.h>
#include <limits>
namespace skimagexx::simd {
std::size_t kitchen_response(const void* const* fields,void* output,std::size_t count,bool single,bool& valid){
    std::size_t i=0;
    if(single){
        const auto zero=_mm256_setzero_ps(),one=_mm256_set1_ps(1.f),two=_mm256_set1_ps(2.f),sign=_mm256_set1_ps(-0.f),limit=_mm256_set1_ps(std::numeric_limits<float>::max());
        for(;i+8<=count;i+=8){
            const auto y=_mm256_loadu_ps(static_cast<const float*>(fields[0])+i),x=_mm256_loadu_ps(static_cast<const float*>(fields[1])+i);
            const auto xy=_mm256_loadu_ps(static_cast<const float*>(fields[2])+i),xx=_mm256_loadu_ps(static_cast<const float*>(fields[3])+i),yy=_mm256_loadu_ps(static_cast<const float*>(fields[4])+i);
            const auto y2=_mm256_mul_ps(y,y),x2=_mm256_mul_ps(x,x);
            const auto numerator=_mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(xx,y2),_mm256_mul_ps(yy,x2)),_mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(two,xy),x),y));
            const auto denominator=_mm256_add_ps(x2,y2),active=_mm256_cmp_ps(denominator,zero,_CMP_NEQ_OQ);
            const auto result=_mm256_blendv_ps(zero,_mm256_div_ps(numerator,_mm256_blendv_ps(one,denominator,active)),active);
            if(_mm256_movemask_ps(_mm256_cmp_ps(_mm256_andnot_ps(sign,result),limit,_CMP_LE_OQ))!=255){valid=false;return i;}
            _mm256_storeu_ps(static_cast<float*>(output)+i,result);
        }
    }else{
        const auto zero=_mm256_setzero_pd(),one=_mm256_set1_pd(1.),two=_mm256_set1_pd(2.),sign=_mm256_set1_pd(-0.),limit=_mm256_set1_pd(std::numeric_limits<double>::max());
        for(;i+4<=count;i+=4){
            const auto y=_mm256_loadu_pd(static_cast<const double*>(fields[0])+i),x=_mm256_loadu_pd(static_cast<const double*>(fields[1])+i);
            const auto xy=_mm256_loadu_pd(static_cast<const double*>(fields[2])+i),xx=_mm256_loadu_pd(static_cast<const double*>(fields[3])+i),yy=_mm256_loadu_pd(static_cast<const double*>(fields[4])+i);
            const auto y2=_mm256_mul_pd(y,y),x2=_mm256_mul_pd(x,x);
            const auto numerator=_mm256_sub_pd(_mm256_add_pd(_mm256_mul_pd(xx,y2),_mm256_mul_pd(yy,x2)),_mm256_mul_pd(_mm256_mul_pd(_mm256_mul_pd(two,xy),x),y));
            const auto denominator=_mm256_add_pd(x2,y2),active=_mm256_cmp_pd(denominator,zero,_CMP_NEQ_OQ);
            const auto result=_mm256_blendv_pd(zero,_mm256_div_pd(numerator,_mm256_blendv_pd(one,denominator,active)),active);
            if(_mm256_movemask_pd(_mm256_cmp_pd(_mm256_andnot_pd(sign,result),limit,_CMP_LE_OQ))!=15){valid=false;return i;}
            _mm256_storeu_pd(static_cast<double*>(output)+i,result);
        }
    }
    return i;
}
}
