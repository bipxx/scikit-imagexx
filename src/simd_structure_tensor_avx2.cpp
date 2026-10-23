#include "structure_tensor_detail.hpp"
#include <immintrin.h>
#include <cmath>
#include <limits>
#include <type_traits>

namespace skimagexx::simd { namespace {
template<class T> __m256d load(const void* input,std::size_t i,double constant){
    if(!input)return _mm256_set1_pd(constant);
    if constexpr(std::is_same_v<T,float>)return _mm256_cvtps_pd(_mm_loadu_ps(static_cast<const float*>(input)+i));
    else return _mm256_loadu_pd(static_cast<const double*>(input)+i);
}
template<class T,bool Derivative> std::size_t pass(const void* left,const void* center,const void* right,void* output,std::size_t count,double cval){
    const auto middle=_mm256_set1_pd(Derivative ? 0. : 2.),negative=_mm256_set1_pd(-1.);
    std::size_t i=0;
    for(;i+4<=count;i+=4){
        const auto a=load<T>(left,i,cval),b=load<T>(right,i,cval);
        auto value=_mm256_mul_pd(load<T>(center,i,cval),middle);
        if constexpr(Derivative)value=_mm256_add_pd(value,_mm256_mul_pd(_mm256_sub_pd(a,b),negative));
        else value=_mm256_add_pd(value,_mm256_add_pd(a,b));
        if constexpr(std::is_same_v<T,float>)_mm_storeu_ps(static_cast<float*>(output)+i,_mm256_cvtpd_ps(value));
        else _mm256_storeu_pd(static_cast<double*>(output)+i,value);
    }
    return i;
}
}
std::size_t structure_validate(const void* input,std::size_t count,bool single,double bound,bool& valid){
    std::size_t i=0;
    if(single){
        const auto rounded=float(bound);
        const auto limit=_mm256_set1_ps(double(rounded)>bound ? std::nextafter(rounded,-std::numeric_limits<float>::infinity()) : rounded),sign=_mm256_set1_ps(-0.f);
        for(;i+8<=count;i+=8)if(_mm256_movemask_ps(_mm256_cmp_ps(_mm256_andnot_ps(sign,_mm256_loadu_ps(static_cast<const float*>(input)+i)),limit,_CMP_LE_OQ))!=255){valid=false;return i;}
    }else{const auto limit=_mm256_set1_pd(bound),sign=_mm256_set1_pd(-0.);
        for(;i+4<=count;i+=4)if(_mm256_movemask_pd(_mm256_cmp_pd(_mm256_andnot_pd(sign,_mm256_loadu_pd(static_cast<const double*>(input)+i)),limit,_CMP_LE_OQ))!=15){valid=false;return i;}
    }
    return i;
}
std::size_t structure_pass(const void* left,const void* center,const void* right,void* output,std::size_t count,bool single,bool derivative,double cval){
    if(single)return derivative ? pass<float,true>(left,center,right,output,count,cval) : pass<float,false>(left,center,right,output,count,cval);
    return derivative ? pass<double,true>(left,center,right,output,count,cval) : pass<double,false>(left,center,right,output,count,cval);
}
std::size_t structure_product(const void* first,const void* second,void* output,std::size_t count,bool single){
    std::size_t i=0;
    if(single)for(;i+8<=count;i+=8)_mm256_storeu_ps(static_cast<float*>(output)+i,_mm256_mul_ps(_mm256_loadu_ps(static_cast<const float*>(first)+i),_mm256_loadu_ps(static_cast<const float*>(second)+i)));
    else for(;i+4<=count;i+=4)_mm256_storeu_pd(static_cast<double*>(output)+i,_mm256_mul_pd(_mm256_loadu_pd(static_cast<const double*>(first)+i),_mm256_loadu_pd(static_cast<const double*>(second)+i)));
    return i;
}
}
