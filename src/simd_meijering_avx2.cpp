#include "meijering_detail.hpp"
#include <immintrin.h>
#include <algorithm>
#include <limits>
namespace skimagexx::simd {
std::size_t meijering_response(const void* values,void* output,std::size_t count,std::size_t n,std::size_t begin,std::size_t end,bool single,double alpha,double& maximum,bool& valid){
 const auto zero=_mm256_setzero_pd(),sign=_mm256_set1_pd(-0.);auto i=begin;
 for(;i+4<=end;i+=4){auto selected=zero,largest=_mm256_set1_pd(-1.);
  for(std::size_t r=0;r<n;++r){__m256d projected;
   if(single){auto sum=_mm_setzero_ps();for(std::size_t c=0;c<n;++c)sum=_mm_add_ps(sum,_mm_mul_ps(_mm_set1_ps(r==c ? 1.f : float(alpha)),_mm_loadu_ps(static_cast<const float*>(values)+c*count+i)));projected=_mm256_cvtps_pd(sum);}
   else{projected=zero;for(std::size_t c=0;c<n;++c)projected=_mm256_add_pd(projected,_mm256_mul_pd(_mm256_set1_pd(r==c ? 1. : alpha),_mm256_loadu_pd(static_cast<const double*>(values)+c*count+i)));}
   const auto magnitude=_mm256_andnot_pd(sign,projected);
   if(_mm256_movemask_pd(_mm256_cmp_pd(magnitude,_mm256_set1_pd(std::numeric_limits<double>::max()),_CMP_LE_OQ))!=15){valid=false;return i-begin;}
   const auto greater=_mm256_cmp_pd(magnitude,largest,_CMP_GT_OQ);selected=_mm256_blendv_pd(selected,projected,greater);largest=_mm256_max_pd(largest,magnitude);
  }
  const auto clipped=_mm256_max_pd(selected,zero);alignas(32) double lanes[4];_mm256_store_pd(lanes,clipped);for(auto v:lanes)maximum=std::max(maximum,v);
  if(single)_mm_storeu_ps(static_cast<float*>(output)+i,_mm256_cvtpd_ps(clipped));else _mm256_storeu_pd(static_cast<double*>(output)+i,clipped);
 }return i-begin;
}
std::size_t meijering_normalize(void* output,const void* previous,std::size_t count,bool single,double divisor){
 std::size_t i=0;
 if(single){auto* p=static_cast<float*>(output);const auto d=_mm256_set1_ps(float(divisor));for(;i+8<=count;i+=8){auto value=_mm256_loadu_ps(p+i);if(divisor>0)value=_mm256_div_ps(value,d);const auto old=previous ? _mm256_loadu_ps(static_cast<const float*>(previous)+i) : _mm256_setzero_ps();_mm256_storeu_ps(p+i,_mm256_max_ps(old,value));}}
 else{auto* p=static_cast<double*>(output);const auto d=_mm256_set1_pd(divisor);for(;i+4<=count;i+=4){auto value=_mm256_loadu_pd(p+i);if(divisor>0)value=_mm256_div_pd(value,d);const auto old=previous ? _mm256_loadu_pd(static_cast<const double*>(previous)+i) : _mm256_setzero_pd();_mm256_storeu_pd(p+i,_mm256_max_pd(old,value));}}
 return i;
}
}
