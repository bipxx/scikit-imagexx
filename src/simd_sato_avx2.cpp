#include "sato_detail.hpp"
#include <immintrin.h>
#include <limits>
namespace skimagexx::simd {
std::size_t sato_negate(void* data,std::size_t count,bool single){
 std::size_t i=0;
 if(single){const auto sign=_mm256_set1_ps(-0.f);auto* p=static_cast<float*>(data);for(;i+8<=count;i+=8)_mm256_storeu_ps(p+i,_mm256_xor_ps(_mm256_loadu_ps(p+i),sign));}
 else{const auto sign=_mm256_set1_pd(-0.);auto* p=static_cast<double*>(data);for(;i+4<=count;i+=4)_mm256_storeu_pd(p+i,_mm256_xor_pd(_mm256_loadu_pd(p+i),sign));}
 return i;
}
std::size_t sato_response(const void* first,const void* second,const void* previous,void* output,std::size_t count,
 bool single,bool previous_single,bool output_single,bool value_single,double factor,bool& valid){
 const auto zero=_mm256_setzero_pd(),sign=_mm256_set1_pd(-0.),weight=_mm256_set1_pd(factor);
 const auto maximum=_mm256_set1_pd(output_single ? double(std::numeric_limits<float>::max()) : std::numeric_limits<double>::max());
 std::size_t i=0;
 for(;i+4<=count;i+=4){__m256d value;
  if(single){auto root=_mm_max_ps(_mm_loadu_ps(static_cast<const float*>(first)+i),_mm_setzero_ps());
   if(second)root=_mm_sqrt_ps(_mm_mul_ps(root,_mm_max_ps(_mm_loadu_ps(static_cast<const float*>(second)+i),_mm_setzero_ps())));
   value=value_single ? _mm256_cvtps_pd(_mm_mul_ps(root,_mm_set1_ps(float(factor)))) : _mm256_mul_pd(_mm256_cvtps_pd(root),weight);
  }else{auto root=_mm256_max_pd(_mm256_loadu_pd(static_cast<const double*>(first)+i),zero);
   if(second)root=_mm256_sqrt_pd(_mm256_mul_pd(root,_mm256_max_pd(_mm256_loadu_pd(static_cast<const double*>(second)+i),zero)));
   value=_mm256_mul_pd(root,weight);
  }
  const auto prior=!previous ? zero : previous_single ? _mm256_cvtps_pd(_mm_loadu_ps(static_cast<const float*>(previous)+i)) : _mm256_loadu_pd(static_cast<const double*>(previous)+i);
  const auto safe=_mm256_and_pd(_mm256_cmp_pd(_mm256_andnot_pd(sign,value),maximum,_CMP_LE_OQ),_mm256_cmp_pd(_mm256_andnot_pd(sign,prior),maximum,_CMP_LE_OQ));
  if(_mm256_movemask_pd(safe)!=15){valid=false;return i;}
  const auto result=_mm256_max_pd(prior,value);
  if(output_single)_mm_storeu_ps(static_cast<float*>(output)+i,_mm256_cvtpd_ps(result));else _mm256_storeu_pd(static_cast<double*>(output)+i,result);
 }return i;
}
}
