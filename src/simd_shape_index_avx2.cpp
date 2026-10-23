#include "shape_index_detail.hpp"
#include "distance_svml.hpp"
#include <immintrin.h>
namespace skimagexx::simd { namespace {
struct Vector {
 using R=__m256d;
 static R c(double x){return _mm256_set1_pd(x);}
 static R mul(R x,R y){return _mm256_mul_pd(x,y);}
 // The existing BSD-licensed SVML polynomial remains accurate without FMA;
 // use explicit operations so this path requires AVX2 only.
 static R fma(R x,R y,R z){return _mm256_add_pd(_mm256_mul_pd(x,y),z);}
};
__m256d angle(__m256d ratio){
 const auto sign=_mm256_set1_pd(-0.),zero=_mm256_setzero_pd(),one=_mm256_set1_pd(1.);
 const auto absolute=_mm256_andnot_pd(sign,ratio),large=_mm256_cmp_pd(absolute,one,_CMP_GT_OQ);
 // Keep inactive inverse lanes away from zero to avoid needless exceptions.
 const auto inverse=_mm256_div_pd(_mm256_set1_pd(-1.),_mm256_blendv_pd(one,absolute,large));
 const auto r=_mm256_blendv_pd(absolute,inverse,large);
 auto result=detail::svml_atan_polynomial<Vector>(r);
 result=_mm256_add_pd(result,_mm256_blendv_pd(zero,_mm256_set1_pd(0x1.921fb54442d18p+0),large));
 return _mm256_or_pd(_mm256_andnot_pd(sign,result),_mm256_and_pd(sign,ratio));
}
}
std::size_t shape_index_response(const void* upper,const void* lower,void* output,std::size_t count,bool single,bool& valid){
 std::size_t i=0;
 if(single){const auto zero=_mm256_setzero_ps(),maximum=_mm256_set1_ps(3.4028234663852886e38f),sign=_mm256_set1_ps(-0.f),factor=_mm256_set1_ps(float(2/0x1.921fb54442d18p+1));
  for(;i+8<=count;i+=8){const auto a=_mm256_loadu_ps(static_cast<const float*>(upper)+i),b=_mm256_loadu_ps(static_cast<const float*>(lower)+i);
   const auto denominator=_mm256_sub_ps(b,a),ratio=_mm256_div_ps(_mm256_add_ps(b,a),denominator);
   const auto overflow=_mm256_and_ps(_mm256_cmp_ps(_mm256_andnot_ps(sign,ratio),maximum,_CMP_GT_OQ),_mm256_cmp_ps(denominator,zero,_CMP_NEQ_OQ));
   if(_mm256_movemask_ps(overflow)){valid=false;return i;}
   const auto first=_mm256_cvtpd_ps(angle(_mm256_cvtps_pd(_mm256_castps256_ps128(ratio))));
   const auto second=_mm256_cvtpd_ps(angle(_mm256_cvtps_pd(_mm256_extractf128_ps(ratio,1))));
   const auto atan=_mm256_insertf128_ps(_mm256_castps128_ps256(first),second,1);
   _mm256_storeu_ps(static_cast<float*>(output)+i,_mm256_mul_ps(factor,atan));
  }
 }else{const auto zero=_mm256_setzero_pd(),maximum=_mm256_set1_pd(1.7976931348623157e308),sign=_mm256_set1_pd(-0.),factor=_mm256_set1_pd(2/0x1.921fb54442d18p+1);
  for(;i+4<=count;i+=4){const auto a=_mm256_loadu_pd(static_cast<const double*>(upper)+i),b=_mm256_loadu_pd(static_cast<const double*>(lower)+i);
   const auto denominator=_mm256_sub_pd(b,a),ratio=_mm256_div_pd(_mm256_add_pd(b,a),denominator);
   const auto overflow=_mm256_and_pd(_mm256_cmp_pd(_mm256_andnot_pd(sign,ratio),maximum,_CMP_GT_OQ),_mm256_cmp_pd(denominator,zero,_CMP_NEQ_OQ));
   if(_mm256_movemask_pd(overflow)){valid=false;return i;}
   _mm256_storeu_pd(static_cast<double*>(output)+i,_mm256_mul_pd(factor,angle(ratio)));
  }
 }return i;
}
}
