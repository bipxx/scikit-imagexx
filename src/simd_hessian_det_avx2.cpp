#include "hessian_det_detail.hpp"
#include <immintrin.h>
#include <cmath>
#include <limits>
namespace skimagexx::simd {
std::size_t hessian_det_boxes(const double* integral,double* output,std::size_t height,std::size_t width,std::size_t row,std::size_t begin,std::size_t end,const detail::DetBox* boxes,double weight){
 const double* top[8];const double* bottom[8];
 for(int j=0;j<8;++j){const auto r=std::clamp(std::ptrdiff_t(row)+boxes[j].row,std::ptrdiff_t(0),std::ptrdiff_t(height)-1),r2=std::clamp(r+boxes[j].rows,std::ptrdiff_t(0),std::ptrdiff_t(height)-1);top[j]=integral+r*width;bottom[j]=integral+r2*width;}
 const auto zero=_mm256_setzero_pd(),negative=_mm256_set1_pd(-0.),w=_mm256_set1_pd(weight),three=_mm256_set1_pd(3.),correction=_mm256_set1_pd(.81);auto c=begin;
 for(;c+4<=end;c+=4){__m256d b[8];
  for(int j=0;j<8;++j){const auto x=std::ptrdiff_t(c)+boxes[j].column,y=x+boxes[j].columns;
   auto sum=_mm256_add_pd(_mm256_loadu_pd(top[j]+x),_mm256_loadu_pd(bottom[j]+y));
   sum=_mm256_sub_pd(_mm256_sub_pd(sum,_mm256_loadu_pd(top[j]+y)),_mm256_loadu_pd(bottom[j]+x));
   b[j]=_mm256_max_pd(sum,zero);
  }
  const auto xy=_mm256_mul_pd(_mm256_xor_pd(negative,_mm256_sub_pd(_mm256_sub_pd(_mm256_add_pd(b[2],b[3]),b[0]),b[1])),w);
  const auto xx=_mm256_mul_pd(_mm256_xor_pd(negative,_mm256_sub_pd(b[4],_mm256_mul_pd(three,b[5]))),w),yy=_mm256_mul_pd(_mm256_xor_pd(negative,_mm256_sub_pd(b[6],_mm256_mul_pd(three,b[7]))),w);
  _mm256_storeu_pd(output+row*width+c,_mm256_sub_pd(_mm256_mul_pd(xx,yy),_mm256_mul_pd(correction,_mm256_mul_pd(xy,xy))));
 }return c-begin;
}
std::size_t hessian_det_fields(const void* const* fields,void* output,std::size_t begin,std::size_t end,std::size_t n,bool single,bool& valid){
 const auto zero=_mm256_setzero_pd(),one=_mm256_set1_pd(1.),signbit=_mm256_set1_pd(-0.);auto i=begin;
 for(;i+4<=end;i+=4){__m256d a[25];std::size_t field=0;auto sign=one,singular=zero;
  for(std::size_t r=0;r<n;++r)for(std::size_t c=r;c<n;++c){const auto v=single ? _mm256_cvtps_pd(_mm_loadu_ps(static_cast<const float*>(fields[field])+i)) : _mm256_loadu_pd(static_cast<const double*>(fields[field])+i);++field;
   if(_mm256_movemask_pd(_mm256_cmp_pd(_mm256_andnot_pd(signbit,v),_mm256_set1_pd(std::numeric_limits<double>::max()),_CMP_NLE_UQ))){valid=false;return i-begin;}
   a[r*n+c]=a[c*n+r]=v;
  }
  alignas(32) double diagonal[5][4],signs[4],singulars[4];
  for(std::size_t k=0;k<n;++k){
   // Select the first maximal absolute pivot independently in each lane.
   auto maximum=_mm256_andnot_pd(signbit,a[k*n+k]);__m256d pivots[5];
   for(std::size_t r=k+1;r<n;++r){const auto value=_mm256_andnot_pd(signbit,a[r*n+k]);pivots[r]=_mm256_cmp_pd(value,maximum,_CMP_GT_OQ);maximum=_mm256_max_pd(maximum,value);}
   // Later maxima supersede earlier choices. Swapping only the final choice
   // retains LAPACK's single swap and its tie/sign behavior.
   auto chosen=zero;for(std::size_t r=n;r-->k+1;){const auto mask=_mm256_andnot_pd(chosen,pivots[r]);chosen=_mm256_or_pd(chosen,mask);
    for(std::size_t c=0;c<n;++c){const auto x=a[k*n+c],y=a[r*n+c];a[k*n+c]=_mm256_blendv_pd(x,y,mask);a[r*n+c]=_mm256_blendv_pd(y,x,mask);}sign=_mm256_xor_pd(sign,_mm256_and_pd(mask,signbit));}
   const auto d=a[k*n+k];_mm256_store_pd(diagonal[k],d);singular=_mm256_or_pd(singular,_mm256_cmp_pd(d,zero,_CMP_EQ_OQ));
   const auto safe=_mm256_blendv_pd(d,one,singular),reciprocal=_mm256_div_pd(one,safe);
   const auto tiny=_mm256_cmp_pd(_mm256_andnot_pd(signbit,safe),_mm256_set1_pd(std::numeric_limits<double>::min()),_CMP_LT_OQ);
   for(auto r=k+1;r<n;++r){const auto numerator=a[r*n+k];auto factor=_mm256_mul_pd(numerator,reciprocal);
    if(_mm256_movemask_pd(tiny))factor=_mm256_blendv_pd(factor,_mm256_div_pd(numerator,safe),tiny);
    for(auto c=k+1;c<n;++c)a[r*n+c]=_mm256_sub_pd(a[r*n+c],_mm256_mul_pd(factor,a[k*n+c]));
   }
  }
  _mm256_store_pd(signs,sign);_mm256_store_pd(singulars,singular);
  for(int lane=0;lane<4;++lane){double result=0.;
   if(!std::signbit(singulars[lane])){double logdet=0.,s=signs[lane];for(std::size_t k=0;k<n;++k){const auto d=diagonal[k][lane];if(d<0.)s=-s;logdet+=std::log(std::abs(d));}result=s*std::exp(logdet);}
   if(!std::isfinite(result) || (single && std::abs(result)>std::numeric_limits<float>::max())){valid=false;return i-begin;}
   if(single)static_cast<float*>(output)[i+lane]=float(result);else static_cast<double*>(output)[i+lane]=result;
  }
 }return i-begin;
}
}
