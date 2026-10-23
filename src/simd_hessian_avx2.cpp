#include "hessian_detail.hpp"
#include <immintrin.h>
#include <type_traits>
namespace skimagexx::simd {
std::size_t hessian_gradient(const void* left,const void* right,void* out,std::size_t count,bool single,bool edge){
 std::size_t i=0;
 if(single){const auto scale=_mm256_set1_ps(edge ? 1.f : .5f);
  for(;i+8<=count;i+=8)_mm256_storeu_ps(static_cast<float*>(out)+i,_mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(static_cast<const float*>(right)+i),_mm256_loadu_ps(static_cast<const float*>(left)+i)),scale));
 }else{const auto scale=_mm256_set1_pd(edge ? 1. : .5);
  for(;i+4<=count;i+=4)_mm256_storeu_pd(static_cast<double*>(out)+i,_mm256_mul_pd(_mm256_sub_pd(_mm256_loadu_pd(static_cast<const double*>(right)+i),_mm256_loadu_pd(static_cast<const double*>(left)+i)),scale));
 }return i;
}
namespace {
template<class T> __m256d load(const void* p,std::size_t i,double c){
 if(!p)return _mm256_set1_pd(c);
 if constexpr(std::is_same_v<T,float>)return _mm256_cvtps_pd(_mm_loadu_ps(static_cast<const float*>(p)+i));
 else return _mm256_loadu_pd(static_cast<const double*>(p)+i);
}
template<class Input,class Output> std::size_t convolve(const void* const* left,const void* const* right,const void* center,void* out,std::size_t count,std::size_t radius,const double* weights,bool derivative,double cval){
 std::size_t i=0;const auto middle=_mm256_set1_pd(weights[radius]);
 for(;i+4<=count;i+=4){
  auto sum=_mm256_mul_pd(load<Input>(center,i,cval),middle);
  for(std::size_t j=0;j<radius;++j){const auto a=load<Input>(left[j],i,cval),b=load<Input>(right[j],i,cval);
   const auto pair=derivative ? _mm256_sub_pd(a,b) : _mm256_add_pd(a,b);
   sum=_mm256_add_pd(sum,_mm256_mul_pd(pair,_mm256_set1_pd(weights[j])));
  }
  if constexpr(std::is_same_v<Output,float>)_mm_storeu_ps(static_cast<float*>(out)+i,_mm256_cvtpd_ps(sum));
  else _mm256_storeu_pd(static_cast<double*>(out)+i,sum);
 }return i;
}
}
std::size_t hessian_convolve(const void* const* a,const void* const* b,const void* c,void* out,std::size_t n,std::size_t r,const double* w,bool d,double cv,bool input_single,bool output_single){
 if(input_single)return output_single ? convolve<float,float>(a,b,c,out,n,r,w,d,cv) : convolve<float,double>(a,b,c,out,n,r,w,d,cv);
 return output_single ? convolve<double,float>(a,b,c,out,n,r,w,d,cv) : convolve<double,double>(a,b,c,out,n,r,w,d,cv);
}
}
