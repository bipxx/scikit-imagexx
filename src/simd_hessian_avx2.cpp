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
template<class T> __m256d load(const T* p){
 if constexpr(std::is_same_v<T,float>)return _mm256_cvtps_pd(_mm_loadu_ps(p));
 else return _mm256_loadu_pd(p);
}
template<class T> void store(T* p,__m256d sum){
 if constexpr(std::is_same_v<T,float>)_mm_storeu_ps(p,_mm256_cvtpd_ps(sum));
 else _mm256_storeu_pd(p,sum);
}
struct SamplePair { __m256d left,right; };

template<bool Derivative> __m256d accumulate(__m256d sum,SamplePair pair,__m256d weight){
 const auto value=Derivative ? _mm256_sub_pd(pair.left,pair.right) : _mm256_add_pd(pair.left,pair.right);
 return _mm256_add_pd(sum,_mm256_mul_pd(value,weight));
}

template<bool Derivative,bool Dual,class Input,class Output,class Samples>
std::size_t convolve(const Input* center,Output* out,Output* derivative_out,std::size_t count,
 const HessianKernel& kernel,const Samples& samples){
 const auto radius=kernel.radius;
 const auto middle=_mm256_set1_pd(kernel.weights[radius]);
 __m256d derivative_middle{};
 if constexpr(Dual)derivative_middle=_mm256_set1_pd(kernel.derivative_weights[radius]);
 std::size_t i=0;
 for(;i+16<=count;i+=16){
  const auto center0=load(center+i),center1=load(center+i+4);
  const auto center2=load(center+i+8),center3=load(center+i+12);
  auto sum0=_mm256_mul_pd(center0,middle),sum1=_mm256_mul_pd(center1,middle);
  auto sum2=_mm256_mul_pd(center2,middle),sum3=_mm256_mul_pd(center3,middle);
  __m256d d0{},d1{},d2{},d3{};
  if constexpr(Dual){
   d0=_mm256_mul_pd(center0,derivative_middle);d1=_mm256_mul_pd(center1,derivative_middle);
   d2=_mm256_mul_pd(center2,derivative_middle);d3=_mm256_mul_pd(center3,derivative_middle);
  }
  std::size_t j=0;
  if constexpr(Dual){
   // One filter may have more exact-zero outer coefficients than the other.
   const auto first=kernel.smooth_start+kernel.derivative_start;
   for(;j<first;++j){
    const bool smooth=kernel.smooth_start==0;
    const auto weight=_mm256_set1_pd(smooth ? kernel.weights[j] : kernel.derivative_weights[j]);
    if(smooth){
     sum0=accumulate<false>(sum0,samples(i,j),weight);
     sum1=accumulate<false>(sum1,samples(i+4,j),weight);
     sum2=accumulate<false>(sum2,samples(i+8,j),weight);
     sum3=accumulate<false>(sum3,samples(i+12,j),weight);
    }else{
     d0=accumulate<true>(d0,samples(i,j),weight);
     d1=accumulate<true>(d1,samples(i+4,j),weight);
     d2=accumulate<true>(d2,samples(i+8,j),weight);
     d3=accumulate<true>(d3,samples(i+12,j),weight);
    }
   }
  }
  for(;j<radius;++j){
   const auto weight=_mm256_set1_pd(kernel.weights[j]);
   __m256d derivative_weight{};
   if constexpr(Dual)derivative_weight=_mm256_set1_pd(kernel.derivative_weights[j]);
   auto step=[&](std::size_t lane,__m256d& sum,__m256d& derivative_sum){
    const auto pair=samples(i+lane,j);
    sum=accumulate<Derivative>(sum,pair,weight);
    if constexpr(Dual)derivative_sum=accumulate<true>(derivative_sum,pair,derivative_weight);
   };
   step(0,sum0,d0);step(4,sum1,d1);step(8,sum2,d2);step(12,sum3,d3);
  }
  store(out+i,sum0);store(out+i+4,sum1);store(out+i+8,sum2);store(out+i+12,sum3);
  if constexpr(Dual){
   store(derivative_out+i,d0);store(derivative_out+i+4,d1);
   store(derivative_out+i+8,d2);store(derivative_out+i+12,d3);
  }
 }
 for(;i+4<=count;i+=4){
  const auto value=load(center+i);
  auto sum=_mm256_mul_pd(value,middle);
  __m256d derivative_sum{};
  if constexpr(Dual)derivative_sum=_mm256_mul_pd(value,derivative_middle);
  for(std::size_t j=0;j<radius;++j){
   const auto pair=samples(i,j);
   if(!Dual || j>=kernel.smooth_start)sum=accumulate<Derivative>(sum,pair,_mm256_set1_pd(kernel.weights[j]));
   if constexpr(Dual)if(j>=kernel.derivative_start)
    derivative_sum=accumulate<true>(derivative_sum,pair,_mm256_set1_pd(kernel.derivative_weights[j]));
  }
  store(out+i,sum);
  if constexpr(Dual)store(derivative_out+i,derivative_sum);
 }
 return i;
}

template<class Input,class Output,class Samples>
std::size_t dispatch(const Input* center,Output* output,Output* derivative_output,
 std::size_t count,const HessianKernel& kernel,const Samples& samples){
 if(derivative_output)return convolve<false,true>(center,output,derivative_output,count,kernel,samples);
 if(kernel.derivative)return convolve<true,false>(center,output,derivative_output,count,kernel,samples);
 return convolve<false,false>(center,output,derivative_output,count,kernel,samples);
}

template<class Input,class Output>
std::size_t direct(const void* center,void* output,void* derivative_output,std::size_t count,
 std::size_t stride,const HessianKernel& kernel){
 const auto* origin=static_cast<const Input*>(center);
 const auto* left=origin-kernel.radius*stride;
 const auto* right=origin+kernel.radius*stride;
 const auto samples=[=](std::size_t i,std::size_t j){
  return SamplePair{load(left+i+j*stride),load(right+i-j*stride)};
 };
 return dispatch(origin,static_cast<Output*>(output),static_cast<Output*>(derivative_output),count,kernel,samples);
}

template<class T,bool Constant>
std::size_t indexed(const void* base,const void* center,void* output,void* derivative_output,
 std::size_t count,const HessianTap* taps,const HessianKernel& kernel,double cval){
 const auto* origin=static_cast<const T*>(base);
 const auto constant=_mm256_set1_pd(cval);
 const auto samples=[=](std::size_t i,std::size_t j){
  const auto [left,right]=taps[j];
  if constexpr(Constant)return SamplePair{left<0 ? constant : load(origin+left+i),right<0 ? constant : load(origin+right+i)};
  else return SamplePair{load(origin+left+i),load(origin+right+i)};
 };
 return dispatch(static_cast<const T*>(center),static_cast<T*>(output),static_cast<T*>(derivative_output),count,kernel,samples);
}
}

std::size_t hessian_convolve_direct(const void* center,void* output,void* derivative_output,
 std::size_t count,std::size_t stride,const HessianKernel& kernel,bool input_single,bool output_single){
 if(input_single)return direct<float,float>(center,output,derivative_output,count,stride,kernel);
 if(output_single)return direct<double,float>(center,output,derivative_output,count,stride,kernel);
 return direct<double,double>(center,output,derivative_output,count,stride,kernel);
}

std::size_t hessian_convolve_indexed(const void* base,const void* center,void* output,
 void* derivative_output,std::size_t count,const HessianTap* taps,const HessianKernel& kernel,
 double cval,bool constant,bool single){
 if(single){
  if(constant)return indexed<float,true>(base,center,output,derivative_output,count,taps,kernel,cval);
  return indexed<float,false>(base,center,output,derivative_output,count,taps,kernel,cval);
 }
 if(constant)return indexed<double,true>(base,center,output,derivative_output,count,taps,kernel,cval);
 return indexed<double,false>(base,center,output,derivative_output,count,taps,kernel,cval);
}
}
