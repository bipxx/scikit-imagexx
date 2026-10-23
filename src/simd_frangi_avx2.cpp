#include "frangi_detail.hpp"
#include "frangi_exp.hpp"
#include <immintrin.h>
#include <algorithm>
#include <cmath>
namespace skimagexx::simd {namespace {
template<bool Single> struct V;
template<> struct V<true>{using T=float;using R=__m128;
 static R load(const float* p){return _mm_loadu_ps(p);}static void store(float* p,R v){_mm_storeu_ps(p,v);}static R c(double v){return _mm_set1_ps(float(v));}
 static R add(R a,R b){return _mm_add_ps(a,b);}static R sub(R a,R b){return _mm_sub_ps(a,b);}static R mul(R a,R b){return _mm_mul_ps(a,b);}static R div(R a,R b){return _mm_div_ps(a,b);}static R root(R a){return _mm_sqrt_ps(a);}static R abs(R a){return _mm_andnot_ps(c(-0.),a);}static R gt(R a,R b){return _mm_cmpgt_ps(a,b);}static R blend(R a,R b,R m){return _mm_blendv_ps(a,b,m);}static R max(R a,R b){return _mm_max_ps(a,b);}
};
template<> struct V<false>{using T=double;using R=__m256d;
 static R load(const double* p){return _mm256_loadu_pd(p);}static void store(double* p,R v){_mm256_storeu_pd(p,v);}static R c(double v){return _mm256_set1_pd(v);}
 static R add(R a,R b){return _mm256_add_pd(a,b);}static R sub(R a,R b){return _mm256_sub_pd(a,b);}static R mul(R a,R b){return _mm256_mul_pd(a,b);}static R div(R a,R b){return _mm256_div_pd(a,b);}static R root(R a){return _mm256_sqrt_pd(a);}static R abs(R a){return _mm256_andnot_pd(c(-0.),a);}static R gt(R a,R b){return _mm256_cmp_pd(a,b,_CMP_GT_OQ);}static R blend(R a,R b,R m){return _mm256_blendv_pd(a,b,m);}static R max(R a,R b){return _mm256_max_pd(a,b);}
};
template<bool Single> std::size_t prepare(const void* source,void* a,void* b,void* s,std::size_t count,std::size_t n,std::size_t begin,std::size_t end,double& maximum,bool& valid){
 using M=V<Single>;using T=typename M::T;using R=typename M::R;auto i=begin;const auto* input=static_cast<const T*>(source);
 for(;i+4<=end;i+=4){R v[3];for(std::size_t d=0;d<n;++d)v[d]=M::load(input+d*count+i);
  auto swap=[&](int x,int y){const auto mask=M::gt(M::abs(v[x]),M::abs(v[y]));const auto left=v[x],right=v[y];v[x]=M::blend(left,right,mask);v[y]=M::blend(right,left,mask);};
  swap(0,1);if(n==3){swap(1,2);swap(0,1);}
  auto sum=M::c(0);for(std::size_t d=0;d<n;++d)sum=M::add(sum,M::mul(v[d],v[d]));const auto norm=M::root(sum),second=M::max(v[1],M::c(1e-10));
  const auto ra=n==2 ? M::c(0) : M::div(second,M::max(v[2],M::c(1e-10)));
  const auto denominator=n==2 ? second : M::root(M::mul(second,M::max(v[2],M::c(1e-10))));const auto rb=M::div(M::abs(v[0]),denominator);
  M::store(static_cast<T*>(a)+i,ra);M::store(static_cast<T*>(b)+i,rb);M::store(static_cast<T*>(s)+i,norm);
  alignas(32) T norms[4],as[4],bs[4];M::store(norms,norm);M::store(as,ra);M::store(bs,rb);
  for(int lane=0;lane<4;++lane){if(!std::isfinite(norms[lane]) || !std::isfinite(as[lane]) || !std::isfinite(bs[lane])){valid=false;return i-begin;}maximum=std::max(maximum,double(norms[lane]));}
 }return i-begin;
}
template<bool Single> std::size_t response(const void* a,const void* b,const void* s,const void* previous,void* output,std::size_t count,bool pair,const FrangiParameters& p,bool& valid){
 using M=V<Single>;using T=typename M::T;using R=typename M::R;std::size_t i=0;
 for(;i+4<=count;i+=4){
  auto exponential=[&](const void* input,double den,bool narrow){auto value=M::load(static_cast<const T*>(input)+i);auto numerator=M::sub(M::c(-0.),M::mul(value,value));
   alignas(32) T lanes[4];
   if constexpr(Single){if(!narrow){const auto wide=_mm256_div_pd(_mm256_cvtps_pd(numerator),_mm256_set1_pd(den));numerator=_mm256_cvtpd_ps(wide);}else numerator=M::div(numerator,M::c(den));}
   else numerator=M::div(numerator,M::c(den));
   M::store(lanes,numerator);for(auto& lane:lanes)if(!std::isfinite(lane)){valid=false;lane=T(0);}
   for(auto& lane:lanes)lane=detail::frangi_exp(lane);return M::load(lanes);
  };
  const R plate=pair ? M::c(1) : M::sub(M::c(1),exponential(a,p.alpha_den,p.alpha_single));
  auto value=M::mul(plate,exponential(b,p.beta_den,p.beta_single));value=M::mul(value,M::sub(M::c(1),exponential(s,p.gamma_den,p.gamma_single)));
  const auto prior=previous ? M::load(static_cast<const T*>(previous)+i) : M::c(0);alignas(32) T old[4];M::store(old,prior);for(auto lane:old)if(!std::isfinite(lane))valid=false;
  if(!valid)return i;M::store(static_cast<T*>(output)+i,M::max(prior,value));
 }return i;
}
}}
namespace skimagexx::simd {
std::size_t frangi_exp_arguments(double* values,std::size_t count,double denominator,bool& valid){
 std::size_t i=0;for(;i+4<=count;i+=4){const auto v=_mm256_loadu_pd(values+i);
  const auto x=_mm256_div_pd(_mm256_sub_pd(_mm256_set1_pd(-0.),_mm256_mul_pd(v,v)),_mm256_set1_pd(denominator));
  alignas(32) double lanes[4];_mm256_store_pd(lanes,x);for(auto lane:lanes)if(!std::isfinite(lane)){valid=false;return i;}
  _mm256_storeu_pd(values+i,x);
 }return i;
}
std::size_t frangi_combine(const double* plate,const double* blob,const double* structure,const double* previous,double* output,std::size_t count,bool& valid){
 std::size_t i=0;for(;i+4<=count;i+=4){const auto one=_mm256_set1_pd(1.);
  const auto a=plate ? _mm256_sub_pd(one,_mm256_loadu_pd(plate+i)) : one;
  const auto v=_mm256_mul_pd(_mm256_mul_pd(a,_mm256_loadu_pd(blob+i)),_mm256_sub_pd(one,_mm256_loadu_pd(structure+i)));
  const auto old=previous ? _mm256_loadu_pd(previous+i) : _mm256_setzero_pd();
  alignas(32) double lanes[4],priors[4];_mm256_store_pd(lanes,v);_mm256_store_pd(priors,old);
  for(int lane=0;lane<4;++lane)if(!std::isfinite(lanes[lane]) || !std::isfinite(priors[lane])){valid=false;return i;}
  _mm256_storeu_pd(output+i,_mm256_max_pd(old,v));
 }return i;
}
std::size_t frangi_prepare(const void* e,void* a,void* b,void* s,std::size_t count,std::size_t n,std::size_t begin,std::size_t end,bool single,double& maximum,bool& valid){return single ? prepare<true>(e,a,b,s,count,n,begin,end,maximum,valid) : prepare<false>(e,a,b,s,count,n,begin,end,maximum,valid);}
std::size_t frangi_response(const void* a,const void* b,const void* s,const void* previous,void* output,std::size_t count,bool single,bool pair,const FrangiParameters& p,bool& valid){return single ? response<true>(a,b,s,previous,output,count,pair,p,valid) : response<false>(a,b,s,previous,output,count,pair,p,valid);}
std::size_t ridge_hessian_finish(void* output,std::size_t count,bool single){std::size_t i=0;
 if(single){auto* p=static_cast<float*>(output);for(;i+8<=count;i+=8){const auto v=_mm256_loadu_ps(p+i);_mm256_storeu_ps(p+i,_mm256_blendv_ps(v,_mm256_set1_ps(1),_mm256_cmp_ps(v,_mm256_setzero_ps(),_CMP_LE_OQ)));}}
 else{auto* p=static_cast<double*>(output);for(;i+4<=count;i+=4){const auto v=_mm256_loadu_pd(p+i);_mm256_storeu_pd(p+i,_mm256_blendv_pd(v,_mm256_set1_pd(1),_mm256_cmp_pd(v,_mm256_setzero_pd(),_CMP_LE_OQ)));}}
 return i;
}
}
