#include "frangi_detail.hpp"
#include "frangi_exp.hpp"
#include <immintrin.h>
#include <algorithm>
#include <limits>
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
inline bool finite(__m128 value){
 return _mm_movemask_ps(_mm_cmp_ps(V<true>::abs(value),V<true>::c(std::numeric_limits<float>::max()),_CMP_LE_OQ))==15;
}
inline bool finite(__m256d value){
 return _mm256_movemask_pd(_mm256_cmp_pd(V<false>::abs(value),V<false>::c(std::numeric_limits<double>::max()),_CMP_LE_OQ))==15;
}
inline double horizontal_max(__m128 value){
 const auto halves=_mm_max_ps(value,_mm_movehl_ps(value,value));
 return _mm_cvtss_f32(_mm_max_ss(halves,_mm_shuffle_ps(halves,halves,_MM_SHUFFLE(1,1,1,1))));
}
inline double horizontal_max(__m256d value){
 const auto halves=_mm_max_pd(_mm256_castpd256_pd128(value),_mm256_extractf128_pd(value,1));
 return _mm_cvtsd_f64(_mm_max_sd(halves,_mm_unpackhi_pd(halves,halves)));
}
template<bool Single> typename V<Single>::R exp_argument(typename V<Single>::R value,double denominator,bool narrow){
 using M=V<Single>;
 // Square in field precision before promoting a float numerator for division.
 const auto numerator=M::sub(M::c(-0.),M::mul(value,value));
 if constexpr(Single){
  if(!narrow)return _mm256_cvtpd_ps(_mm256_div_pd(_mm256_cvtps_pd(numerator),_mm256_set1_pd(denominator)));
 }
 return M::div(numerator,M::c(denominator));
}
template<bool Single,std::size_t n> std::size_t prepare(const void* source,void* a,void* b,void* s,std::size_t count,std::size_t begin,std::size_t end,const FrangiParameters* fixed,double& maximum,bool& valid){
 using M=V<Single>;using T=typename M::T;using R=typename M::R;auto i=begin;const auto* input=static_cast<const T*>(source);
 auto max_norm=M::c(0);
 for(;i+4<=end;i+=4){R v[3];for(std::size_t d=0;d<n;++d)v[d]=M::load(input+d*count+i);
  auto swap=[&](int x,int y){const auto mask=M::gt(M::abs(v[x]),M::abs(v[y]));const auto left=v[x],right=v[y];v[x]=M::blend(left,right,mask);v[y]=M::blend(right,left,mask);};
  swap(0,1);if constexpr(n==3){swap(1,2);swap(0,1);}
  auto sum=M::c(0);for(std::size_t d=0;d<n;++d)sum=M::add(sum,M::mul(v[d],v[d]));
  auto norm=M::root(sum);const auto second=M::max(v[1],M::c(1e-10));
  auto ra=M::c(0),denominator=second;
  if constexpr(n==3){
   const auto third=M::max(v[2],M::c(1e-10));
   ra=M::div(second,third);
   denominator=M::root(M::mul(second,third));
  }
  auto rb=M::div(M::abs(v[0]),denominator);
  if(fixed){
   if constexpr(n==3)ra=exp_argument<Single>(ra,fixed->alpha_den,fixed->alpha_single);
   rb=exp_argument<Single>(rb,fixed->beta_den,fixed->beta_single);
   norm=exp_argument<Single>(norm,fixed->gamma_den,fixed->gamma_single);
  }
  if(!finite(norm) || !finite(rb) || (n==3 && !finite(ra))){valid=false;return i-begin;}
  if(!fixed)max_norm=M::max(max_norm,norm);
  M::store(static_cast<T*>(a)+i,ra);M::store(static_cast<T*>(b)+i,rb);M::store(static_cast<T*>(s)+i,norm);
 }
 if(!fixed)maximum=std::max(maximum,horizontal_max(max_norm));
 return i-begin;
}
template<bool Single> std::size_t response(const void* a,const void* b,const void* s,const void* previous,void* output,std::size_t count,bool pair,const FrangiParameters& p,bool prepared_arguments,bool& valid){
 using M=V<Single>;using T=typename M::T;using R=typename M::R;std::size_t i=0;
 const bool use_fma=Single && detail::frangi_fma_available();
 for(;i+4<=count;i+=4){
  auto exponential=[&](const void* input,double den,bool narrow){
   const auto value=M::load(static_cast<const T*>(input)+i);
   const auto exponent=prepared_arguments ? value : exp_argument<Single>(value,den,narrow);
   if(!finite(exponent)){valid=false;return M::c(0);}
   alignas(32) T lanes[4];
   M::store(lanes,exponent);
   if constexpr(Single){if(use_fma){alignas(16) float out[4];frangi_exp_float4(lanes,out);return M::load(out);}}
   for(auto& lane:lanes)lane=detail::frangi_exp(lane);
   return M::load(lanes);
  };
  const R plate=pair ? M::c(1) : M::sub(M::c(1),exponential(a,p.alpha_den,p.alpha_single));
  auto value=M::mul(plate,exponential(b,p.beta_den,p.beta_single));value=M::mul(value,M::sub(M::c(1),exponential(s,p.gamma_den,p.gamma_single)));
  const auto prior=previous ? M::load(static_cast<const T*>(previous)+i) : M::c(0);
  if(!finite(prior))valid=false;
  if(!valid)return i;
  M::store(static_cast<T*>(output)+i,M::max(prior,value));
 }return i;
}
}}
namespace skimagexx::simd {
std::size_t frangi_exp_arguments(float* values,std::size_t count,double denominator,bool narrow,bool& valid){
 std::size_t i=0;for(;i+4<=count;i+=4){const auto v=_mm_loadu_ps(values+i);
  const auto x=exp_argument<true>(v,denominator,narrow);
  if(!finite(x)){valid=false;return i;}
  _mm_storeu_ps(values+i,x);
 }return i;
}
std::size_t frangi_combine(const float* plate,const float* blob,const float* structure,const float* previous,float* output,std::size_t count,bool& valid){
 std::size_t i=0;for(;i+4<=count;i+=4){const auto one=_mm_set1_ps(1.f);
  const auto a=plate ? _mm_sub_ps(one,_mm_loadu_ps(plate+i)) : one;
  const auto v=_mm_mul_ps(_mm_mul_ps(a,_mm_loadu_ps(blob+i)),_mm_sub_ps(one,_mm_loadu_ps(structure+i)));
  const auto old=previous ? _mm_loadu_ps(previous+i) : _mm_setzero_ps();
  if(!finite(v) || !finite(old)){valid=false;return i;}
  _mm_storeu_ps(output+i,_mm_max_ps(old,v));
 }return i;
}
std::size_t frangi_exp_arguments(double* values,std::size_t count,double denominator,bool& valid){
 std::size_t i=0;for(;i+4<=count;i+=4){const auto v=_mm256_loadu_pd(values+i);
  const auto x=exp_argument<false>(v,denominator,false);
  if(!finite(x)){valid=false;return i;}
  _mm256_storeu_pd(values+i,x);
 }return i;
}
std::size_t frangi_combine(const double* plate,const double* blob,const double* structure,const double* previous,double* output,std::size_t count,bool& valid){
 std::size_t i=0;for(;i+4<=count;i+=4){const auto one=_mm256_set1_pd(1.);
  const auto a=plate ? _mm256_sub_pd(one,_mm256_loadu_pd(plate+i)) : one;
  const auto v=_mm256_mul_pd(_mm256_mul_pd(a,_mm256_loadu_pd(blob+i)),_mm256_sub_pd(one,_mm256_loadu_pd(structure+i)));
  const auto old=previous ? _mm256_loadu_pd(previous+i) : _mm256_setzero_pd();
  if(!finite(v) || !finite(old)){valid=false;return i;}
  _mm256_storeu_pd(output+i,_mm256_max_pd(old,v));
 }return i;
}
std::size_t frangi_prepare(const void* e,void* a,void* b,void* s,std::size_t count,std::size_t n,std::size_t begin,std::size_t end,bool single,const FrangiParameters* fixed,double& maximum,bool& valid){
 if(single)return n==2 ? prepare<true,2>(e,a,b,s,count,begin,end,fixed,maximum,valid)
                       : prepare<true,3>(e,a,b,s,count,begin,end,fixed,maximum,valid);
 return n==2 ? prepare<false,2>(e,a,b,s,count,begin,end,fixed,maximum,valid)
             : prepare<false,3>(e,a,b,s,count,begin,end,fixed,maximum,valid);
}
std::size_t frangi_prepare(const void* e,void* a,void* b,void* s,std::size_t count,std::size_t n,std::size_t begin,std::size_t end,bool single,double& maximum,bool& valid){
 return frangi_prepare(e,a,b,s,count,n,begin,end,single,nullptr,maximum,valid);
}
std::size_t frangi_response(const void* a,const void* b,const void* s,const void* previous,void* output,std::size_t count,bool single,bool pair,const FrangiParameters& p,bool prepared_arguments,bool& valid){
 return single ? response<true>(a,b,s,previous,output,count,pair,p,prepared_arguments,valid)
               : response<false>(a,b,s,previous,output,count,pair,p,prepared_arguments,valid);
}
std::size_t frangi_response(const void* a,const void* b,const void* s,const void* previous,void* output,std::size_t count,bool single,bool pair,const FrangiParameters& p,bool& valid){
 return frangi_response(a,b,s,previous,output,count,single,pair,p,false,valid);
}
std::size_t ridge_hessian_finish(void* output,std::size_t count,bool single){std::size_t i=0;
 if(single){auto* p=static_cast<float*>(output);for(;i+8<=count;i+=8){const auto v=_mm256_loadu_ps(p+i);_mm256_storeu_ps(p+i,_mm256_blendv_ps(v,_mm256_set1_ps(1),_mm256_cmp_ps(v,_mm256_setzero_ps(),_CMP_LE_OQ)));}}
 else{auto* p=static_cast<double*>(output);for(;i+4<=count;i+=4){const auto v=_mm256_loadu_pd(p+i);_mm256_storeu_pd(p+i,_mm256_blendv_pd(v,_mm256_set1_pd(1),_mm256_cmp_pd(v,_mm256_setzero_pd(),_CMP_LE_OQ)));}}
 return i;
}
}
