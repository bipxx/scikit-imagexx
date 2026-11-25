#include "skimagexx/frangi.hpp"
#include "skimagexx/symmetric_eigen.hpp"
#include "skimagexx/image.hpp"
#include "execution.hpp"
#include "frangi_detail.hpp"
#include "frangi_exp.hpp"
#include <atomic>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#ifdef _MSC_VER
#include <intrin.h>
#endif
#ifdef SKIMAGEXX_AVX2
namespace skimagexx::simd {
std::size_t frangi_prepare(const void*,void*,void*,void*,std::size_t,std::size_t,
 std::size_t,std::size_t,bool,const FrangiParameters*,double&,bool&);
std::size_t frangi_response(const void*,const void*,const void*,const void*,void*,std::size_t,
 bool,bool,const FrangiParameters&,bool,bool&);
}
#endif
namespace skimagexx::detail {
std::unique_ptr<void,void(*)(void*)> acquire_ridge_scratch(std::size_t,bool);

bool frangi_fma_available(){
#ifdef SKIMAGEXX_AVX2
 if(!has_avx2())return false;
#ifdef _MSC_VER
 static const bool supported=[]{int regs[4];__cpuidex(regs,1,0);return (regs[2]&(1<<12))!=0;}();
 return supported;
#elif defined(__GNUC__) || defined(__clang__)
 return bool(__builtin_cpu_supports("fma"));
#else
 return false;
#endif
#else
 return false;
#endif
}
void frangi_exp_values_into(const float* input,float* output,std::size_t count){
 std::size_t i=0;
#ifdef SKIMAGEXX_AVX2
 if(frangi_fma_available())for(;i+4<=count;i+=4)simd::frangi_exp_float4(input+i,output+i);
#endif
 for(;i<count;++i)output[i]=frangi_exp(input[i]);
}
}
namespace skimagexx { namespace {
template<class T> T exp_argument(T value,double denominator,bool narrow){
 const T negative=-T(value*value);
 return narrow ? T(negative/T(denominator)) : T(double(negative)/denominator);
}
template<class T> bool valid_denominator(double denominator,bool narrow){
 if(!(denominator>0) || !std::isfinite(denominator))return false;
 if constexpr(sizeof(T)==4){
  if(narrow)return std::isfinite(T(denominator)) && T(denominator)!=0;
 }
 return true;
}
template<class T,std::size_t n> bool run(const void* const* fields,const void* previous,void* destination,std::size_t count,FrangiParameters& parameters){
 auto workspace=detail::acquire_ridge_scratch((n+3)*count,sizeof(T)==4);
 auto* eigen=static_cast<T*>(workspace.get());auto* ra=eigen+n*count;auto* rb=ra+count;auto* norm=rb+count;auto* output=static_cast<T*>(destination);const auto* old=static_cast<const T*>(previous);
 const bool eigen_valid=parameters.eigen_provider && parameters.eigen_provider->evaluate
     ? parameters.eigen_provider->evaluate(fields,eigen,count,n,sizeof(T)==4,parameters.eigen_provider->context)
     : symmetric_eigenvalues_into(fields,eigen,count,n,sizeof(T)==4);
 if(!eigen_valid)return false;
 const auto blocks=(count+3)/4;const auto workers=detail::active_workers(count*16);
 const bool automatic_gamma=parameters.automatic_gamma;
 const auto denominators_valid=[&]{
  return valid_denominator<T>(parameters.alpha_den,parameters.alpha_single)
      && valid_denominator<T>(parameters.beta_den,parameters.beta_single)
      && valid_denominator<T>(parameters.gamma_den,parameters.gamma_single);
 };
 if(!automatic_gamma && !denominators_valid())return false;
 std::atomic<bool> valid{true};std::atomic<double> maximum{0.};
 detail::partition(blocks,workers,[&](auto first,auto last,auto){
  auto begin=first*4;const auto end=std::min(count,last*4);double local_max=0.;bool supported=true;
#ifdef SKIMAGEXX_AVX2
  if(has_avx2())begin+=simd::frangi_prepare(eigen,ra,rb,norm,count,n,begin,end,sizeof(T)==4,
   automatic_gamma ? nullptr : &parameters,local_max,supported);
#endif
  if(!supported){valid.store(false,std::memory_order_relaxed);return;}
  for(;begin<end;++begin){T values[n];for(std::size_t d=0;d<n;++d)values[d]=eigen[d*count+begin];
   const auto order=[&](std::size_t a,std::size_t b){if(std::abs(values[b])<std::abs(values[a]))std::swap(values[a],values[b]);};
   order(0,1);
   if constexpr(n==3){order(1,2);order(0,1);}
   const T second=std::max(values[1],T(1e-10));T sum=0;for(std::size_t d=0;d<n;++d)sum=T(sum+T(values[d]*values[d]));
   norm[begin]=std::sqrt(sum);
   if constexpr(n==2){ra[begin]=T(0);rb[begin]=T(std::abs(values[0])/second);}
   else{const T third=std::max(values[2],T(1e-10));ra[begin]=T(second/third);rb[begin]=T(std::abs(values[0])/std::sqrt(T(second*third)));}
   if(!automatic_gamma){
    if constexpr(n==3)ra[begin]=exp_argument(ra[begin],parameters.alpha_den,parameters.alpha_single);
    rb[begin]=exp_argument(rb[begin],parameters.beta_den,parameters.beta_single);
    norm[begin]=exp_argument(norm[begin],parameters.gamma_den,parameters.gamma_single);
   }
   if(!std::isfinite(norm[begin]) || !std::isfinite(ra[begin]) || !std::isfinite(rb[begin])){valid.store(false,std::memory_order_relaxed);return;}
   if(automatic_gamma)local_max=std::max(local_max,double(norm[begin]));
  }
  if(automatic_gamma){auto current=maximum.load(std::memory_order_relaxed);while(current<local_max && !maximum.compare_exchange_weak(current,local_max,std::memory_order_relaxed)){}}
 });
 if(!valid.load(std::memory_order_relaxed))return false;
 if(automatic_gamma){
  double gamma=maximum.load(std::memory_order_relaxed)/2.;if(!gamma)gamma=1.;
  parameters.gamma_den=parameters.gamma_compute_single ? double(T(T(2)*T(T(gamma)*T(gamma)))) : 2*gamma*gamma;
  parameters.automatic_gamma=false;
  if(!denominators_valid())return false;
 }
 bool supplied_exp=false;
 if constexpr(sizeof(T)==8)supplied_exp=parameters.exp_provider && parameters.exp_provider->evaluate;
 else supplied_exp=parameters.float_exp_provider && parameters.float_exp_provider->evaluate;
 if(supplied_exp){
  // Whole fields preserve the provider's SIMD/tail dispatch independently of
  // worker partitions. Reuse the now-dead eigenvalue buffer for exp outputs.
  T* inputs[]={ra,rb,norm};const double denominators[]={parameters.alpha_den,parameters.beta_den,parameters.gamma_den};
  const bool narrows[]={parameters.alpha_single,parameters.beta_single,parameters.gamma_single};
  const std::size_t first_factor=n==2 ? 1 : 0;
  for(std::size_t f=first_factor;f<3;++f){
   if(automatic_gamma){
    detail::partition(blocks,workers,[&](auto first,auto last,auto){
     auto begin=first*4;const auto end=std::min(count,last*4);bool supported=true;
#ifdef SKIMAGEXX_AVX2
     if(has_avx2()){
      if constexpr(sizeof(T)==8)begin+=simd::frangi_exp_arguments(inputs[f]+begin,end-begin,denominators[f],supported);
      else begin+=simd::frangi_exp_arguments(inputs[f]+begin,end-begin,denominators[f],narrows[f],supported);
     }
#endif
     for(;supported && begin<end;++begin){
      const T x=exp_argument(inputs[f][begin],denominators[f],narrows[f]);
      if(!std::isfinite(x)){supported=false;break;}
      inputs[f][begin]=x;
     }
     if(!supported)valid.store(false,std::memory_order_relaxed);
    });
    if(!valid.load(std::memory_order_relaxed))return false;
   }
   if constexpr(sizeof(T)==8)parameters.exp_provider->evaluate(inputs[f],eigen+(f-first_factor)*count,count,parameters.exp_provider->context);
   else parameters.float_exp_provider->evaluate(inputs[f],eigen+(f-first_factor)*count,count,parameters.float_exp_provider->context);
  }
  const T* plate=n==2 ? nullptr : eigen;const auto* blob=eigen+(n==2 ? 0 : count);const auto* structure=blob+count;
  detail::partition(blocks,workers,[&](auto first,auto last,auto){
   auto begin=first*4;const auto end=std::min(count,last*4);bool supported=true;
#ifdef SKIMAGEXX_AVX2
   if(has_avx2())begin+=simd::frangi_combine(plate ? plate+begin : nullptr,blob+begin,structure+begin,old ? old+begin : nullptr,output+begin,end-begin,supported);
#endif
   for(;supported && begin<end;++begin){T value=T((plate ? T(T(1)-plate[begin]) : T(1))*blob[begin]);value=T(value*T(T(1)-structure[begin]));const T prior=old ? old[begin] : T(0);
    if(!std::isfinite(value) || !std::isfinite(prior)){supported=false;break;}output[begin]=value>=prior ? value : prior;}
   if(!supported)valid.store(false,std::memory_order_relaxed);
  });return valid.load(std::memory_order_relaxed);
 }
 detail::partition(blocks,workers,[&](auto first,auto last,auto){
  auto begin=first*4;const auto end=std::min(count,last*4);bool supported=true;
#ifdef SKIMAGEXX_AVX2
  if(has_avx2())begin+=simd::frangi_response(ra+begin,rb+begin,norm+begin,old ? old+begin : nullptr,output+begin,end-begin,sizeof(T)==4,n==2,parameters,!automatic_gamma,supported);
#endif
  if(!supported){valid.store(false,std::memory_order_relaxed);return;}
  const auto factor=[&](T value,double denominator,bool narrow){
   const T exponent=automatic_gamma ? exp_argument(value,denominator,narrow) : value;
   if(!std::isfinite(exponent)){supported=false;return T(0);}
   return detail::frangi_exp(exponent);
  };
  for(;begin<end;++begin){
   const T plate=n==2 ? T(1) : T(T(1)-factor(ra[begin],parameters.alpha_den,parameters.alpha_single));
   T value=T(plate*factor(rb[begin],parameters.beta_den,parameters.beta_single));
   value=T(value*T(T(1)-factor(norm[begin],parameters.gamma_den,parameters.gamma_single)));
   const T prior=old ? old[begin] : T(0);
   if(!supported || !std::isfinite(value) || !std::isfinite(prior)){valid.store(false,std::memory_order_relaxed);return;}
   output[begin]=value>=prior ? value : prior;
  }
 });return valid.load(std::memory_order_relaxed);
}
}}
namespace skimagexx {
bool frangi_accumulate_into(const void* const* fields,const void* previous,void* output,std::size_t count,std::size_t n,bool single,FrangiParameters& parameters){
 if(n<2 || n>3)throw std::invalid_argument("Frangi dimension must be 2 or3");
 if(count>std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/(3*sizeof(double)))throw std::overflow_error("Frangi dimensions overflow");
 if(!count)return true;
 if(!fields || !output)throw std::invalid_argument("nonnull Frangi fields/output required");
 for(std::size_t f=0;f<n*(n+1)/2;++f)if(!fields[f])throw std::invalid_argument("nonnull Frangi component required");
 if(n==2)return single ? run<float,2>(fields,previous,output,count,parameters) : run<double,2>(fields,previous,output,count,parameters);
 return single ? run<float,3>(fields,previous,output,count,parameters) : run<double,3>(fields,previous,output,count,parameters);
}
void ridge_hessian_finish_into(void* output,std::size_t count,bool single){
 if(count && !output)throw std::invalid_argument("nonnull ridge Hessian output required");
 detail::partition(count,detail::active_workers(count),[&](auto begin,auto end,auto){
#ifdef SKIMAGEXX_AVX2
  if(has_avx2())begin+=simd::ridge_hessian_finish(single ? static_cast<void*>(static_cast<float*>(output)+begin) : static_cast<void*>(static_cast<double*>(output)+begin),end-begin,single);
#endif
  if(single){auto* p=static_cast<float*>(output);for(;begin<end;++begin)if(p[begin]<=0)p[begin]=1;}
  else{auto* p=static_cast<double*>(output);for(;begin<end;++begin)if(p[begin]<=0)p[begin]=1;}
 });
}
}
