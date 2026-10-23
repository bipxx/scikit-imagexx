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
namespace skimagexx { namespace {
template<class T> T factor(T value,double denominator,bool narrow,bool& valid){
 const T square=T(value*value),negative=-square;
 const T exponent=narrow ? T(negative/T(denominator)) : T(double(negative)/denominator);
 if(!std::isfinite(exponent)){valid=false;return T(0);}
 return detail::frangi_exp(exponent);
}
template<class T> bool run(const void* const* fields,const void* previous,void* destination,std::size_t count,std::size_t n,FrangiParameters& parameters){
 std::unique_ptr<T[]> eigen(new T[n*count]),work(new T[3*count]);auto* ra=work.get();auto* rb=ra+count;auto* norm=rb+count;auto* output=static_cast<T*>(destination);const auto* old=static_cast<const T*>(previous);
 const bool eigen_valid=parameters.eigen_provider && parameters.eigen_provider->evaluate
     ? parameters.eigen_provider->evaluate(fields,eigen.get(),count,n,sizeof(T)==4,parameters.eigen_provider->context)
     : symmetric_eigenvalues_into(fields,eigen.get(),count,n,sizeof(T)==4);
 if(!eigen_valid)return false;
 const auto blocks=(count+3)/4;const auto workers=detail::active_workers(count*16);
 std::atomic<bool> valid{true};std::atomic<double> maximum{0.};
 detail::partition(blocks,workers,[&](auto first,auto last,auto){
  auto begin=first*4;const auto end=std::min(count,last*4);double local_max=0.;bool supported=true;
#ifdef SKIMAGEXX_AVX2
  if(has_avx2())begin+=simd::frangi_prepare(eigen.get(),ra,rb,norm,count,n,begin,end,sizeof(T)==4,local_max,supported);
#endif
  if(!supported){valid.store(false,std::memory_order_relaxed);return;}
  for(;begin<end;++begin){T values[3];for(std::size_t d=0;d<n;++d)values[d]=eigen[d*count+begin];
   for(std::size_t d=1;d<n;++d)for(auto j=d;j>0 && std::abs(values[j])<std::abs(values[j-1]);--j)std::swap(values[j],values[j-1]);
   const T second=std::max(values[1],T(1e-10));T sum=0;for(std::size_t d=0;d<n;++d)sum=T(sum+T(values[d]*values[d]));
   norm[begin]=std::sqrt(sum);
   if(n==2){ra[begin]=T(0);rb[begin]=T(std::abs(values[0])/second);}
   else{const T third=std::max(values[2],T(1e-10));ra[begin]=T(second/third);rb[begin]=T(std::abs(values[0])/std::sqrt(T(second*third)));}
   if(!std::isfinite(norm[begin]) || !std::isfinite(ra[begin]) || !std::isfinite(rb[begin])){valid.store(false,std::memory_order_relaxed);return;}
   local_max=std::max(local_max,double(norm[begin]));
  }
  auto current=maximum.load(std::memory_order_relaxed);while(current<local_max && !maximum.compare_exchange_weak(current,local_max,std::memory_order_relaxed)){}
 });
 if(!valid.load(std::memory_order_relaxed))return false;
 if(parameters.automatic_gamma){
  double gamma=maximum.load(std::memory_order_relaxed)/2.;if(!gamma)gamma=1.;
  parameters.gamma_den=parameters.gamma_compute_single ? double(T(T(2)*T(T(gamma)*T(gamma)))) : 2*gamma*gamma;
  parameters.automatic_gamma=false;
 }
 for(const auto den:{parameters.alpha_den,parameters.beta_den,parameters.gamma_den})if(!(den>0) || !std::isfinite(den))return false;
 if(sizeof(T)==4 && ((parameters.alpha_single && (!std::isfinite(T(parameters.alpha_den)) || T(parameters.alpha_den)==0)) || (parameters.beta_single && (!std::isfinite(T(parameters.beta_den)) || T(parameters.beta_den)==0)) || (parameters.gamma_single && (!std::isfinite(T(parameters.gamma_den)) || T(parameters.gamma_den)==0))))return false;
 if constexpr(sizeof(T)==8){if(parameters.exp_provider && parameters.exp_provider->evaluate){
  // Whole fields preserve the provider's SIMD/tail dispatch independently of
  // worker partitions. Reuse the now-dead eigenvalue buffer for exp outputs.
  double* inputs[]={ra,rb,norm};const double denominators[]={parameters.alpha_den,parameters.beta_den,parameters.gamma_den};
  const std::size_t first_factor=n==2 ? 1 : 0;
  for(std::size_t f=first_factor;f<3;++f){
   detail::partition(blocks,workers,[&](auto first,auto last,auto){
    auto begin=first*4;const auto end=std::min(count,last*4);bool supported=true;
#ifdef SKIMAGEXX_AVX2
    if(has_avx2())begin+=simd::frangi_exp_arguments(inputs[f]+begin,end-begin,denominators[f],supported);
#endif
    for(;supported && begin<end;++begin){const double v=inputs[f][begin];const double x=-(v*v)/denominators[f];if(!std::isfinite(x)){supported=false;break;}inputs[f][begin]=x;}
    if(!supported)valid.store(false,std::memory_order_relaxed);
   });
   if(!valid.load(std::memory_order_relaxed))return false;
   parameters.exp_provider->evaluate(inputs[f],eigen.get()+(f-first_factor)*count,count,parameters.exp_provider->context);
  }
  const double* plate=n==2 ? nullptr : eigen.get();const auto* blob=eigen.get()+(n==2 ? 0 : count);const auto* structure=blob+count;
  detail::partition(blocks,workers,[&](auto first,auto last,auto){
   auto begin=first*4;const auto end=std::min(count,last*4);bool supported=true;
#ifdef SKIMAGEXX_AVX2
   if(has_avx2())begin+=simd::frangi_combine(plate ? plate+begin : nullptr,blob+begin,structure+begin,old ? old+begin : nullptr,output+begin,end-begin,supported);
#endif
   for(;supported && begin<end;++begin){const double value=(plate ? 1-plate[begin] : 1)*blob[begin]*(1-structure[begin]);const double prior=old ? old[begin] : 0.;
    if(!std::isfinite(value) || !std::isfinite(prior)){supported=false;break;}output[begin]=value>=prior ? value : prior;}
   if(!supported)valid.store(false,std::memory_order_relaxed);
  });return valid.load(std::memory_order_relaxed);
 }}
 detail::partition(blocks,workers,[&](auto first,auto last,auto){
  auto begin=first*4;const auto end=std::min(count,last*4);bool supported=true;
#ifdef SKIMAGEXX_AVX2
  if(has_avx2())begin+=simd::frangi_response(ra+begin,rb+begin,norm+begin,old ? old+begin : nullptr,output+begin,end-begin,sizeof(T)==4,n==2,parameters,supported);
#endif
  if(!supported){valid.store(false,std::memory_order_relaxed);return;}
  for(;begin<end;++begin){
   const T plate=n==2 ? T(1) : T(T(1)-factor(ra[begin],parameters.alpha_den,parameters.alpha_single,supported));
   T value=T(plate*factor(rb[begin],parameters.beta_den,parameters.beta_single,supported));
   value=T(value*T(T(1)-factor(norm[begin],parameters.gamma_den,parameters.gamma_single,supported)));
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
 if(!count)return true;if(!fields || !output)throw std::invalid_argument("nonnull Frangi fields/output required");
 for(std::size_t f=0;f<n*(n+1)/2;++f)if(!fields[f])throw std::invalid_argument("nonnull Frangi component required");
 return single ? run<float>(fields,previous,output,count,n,parameters) : run<double>(fields,previous,output,count,n,parameters);
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
