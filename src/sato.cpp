#include "skimagexx/sato.hpp"
#include "skimagexx/gaussian_nd.hpp"
#include "skimagexx/symmetric_eigen.hpp"
#include "skimagexx/image.hpp"
#include "execution.hpp"
#include "sato_detail.hpp"
#include <atomic>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
namespace skimagexx { namespace {
template<class T> void negate(T* data,std::size_t count){
 detail::partition(count,detail::active_workers(count),[&](auto begin,auto end,auto){
#ifdef SKIMAGEXX_AVX2
  if(has_avx2())begin+=simd::sato_negate(data+begin,end-begin,sizeof(T)==4);
#endif
  for(;begin<end;++begin)data[begin]=-data[begin];
 });
}
template<class T> bool response(const void* const* fields,const void* previous,void* output,std::size_t count,std::size_t n,
 bool previous_single,bool output_single,bool value_single,double factor){
 std::unique_ptr<T[]> eigen(new T[n*count]);
 if(!symmetric_eigenvalues_into(fields,eigen.get(),count,n,sizeof(T)==4))return false;
 std::atomic<bool> valid{true};
 detail::partition((count+3)/4,detail::active_workers(count*8),[&](auto first,auto last,auto){
  auto begin=first*4;const auto end=std::min(count,last*4);bool supported=true;
#ifdef SKIMAGEXX_AVX2
  if(has_avx2())begin+=simd::sato_response(eigen.get()+begin,n==3 ? eigen.get()+count+begin : nullptr,
     previous ? (previous_single ? static_cast<const void*>(static_cast<const float*>(previous)+begin) : static_cast<const void*>(static_cast<const double*>(previous)+begin)) : nullptr,
     output_single ? static_cast<void*>(static_cast<float*>(output)+begin) : static_cast<void*>(static_cast<double*>(output)+begin),
     end-begin,sizeof(T)==4,previous_single,output_single,value_single,factor,supported);
#endif
  if(!supported){valid.store(false,std::memory_order_relaxed);return;}
  for(;begin<end;++begin){
   T value=std::max(T(0),eigen[begin]);
   if(n==3)value=std::sqrt(T(value*std::max(T(0),eigen[count+begin])));
   const double scaled=value_single ? double(float(value)*float(factor)) : double(value)*factor;
   const double prior=previous ? (previous_single ? double(static_cast<const float*>(previous)[begin]) : static_cast<const double*>(previous)[begin]) : 0.;
   const double result=scaled>=prior ? scaled : prior;
   if(!std::isfinite(scaled) || !std::isfinite(prior) || (output_single && std::abs(result)>std::numeric_limits<float>::max())){valid.store(false,std::memory_order_relaxed);return;}
   if(output_single)static_cast<float*>(output)[begin]=float(result);else static_cast<double*>(output)[begin]=result;
  }
 });return valid.load(std::memory_order_relaxed);
}
}}
namespace skimagexx {
void ridge_prepare_into(const void* source,PixelType type,void* output,std::size_t count,bool single,bool negative){
 gaussian_nd_into(source,output,type,single ? PixelType::float32 : PixelType::float64,{count},{},true);
 if(negative){if(single)negate(static_cast<float*>(output),count);else negate(static_cast<double*>(output),count);}
}
bool sato_accumulate_into(const void* const* fields,const void* previous,void* output,std::size_t count,std::size_t n,
 bool single,bool previous_single,bool output_single,bool value_single,double factor){
 if(n<2 || n>3 || (!single && value_single) || (output_single && !value_single) || (previous && !previous_single && output_single))throw std::invalid_argument("invalid Sato dimensions or precision");
 if(count>std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/(n*sizeof(double)))throw std::overflow_error("Sato dimensions overflow");
 if(!count)return true;if(!fields || !output)throw std::invalid_argument("nonnull Sato fields/output required");
 for(std::size_t f=0;f<n*(n+1)/2;++f)if(!fields[f])throw std::invalid_argument("nonnull Sato component required");
 if(!std::isfinite(factor))return false;
 return single ? response<float>(fields,previous,output,count,n,previous_single,output_single,value_single,factor)
               : response<double>(fields,previous,output,count,n,previous_single,output_single,value_single,factor);
}
}
