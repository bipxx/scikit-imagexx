#include "skimagexx/meijering.hpp"
#include "skimagexx/symmetric_eigen.hpp"
#include "skimagexx/image.hpp"
#include "execution.hpp"
#include "meijering_detail.hpp"
#include "structure_tensor_detail.hpp"
#include <atomic>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
namespace skimagexx { namespace {
template<class T> bool run(const void* const* fields,const void* previous,void* destination,std::size_t count,std::size_t n,double alpha){
 std::unique_ptr<T[]> eigen(new T[n*count]);auto* output=static_cast<T*>(destination);const auto* old=static_cast<const T*>(previous);
 if(old){std::size_t checked=0;bool valid=true;
#ifdef SKIMAGEXX_AVX2
  if(has_avx2())checked=simd::structure_validate(old,count,sizeof(T)==4,double(std::numeric_limits<T>::max()),valid);
#endif
  if(!valid)return false;for(;checked<count;++checked)if(!std::isfinite(old[checked]))return false;
 }
 if(!symmetric_eigenvalues_into(fields,eigen.get(),count,n,sizeof(T)==4))return false;
 const T coefficient=T(alpha);if(!std::isfinite(coefficient))return false;
 const auto blocks=(count+3)/4;const auto workers=std::min(detail::active_workers(count*n*n),int(std::min(blocks,std::size_t(std::numeric_limits<int>::max()))));
 std::atomic<bool> valid{true};std::atomic<double> maximum{0.};
 detail::partition(blocks,workers,[&](auto first,auto last,auto){
  auto begin=first*4;const auto end=std::min(count,last*4);double local_max=0.;bool supported=true;
#ifdef SKIMAGEXX_AVX2
  if(has_avx2())begin+=simd::meijering_response(eigen.get(),output,count,n,begin,end,sizeof(T)==4,alpha,local_max,supported);
#endif
  if(!supported){valid.store(false,std::memory_order_relaxed);return;}
  for(;begin<end;++begin){T selected=0,largest=-1;
   for(std::size_t r=0;r<n;++r){T projected=0;
    for(std::size_t c=0;c<n;++c)projected=T(projected+T((r==c ? T(1) : coefficient)*eigen[c*count+begin]));
    if(!std::isfinite(projected)){valid.store(false,std::memory_order_relaxed);return;}
    const auto magnitude=std::abs(projected);if(magnitude>largest){selected=projected;largest=magnitude;}
   }
   output[begin]=std::max(selected,T(0));local_max=std::max(local_max,double(output[begin]));
  }
  auto current=maximum.load(std::memory_order_relaxed);
  while(current<local_max && !maximum.compare_exchange_weak(current,local_max,std::memory_order_relaxed)){}
 });
 if(!valid.load(std::memory_order_relaxed))return false;
 const T divisor=T(maximum.load(std::memory_order_relaxed));
 detail::partition(blocks,workers,[&](auto first,auto last,auto){
  auto begin=first*4;const auto end=std::min(count,last*4);
#ifdef SKIMAGEXX_AVX2
  if(has_avx2())begin+=simd::meijering_normalize(output+begin,old ? old+begin : nullptr,end-begin,sizeof(T)==4,divisor);
#endif
  for(;begin<end;++begin){const T normalized=divisor>T(0) ? T(output[begin]/divisor) : output[begin];const T prior=old ? old[begin] : T(0);output[begin]=normalized>=prior ? normalized : prior;}
 });return true;
}
}}
namespace skimagexx {
bool meijering_accumulate_into(const void* const* fields,const void* previous,void* output,std::size_t count,std::size_t n,bool single,double alpha){
 if(n<1 || n>5)throw std::invalid_argument("Meijering dimension must be 1..5");
 if(count>std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/(n*n*sizeof(double)))throw std::overflow_error("Meijering dimensions overflow");
 if(!count)return true;if(!fields || !output)throw std::invalid_argument("nonnull Meijering fields/output required");
 for(std::size_t f=0;f<n*(n+1)/2;++f)if(!fields[f])throw std::invalid_argument("nonnull Meijering component required");
 if(!std::isfinite(alpha))return false;
 return single ? run<float>(fields,previous,output,count,n,alpha) : run<double>(fields,previous,output,count,n,alpha);
}
}
