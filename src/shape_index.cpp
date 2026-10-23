#include "skimagexx/shape_index.hpp"
#include "skimagexx/symmetric_eigen.hpp"
#include "skimagexx/image.hpp"
#include "shape_index_detail.hpp"
#include "execution.hpp"
#include <atomic>
#include <cmath>
#include <memory>
#include <limits>
#include <stdexcept>
namespace skimagexx { namespace {
template<class T> bool run(const void* const* fields,void* destination,std::size_t count){
 constexpr bool single=sizeof(T)==4;std::unique_ptr<T[]> eigen(new T[2*count]);
 if(!symmetric_eigenvalues_into(fields,eigen.get(),count,2,single))return false;
 auto* output=static_cast<T*>(destination);const auto* upper=eigen.get();const auto* lower=upper+count;
 std::atomic<bool> valid{true};const T factor=T(2/0x1.921fb54442d18p+1);
 const auto blocks=(count+7)/8;
 detail::partition(blocks,detail::active_workers(count*4),[&](auto block_begin,auto block_end,auto){
  auto begin=block_begin*8;const auto end=std::min(count,block_end*8);
  bool supported=true;
#ifdef SKIMAGEXX_AVX2
  if(has_avx2())begin+=simd::shape_index_response(upper+begin,lower+begin,output+begin,end-begin,single,supported);
#endif
  if(!supported){valid.store(false,std::memory_order_relaxed);return;}
  for(;begin<end;++begin){const T numerator=T(lower[begin]+upper[begin]),denominator=T(lower[begin]-upper[begin]);
   const T ratio=T(numerator/denominator);
   if(std::isinf(ratio) && denominator!=T(0)){valid.store(false,std::memory_order_relaxed);return;}
   output[begin]=T(factor*T(std::atan(ratio)));
  }
 });return valid.load(std::memory_order_relaxed);
}
}}
namespace skimagexx {
bool shape_index_from_hessian_into(const void* const* fields,void* output,std::size_t count,bool single){
 if(count>std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/(2*(single ? sizeof(float) : sizeof(double))))throw std::overflow_error("shape index dimensions overflow");
 if(!count)return true;
 if(!fields || !output)throw std::invalid_argument("nonnull shape index buffers required");
 return single ? run<float>(fields,output,count) : run<double>(fields,output,count);
}
}
