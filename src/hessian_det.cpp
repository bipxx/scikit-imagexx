#include "skimagexx/hessian_det.hpp"
#include "skimagexx/gaussian_nd.hpp"
#include "skimagexx/integral.hpp"
#include "skimagexx/image.hpp"
#include "hessian_det_detail.hpp"
#include "execution.hpp"
#include <atomic>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
namespace skimagexx { namespace {
template<class T> bool fields(const void* const* input,T* output,std::size_t count,std::size_t n){
 std::atomic<bool> valid{true};
 detail::partition((count+3)/4,detail::active_workers(count*n*n*n),[&](auto first,auto last,auto){
  auto begin=first*4;const auto end=std::min(count,last*4);bool supported=true;
#ifdef SKIMAGEXX_AVX2
  if(has_avx2())begin+=simd::hessian_det_fields(input,output,begin,end,n,sizeof(T)==4,supported);
#endif
  if(!supported){valid.store(false,std::memory_order_relaxed);return;}
  for(auto i=begin;i<end;++i){
   double a[25],sign=1.,logdet=0.;std::size_t field=0;bool singular=false;
   for(std::size_t r=0;r<n;++r)for(std::size_t c=r;c<n;++c){const auto v=double(static_cast<const T*>(input[field++])[i]);
    if(!std::isfinite(v)){valid.store(false,std::memory_order_relaxed);return;}a[r*n+c]=a[c*n+r]=v;}
   for(std::size_t k=0;k<n;++k){
    auto pivot=k;for(auto r=k+1;r<n;++r)if(std::abs(a[r*n+k])>std::abs(a[pivot*n+k]))pivot=r;
    if(a[pivot*n+k]==0.){singular=true;break;}
    if(pivot!=k){for(std::size_t c=0;c<n;++c)std::swap(a[k*n+c],a[pivot*n+c]);sign=-sign;}
    const auto diagonal=a[k*n+k];if(diagonal<0.)sign=-sign;logdet+=std::log(std::abs(diagonal));
    for(auto r=k+1;r<n;++r){
     const auto factor=std::abs(diagonal)>=std::numeric_limits<double>::min() ? a[r*n+k]*(1./diagonal) : a[r*n+k]/diagonal;
     for(auto c=k+1;c<n;++c)a[r*n+c]=a[r*n+c]-factor*a[k*n+c];
    }
   }
   const auto result=singular ? 0. : sign*std::exp(logdet);const T value=T(result);
   if(!std::isfinite(value)){valid.store(false,std::memory_order_relaxed);return;}output[i]=value;
  }
 });return valid.load(std::memory_order_relaxed);
}
}}
namespace skimagexx {
bool hessian_det_approx_into(const void* input,PixelType type,double* output,std::size_t height,std::size_t width,std::int64_t size){
 if(size < -3000 || size>3000)throw std::invalid_argument("approximate Hessian size outside supported range");
 const auto limit=std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/sizeof(double);
 if(height>limit || width>limit || (height && width>limit/height))throw std::overflow_error("approximate Hessian shape overflow");
 const auto count=height*width;if(!count)return true;
 if(!input || !output)throw std::invalid_argument("nonnull approximate Hessian buffers required");
 std::unique_ptr<double[]> integral(new double[count]);const std::vector<std::size_t> shape{height,width};
 if(type==PixelType::float32 || type==PixelType::float64)integral_image_into(input,integral.get(),type,PixelType::float64,shape);
 else{gaussian_nd_into(input,output,type,PixelType::float64,shape,{},false);integral_image_into(output,integral.get(),PixelType::float64,PixelType::float64,shape);}
 for(std::size_t i=0;i<count;++i)if(!std::isfinite(integral[i]) || std::abs(integral[i])>1e140)return false;
 if(!size){std::fill(output,output+count,std::numeric_limits<double>::quiet_NaN());return true;}
 detail::DetBox boxes[8];detail::det_boxes(std::ptrdiff_t(size),boxes);const double weight=1./double(size)/double(size);
 // An interior SIMD span uses contiguous loads; clipped edge coordinates are
 // evaluated separately, preserving the upstream start-before-end clipping.
 std::ptrdiff_t left=0,right=std::ptrdiff_t(width);
 for(const auto& b:boxes){left=std::max({left,-b.column,-b.column-b.columns});right=std::min({right,std::ptrdiff_t(width)-b.column,std::ptrdiff_t(width)-b.column-b.columns});}
 detail::partition(height,detail::active_workers(count*16),[&](auto begin,auto end,auto){
  for(auto r=begin;r<end;++r){std::size_t c=0;const auto first=std::size_t(std::clamp(left,std::ptrdiff_t(0),std::ptrdiff_t(width)));
   auto scalar=[&](std::size_t column){double values[8];for(int j=0;j<8;++j)values[j]=detail::det_box_value(integral.get(),height,width,r,column,boxes[j]);output[r*width+column]=detail::det_box_response(values,weight);};
   for(;c<first;++c)scalar(c);
#ifdef SKIMAGEXX_AVX2
   if(has_avx2() && right>std::ptrdiff_t(c))c+=simd::hessian_det_boxes(integral.get(),output,height,width,r,c,std::size_t(right),boxes,weight);
#endif
   for(;c<width;++c)scalar(c);
  }
 });return true;
}
bool hessian_det_from_fields_into(const void* const* input,void* output,std::size_t count,std::size_t n,bool single){
 if(n<2 || n>5)throw std::invalid_argument("determinant dimensions must be 2..5");
 if(count>std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/(n*n*sizeof(double)))throw std::overflow_error("determinant shape overflow");
 if(!count)return true;if(!input || !output)throw std::invalid_argument("nonnull determinant buffers required");
 for(std::size_t f=0;f<n*(n+1)/2;++f)if(!input[f])throw std::invalid_argument("nonnull determinant field required");
 return single ? fields(input,static_cast<float*>(output),count,n) : fields(input,static_cast<double*>(output),count,n);
}
}
