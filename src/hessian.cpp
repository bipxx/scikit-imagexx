#include "skimagexx/hessian.hpp"
#include "skimagexx/image.hpp"
#include "hessian_detail.hpp"
#include "structure_tensor_detail.hpp"
#include "execution.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
namespace skimagexx { namespace {
std::ptrdiff_t mapped(std::ptrdiff_t x,std::size_t length,GaussianBorder mode){
 const auto n=std::ptrdiff_t(length);if(x>=0 && x<n)return x;
 if(mode==GaussianBorder::constant)return -1;
 if(mode==GaussianBorder::nearest)return std::clamp(x,std::ptrdiff_t(0),n-1);
 if(mode==GaussianBorder::wrap){x%=n;return x<0 ? x+n : x;}
 if(n==1)return 0;
 const auto period=mode==GaussianBorder::reflect ? 2*n : 2*(n-1);x%=period;if(x<0)x+=period;
 return x<n ? x : mode==GaussianBorder::reflect ? period-1-x : period-x;
}
template<class T> bool finite(const T* a,std::size_t count){
 std::size_t i=0;bool valid=true;
#ifdef SKIMAGEXX_AVX2
 if(has_avx2())i=simd::structure_validate(a,count,std::is_same_v<T,float>,double(std::numeric_limits<T>::max())/16,valid);
#endif
 if(!valid)return false;
 for(;i<count;++i)if(!std::isfinite(a[i]) || std::abs(double(a[i]))>double(std::numeric_limits<T>::max())/16)return false;
 return true;
}
template<class T> void gradient(const T* input,T* output,std::size_t count,std::size_t length,std::size_t stride,int workers){
 detail::partition(count,workers,[&](auto begin,auto end,auto){
  while(begin<end){const auto coordinate=begin/stride%length,inner=begin%stride;
   auto stop=std::min(end,begin+stride-inner);
   if(stride==1 && coordinate && coordinate+1<length)stop=std::min(end,begin+length-1-coordinate);
   const bool edge=coordinate==0 || coordinate+1==length;
   const auto* left=input+begin-(coordinate ? stride : 0);
   const auto* right=input+begin+(coordinate+1<length ? stride : 0);
   std::size_t done=0;
#ifdef SKIMAGEXX_AVX2
   if(has_avx2())done=simd::hessian_gradient(left,right,output+begin,stop-begin,std::is_same_v<T,float>,edge);
#endif
   for(;done<stop-begin;++done){const T difference=T(right[done]-left[done]);output[begin+done]=edge ? difference : T(difference/T(2));}
   begin=stop;
  }
 });
}
template<class T> void convolve(const T* input,T* output,std::size_t count,const std::vector<std::size_t>& shape,const HessianPass& plan,int workers){
 const auto& p=plan.filter;const auto original_radius=p.weights.size()/2,length=shape[p.axis];
 // ndimage chooses its symmetric arithmetic before antisymmetric arithmetic.
 // Tiny derivative kernels can satisfy both tests within double epsilon.
 bool derivative=plan.derivative;
 if(derivative){derivative=false;for(std::size_t j=0;j<original_radius;++j)
  if(std::abs(p.weights[j]-p.weights[p.weights.size()-1-j])>std::numeric_limits<double>::epsilon()){derivative=true;break;}}

 std::size_t first=0;while(first<original_radius && p.weights[first]==0.)++first;
 const auto radius=original_radius-first;const auto* weights=p.weights.data()+first;std::size_t stride=1;
 for(std::size_t axis=p.axis+1;axis<shape.size();++axis)stride*=shape[axis];
 // Pad a contiguous last-axis line once, retaining double cval precision even
 // for float32 inputs. Every output pixel can then use the SIMD dot product.
 if(stride==1){
  detail::partition(count/length,workers,[&](auto row_begin,auto row_end,auto){
   std::vector<double> padded(length+2*original_radius);
   std::vector<const void*> left(radius),right(radius);
   for(std::size_t j=0;j<radius;++j){left[j]=padded.data()+original_radius-radius+j;right[j]=padded.data()+original_radius+radius-j;}
   for(auto row=row_begin;row<row_end;++row){const auto* from=input+row*length;auto* target=output+row*length;
    if(row>row_begin && std::memcmp(from,from-length,length*sizeof(T))==0){std::copy_n(target-length,length,target);continue;}
    std::copy_n(from,length,padded.data()+original_radius);
    for(std::size_t j=0;j<original_radius;++j){
     const auto lo=mapped(std::ptrdiff_t(j)-std::ptrdiff_t(original_radius),length,p.border),hi=mapped(std::ptrdiff_t(length)+std::ptrdiff_t(j),length,p.border);
     padded[j]=lo<0 ? p.cval : double(from[std::size_t(lo)]);
     padded[original_radius+length+j]=hi<0 ? p.cval : double(from[std::size_t(hi)]);
    }
    std::size_t done=0;
#ifdef SKIMAGEXX_AVX2
    if(has_avx2())done=simd::hessian_convolve(left.data(),right.data(),padded.data()+original_radius,target,length,radius,weights,derivative,p.cval,false,std::is_same_v<T,float>);
#endif
    for(;done<length;++done){double sum=padded[original_radius+done]*weights[radius];
     for(std::size_t j=0;j<radius;++j){const auto a=padded[original_radius+done-radius+j],b=padded[original_radius+done+radius-j];sum+=(derivative ? a-b : a+b)*weights[j];}
     target[done]=T(sum);
    }
    if(first)for(std::size_t x=0;x<length;++x)if(target[x]==T(0) && std::signbit(target[x])){
     double sum=padded[original_radius+x]*p.weights[original_radius];
     for(std::size_t j=0;j<original_radius;++j){const auto a=padded[x+j],b=padded[2*original_radius+x-j];sum+=(derivative ? a-b : a+b)*p.weights[j];}
     target[x]=T(sum);
    }
   }
  });return;
 }
 // Identical inner blocks have identical ordered results. Compute one value
 // per block, including the original zero tails, then copy its exact bits.
 if(stride>1){
  bool repeated=true;const auto blocks=count/stride;
  for(std::size_t block=0;block<blocks;++block)
   if(std::memcmp(input+block*stride,input+block*stride+1,(stride-1)*sizeof(T))){repeated=false;break;}
  if(repeated){
   detail::partition(blocks,workers,[&](auto begin,auto end,auto){
    for(auto position=begin;position<end;++position){const auto x=position%length,base=(position/length)*length;
     double sum=double(input[position*stride])*p.weights[original_radius];
     for(std::size_t j=0;j<original_radius;++j){const auto distance=std::ptrdiff_t(original_radius-j);
      const auto lo=mapped(std::ptrdiff_t(x)-distance,length,p.border),hi=mapped(std::ptrdiff_t(x)+distance,length,p.border);
      const auto a=lo<0 ? p.cval : double(input[(base+std::size_t(lo))*stride]),b=hi<0 ? p.cval : double(input[(base+std::size_t(hi))*stride]);
      sum+=(derivative ? a-b : a+b)*p.weights[j];
     }std::fill_n(output+position*stride,stride,T(sum));
    }
   });return;
  }
 }
 detail::partition(count,workers,[&](auto begin,auto end,auto){
  std::vector<const void*> left(radius),right(radius);
  while(begin<end){const auto coordinate=begin/stride%length,inner=begin%stride,block=begin-coordinate*stride-inner;
   auto stop=std::min(end,begin+stride-inner);
   for(std::size_t j=0;j<radius;++j){const auto distance=std::ptrdiff_t(radius-j);
    const auto lo=mapped(std::ptrdiff_t(coordinate)-distance,length,p.border),hi=mapped(std::ptrdiff_t(coordinate)+distance,length,p.border);
    left[j]=lo<0 ? nullptr : input+block+std::size_t(lo)*stride+inner;
    right[j]=hi<0 ? nullptr : input+block+std::size_t(hi)*stride+inner;
   }
   std::size_t done=0;
#ifdef SKIMAGEXX_AVX2
   if(has_avx2())done=simd::hessian_convolve(left.data(),right.data(),input+begin,output+begin,stop-begin,radius,weights,derivative,p.cval,std::is_same_v<T,float>,std::is_same_v<T,float>);
#endif
   for(;done<stop-begin;++done){double sum=double(input[begin+done])*weights[radius];
    for(std::size_t j=0;j<radius;++j){const auto a=left[j] ? double(static_cast<const T*>(left[j])[done]) : p.cval,b=right[j] ? double(static_cast<const T*>(right[j])[done]) : p.cval;
     sum+=(derivative ? a-b : a+b)*weights[j];}
    output[begin+done]=T(sum);
   }
   // Exactly zero tails cannot change finite nonzero sums. They can change a
   // negative zero into positive zero. Repair only that cold output case with
   // the original ordered pass, preserving ndimage's signed-zero behavior.
   if(first)for(auto position=begin;position<stop;++position)if(output[position]==T(0) && std::signbit(output[position])){
    const auto x=position/stride%length,k=position%stride,base=position-x*stride-k;
    double sum=double(input[position])*p.weights[original_radius];
    for(std::size_t j=0;j<original_radius;++j){const auto distance=std::ptrdiff_t(original_radius-j);
     const auto lo=mapped(std::ptrdiff_t(x)-distance,length,p.border),hi=mapped(std::ptrdiff_t(x)+distance,length,p.border);
     const auto a=lo<0 ? p.cval : double(input[base+std::size_t(lo)*stride+k]),b=hi<0 ? p.cval : double(input[base+std::size_t(hi)*stride+k]);
     sum+=(derivative ? a-b : a+b)*p.weights[j];
    }output[position]=T(sum);
   }
   begin=stop;
  }
 });
}
template<class T> bool execute(const void* input,PixelType source,void* const* output,const std::vector<std::size_t>& shape,const std::vector<std::vector<HessianPass>>& plans,bool gaussian,bool xy,std::size_t count){
 constexpr bool single=std::is_same_v<T,float>;const auto type=single ? PixelType::float32 : PixelType::float64;
 if(gaussian && std::all_of(plans.begin(),plans.end(),[](const auto& passes){return passes.empty();})){
  auto* normalized=static_cast<T*>(output[0]);
  if(source==type)std::copy_n(static_cast<const T*>(input),count,normalized);
  else gaussian_nd_into(input,normalized,source,type,shape,{},false);
  if(!finite(normalized,count))return false;
  const auto fields=shape.size()*(shape.size()+1)/2;
  detail::partition(count,detail::active_workers(count*fields),[&](auto begin,auto end,auto){
   for(std::size_t field=1;field<fields;++field)std::copy(normalized+begin,normalized+end,static_cast<T*>(output[field])+begin);
  });return true;
 }
 std::unique_ptr<T[]> converted,scratch(new T[count]),other(new T[count]);
 const T* prepared;
 if(source==type)prepared=static_cast<const T*>(input);
 else{converted.reset(new T[count]);gaussian_nd_into(input,converted.get(),source,type,shape,{},false);prepared=converted.get();}
 if(!finite(prepared,count))return false;
 const auto n=shape.size();const auto workers=detail::active_workers(count*4);
 auto filter=[&](const T* from,T* to,const std::vector<HessianPass>& passes){
  if(passes.empty()){std::copy(from,from+count,to);return true;}
  const T* current=from;
  for(std::size_t i=0;i<passes.size();++i){T* target=i+1==passes.size() ? to : current==scratch.get() ? other.get() : scratch.get();
   convolve(current,target,count,shape,passes[i],workers);if(!finite(target,count))return false;current=target;
  }return true;
 };
 std::vector<std::unique_ptr<T[]>> derivatives;for(std::size_t i=0;i<n;++i)derivatives.emplace_back(new T[count]);
 if(gaussian){for(std::size_t axis=0;axis<n;++axis)if(!filter(prepared,derivatives[axis].get(),plans[axis]))return false;}
 else{
  std::vector<GaussianPass> smoothing;for(const auto& pass:plans[0])smoothing.push_back(pass.filter);
  gaussian_nd_into(prepared,other.get(),type,type,shape,smoothing,true);
  if(!finite(other.get(),count))return false;
  std::size_t stride=count;for(std::size_t axis=0;axis<n;++axis){stride/=shape[axis];gradient(other.get(),derivatives[axis].get(),count,shape[axis],stride,workers);}
 }
 std::size_t field=0;for(std::size_t r=0;r<n;++r)for(std::size_t c=r;c<n;++c){
  const auto first=xy ? n-1-r : r,second=xy ? n-1-c : c;auto* target=static_cast<T*>(output[field++]);
  if(gaussian){if(!filter(derivatives[first].get(),target,plans[second]))return false;}
  else{std::size_t stride=1;for(std::size_t a=second+1;a<n;++a)stride*=shape[a];gradient(derivatives[first].get(),target,count,shape[second],stride,workers);if(!finite(target,count))return false;}
 }return true;
}
}}
namespace skimagexx {
bool hessian_matrix_into(const void* input,PixelType source,void* const* output,const std::vector<std::size_t>& shape,const std::vector<std::vector<HessianPass>>& plans,bool gaussian,bool xy){
 if(shape.empty() || shape.size()>5 || (xy && shape.size()!=2) || plans.size()!=(gaussian ? shape.size() : 1))throw std::invalid_argument("invalid Hessian dimensions/plan");
 const auto bytes=std::max(sizeof(double),pixel_size(source));std::size_t count=1;
 for(auto length:shape){if(!length)return true;if(!gaussian && length<2)throw std::invalid_argument("finite Hessian axes require two samples");if(length>std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/bytes/count)throw std::overflow_error("Hessian dimensions overflow");count*=length;}
 for(const auto& list:plans)for(const auto& plan:list){const auto& p=plan.filter;
  if(p.axis>=shape.size() || p.weights.empty() || p.weights.size()%2==0 || int(p.border)<0 || int(p.border)>4)throw std::invalid_argument("invalid Hessian pass");
  const auto constant_limit=source==PixelType::float32 ? double(std::numeric_limits<float>::max())/16 : std::numeric_limits<double>::max()/16;
  if(!std::isfinite(p.cval) || std::abs(p.cval)>constant_limit)return false;
  double sum=0.;for(auto weight:p.weights){if(!std::isfinite(weight))return false;sum+=std::abs(weight);}
  if(sum>1e6)return false;
  for(std::size_t i=0;i<p.weights.size();++i)if(std::abs(p.weights[i]-(plan.derivative ? -1 : 1)*p.weights[p.weights.size()-1-i])>16*std::numeric_limits<double>::epsilon()*std::max(1.,sum))throw std::invalid_argument("Hessian kernel symmetry mismatch");
 }
 if(!input || !output)throw std::invalid_argument("nonnull Hessian buffers required");
 for(std::size_t i=0;i<shape.size()*(shape.size()+1)/2;++i)if(!output[i])throw std::invalid_argument("nonnull Hessian output required");
 return source==PixelType::float32 ? execute<float>(input,source,output,shape,plans,gaussian,xy,count) : execute<double>(input,source,output,shape,plans,gaussian,xy,count);
}
}
