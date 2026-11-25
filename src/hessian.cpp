#include "skimagexx/hessian.hpp"
#include "skimagexx/image.hpp"
#include "hessian_detail.hpp"
#include "structure_tensor_detail.hpp"
#include "execution.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
namespace skimagexx::detail {
std::unique_ptr<void,void(*)(void*)> acquire_ridge_scratch(std::size_t,bool);
}
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
template<class T> bool finite(const T* a,std::size_t count,double limit=double(std::numeric_limits<T>::max())/16){
 std::size_t i=0;bool valid=true;
#ifdef SKIMAGEXX_AVX2
 if(has_avx2())i=simd::structure_validate(a,count,std::is_same_v<T,float>,limit,valid);
#endif
 if(!valid)return false;
 for(;i<count;++i)if(!std::isfinite(a[i]) || std::abs(double(a[i]))>limit)return false;
 return true;
}
template<class T> void gradient(const T* input,T* output,std::size_t count,std::size_t length,std::size_t stride,int workers){
#ifdef SKIMAGEXX_AVX2
 const bool use_simd=has_avx2();
#endif
 detail::partition(count,workers,[&](auto begin,auto end,auto){
  while(begin<end){const auto coordinate=begin/stride%length,inner=begin%stride;
   auto stop=std::min(end,begin+stride-inner);
   if(stride==1 && coordinate && coordinate+1<length)stop=std::min(end,begin+length-1-coordinate);
   const bool edge=coordinate==0 || coordinate+1==length;
   const auto* left=input+begin-(coordinate ? stride : 0);
   const auto* right=input+begin+(coordinate+1<length ? stride : 0);
   std::size_t done=0;
#ifdef SKIMAGEXX_AVX2
   if(use_simd)done=simd::hessian_gradient(left,right,output+begin,stop-begin,std::is_same_v<T,float>,edge);
#endif
   for(;done<stop-begin;++done){const T difference=T(right[done]-left[done]);output[begin+done]=edge ? difference : T(difference/T(2));}
   begin=stop;
  }
 });
}
bool antisymmetric(const HessianPass& plan){
 if(!plan.derivative)return false;
 const auto& w=plan.filter.weights;
 for(std::size_t j=0;j<w.size()/2;++j)
  if(std::abs(w[j]-w[w.size()-1-j])>std::numeric_limits<double>::epsilon())return true;
 return false;
}

std::size_t zero_prefix(const std::vector<double>& weights){
 std::size_t first=0;
 while(first<weights.size()/2 && weights[first]==0.)++first;
 return first;
}

template<class T,class Samples>
void convolve_point(double center,T& output,T* derivative_output,const simd::HessianKernel& kernel,const Samples& samples){
 const auto radius=kernel.radius;
 double sum=center*kernel.weights[radius],derivative_sum=0.;
 if(derivative_output)derivative_sum=center*kernel.derivative_weights[radius];
 for(std::size_t j=0;j<radius;++j){
  const auto [a,b]=samples(j);
  if(j>=kernel.smooth_start)sum+=(kernel.derivative ? a-b : a+b)*kernel.weights[j];
  if(derivative_output && j>=kernel.derivative_start)derivative_sum+=(a-b)*kernel.derivative_weights[j];
 }
 output=T(sum);
 if(derivative_output)*derivative_output=T(derivative_sum);
}

template<class T> void convolve(const T* input,T* output,std::size_t count,const std::vector<std::size_t>& shape,
 const HessianPass& plan,int workers,T* derivative_output=nullptr,const HessianPass* derivative_plan=nullptr){
 const auto& p=plan.filter;
 const auto original_radius=p.weights.size()/2,length=shape[p.axis];
 const auto* derivative_weights=derivative_plan ? &derivative_plan->filter.weights : nullptr;
 const auto first=zero_prefix(p.weights),derivative_first=derivative_weights ? zero_prefix(*derivative_weights) : first;
 const auto trim=std::min(first,derivative_first);
 const simd::HessianKernel full{p.weights.data(),derivative_weights ? derivative_weights->data() : nullptr,
  original_radius,0,0,antisymmetric(plan)};
 const simd::HessianKernel kernel{full.weights+trim,full.derivative_weights ? full.derivative_weights+trim : nullptr,
  original_radius-trim,first-trim,derivative_first-trim,full.derivative};
 const auto radius=kernel.radius;
 const auto smooth_full=simd::HessianKernel{full.weights,nullptr,original_radius,0,0,full.derivative};
 const auto derivative_full=simd::HessianKernel{full.derivative_weights,nullptr,original_radius,0,0,true};
 // Skipping exact-zero tails can only affect a cropped negative zero.
 // Recompute that cold case with each filter's original ordered taps.
 auto repair=[&](double center,T& value,T* derivative_value,const auto& samples){
  if(first && value==T(0) && std::signbit(value))
   convolve_point(center,value,static_cast<T*>(nullptr),smooth_full,samples);
  if(derivative_value && derivative_first && *derivative_value==T(0) && std::signbit(*derivative_value))
   convolve_point(center,*derivative_value,static_cast<T*>(nullptr),derivative_full,samples);
 };

 std::size_t stride=1;
 for(std::size_t axis=p.axis+1;axis<shape.size();++axis)stride*=shape[axis];
#ifdef SKIMAGEXX_AVX2
 const bool use_simd=has_avx2();
 constexpr bool single=std::is_same_v<T,float>;
#endif
 // Keep the padded cval in double precision even for float32 images.
 if(stride==1){
  std::vector<simd::HessianTap> border(original_radius);
  for(std::size_t j=0;j<original_radius;++j){
   border[j]={mapped(std::ptrdiff_t(j)-std::ptrdiff_t(original_radius),length,p.border),
       mapped(std::ptrdiff_t(length)+std::ptrdiff_t(j),length,p.border)};
  }
  detail::partition(count/length,workers,[&](auto row_begin,auto row_end,auto){
   std::vector<double> padded(length+2*original_radius);
   for(auto row=row_begin;row<row_end;++row){
    const auto* from=input+row*length;
    auto* target=output+row*length;
    auto* derivative_target=derivative_output ? derivative_output+row*length : nullptr;
    if(row>row_begin && std::memcmp(from,from-length,length*sizeof(T))==0){
     std::copy_n(target-length,length,target);
     if(derivative_target)std::copy_n(derivative_target-length,length,derivative_target);
     continue;
    }
    auto* center=padded.data()+original_radius;
    std::copy_n(from,length,center);
    for(std::size_t j=0;j<original_radius;++j){
     const auto [lo,hi]=border[j];
     padded[j]=lo<0 ? p.cval : double(from[std::size_t(lo)]);
     center[length+j]=hi<0 ? p.cval : double(from[std::size_t(hi)]);
    }
    std::size_t done=0;
#ifdef SKIMAGEXX_AVX2
    if(use_simd)done=simd::hessian_convolve_direct(center,target,derivative_target,length,1,kernel,false,single);
#endif
    for(;done<length;++done){
     const auto samples=[&](std::size_t j){return std::pair{padded[original_radius+done+j-radius],center[done+radius-j]};};
     convolve_point(center[done],target[done],derivative_target ? derivative_target+done : nullptr,kernel,samples);
    }
    if(first || (derivative_target && derivative_first))for(std::size_t x=0;x<length;++x){
     const auto samples=[&](std::size_t j){return std::pair{padded[x+j],padded[2*original_radius+x-j]};};
     repair(center[x],target[x],derivative_target ? derivative_target+x : nullptr,samples);
    }
   }
  });
  return;
 }

 const auto blocks=count/stride;
 const auto row_bytes=(stride-1)*sizeof(T);
 // Typical images differ immediately in the first row.
 bool repeated=std::memcmp(input,input+1,row_bytes)==0;
 for(std::size_t block=1;repeated && block<blocks;++block){
  const auto* row=input+block*stride;
  repeated=std::memcmp(row,row+1,row_bytes)==0;
 }
 const auto offsets=[&](std::size_t x,std::size_t distance){
  const auto d=std::ptrdiff_t(distance);
  const auto lo=mapped(std::ptrdiff_t(x)-d,length,p.border),hi=mapped(std::ptrdiff_t(x)+d,length,p.border);
  return simd::HessianTap{lo<0 ? -1 : lo*std::ptrdiff_t(stride),hi<0 ? -1 : hi*std::ptrdiff_t(stride)};
 };
 const auto sample=[&](const T* base,simd::HessianTap tap){
  return std::pair{tap.left<0 ? p.cval : double(base[tap.left]),tap.right<0 ? p.cval : double(base[tap.right])};
 };
 if(repeated){
  detail::partition(blocks,workers,[&](auto begin,auto end,auto){
   for(auto position=begin;position<end;++position){
    const auto x=position%length;
    const auto* base=input+(position-x)*stride;
    const auto samples=[&](std::size_t j){return sample(base,offsets(x,original_radius-j));};
    T value,derivative_value;
    convolve_point(double(input[position*stride]),value,derivative_output ? &derivative_value : nullptr,full,samples);
    std::fill_n(output+position*stride,stride,value);
    if(derivative_output)std::fill_n(derivative_output+position*stride,stride,derivative_value);
   }
  });
  return;
 }

 detail::partition(count,workers,[&](auto begin,auto end,auto){
  std::vector<simd::HessianTap> taps(radius);
  while(begin<end){
   const auto x=begin/stride%length,inner=begin%stride,base_index=begin-x*stride;
   const auto* base=input+base_index;
   const auto stop=std::min(end,begin+stride-inner);
   const bool interior=radius<=x && radius<length-x;
   if(!interior)for(std::size_t j=0;j<radius;++j)taps[j]=offsets(x,radius-j);
   auto* derivative_target=derivative_output ? derivative_output+begin : nullptr;
   std::size_t done=0;
#ifdef SKIMAGEXX_AVX2
   if(use_simd){
    if(interior)done=simd::hessian_convolve_direct(input+begin,output+begin,derivative_target,stop-begin,stride,kernel,single,single);
    else done=simd::hessian_convolve_indexed(base,input+begin,output+begin,derivative_target,stop-begin,
      taps.data(),kernel,p.cval,p.border==GaussianBorder::constant,single);
   }
#endif
   for(;done<stop-begin;++done){
    const auto* point=input+begin+done;
    const auto samples=[&](std::size_t j){
     if(!interior)return sample(base+done,taps[j]);
     const auto distance=(radius-j)*stride;
     return std::pair{double(*(point-distance)),double(*(point+distance))};
    };
    convolve_point(double(*point),output[begin+done],derivative_target ? derivative_target+done : nullptr,kernel,samples);
   }
   if(first || (derivative_target && derivative_first))for(auto position=begin;position<stop;++position){
    const auto samples=[&](std::size_t j){return sample(base+(position-begin),offsets(x,original_radius-j));};
    repair(double(input[position]),output[position],derivative_output ? derivative_output+position : nullptr,samples);
   }
   begin=stop;
  }
 });
}

bool can_pair(const std::vector<HessianPass>& a,const std::vector<HessianPass>& b){
 if(a.empty() || b.empty() || a.front().derivative==b.front().derivative)return false;
 const auto& x=a.front().filter;
 const auto& y=b.front().filter;
 return x.axis==y.axis && x.weights.size()==y.weights.size() && x.border==y.border &&
  (x.border!=GaussianBorder::constant || std::memcmp(&x.cval,&y.cval,sizeof(double))==0) &&
  antisymmetric(a.front().derivative ? a.front() : b.front());
}

constexpr double bounded_input_limit=4294967296.0; // 2^32
template<class T> bool execute(const void* input,PixelType source,void* const* output,const std::vector<std::size_t>& shape,const std::vector<std::vector<HessianPass>>& plans,bool gaussian,bool xy,std::size_t count,bool bounded_plan){
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
 const auto n=shape.size();
 auto workspace=detail::acquire_ridge_scratch(count*(n+2+(source!=type)),single);
 auto* scratch=static_cast<T*>(workspace.get());
 auto* other=scratch+count;
 auto* derivatives=other+count;
 const T* prepared;
 if(source==type)prepared=static_cast<const T*>(input);
 else{auto* converted=derivatives+n*count;gaussian_nd_into(input,converted,source,type,shape,{},false);prepared=converted;}
 const bool bounded=bounded_plan && finite(prepared,count,bounded_input_limit);
 if(!bounded && !finite(prepared,count))return false;
 const auto workers=detail::active_workers(count*4);
 auto filter=[&](const T* from,T* to,const std::vector<HessianPass>& passes){
  if(passes.empty()){std::copy(from,from+count,to);return true;}
  const T* current=from;
  for(std::size_t i=0;i<passes.size();++i){T* target=i+1==passes.size() ? to : current==scratch ? other : scratch;
   convolve(current,target,count,shape,passes[i],workers);if(!bounded && !finite(target,count))return false;current=target;
  }return true;
 };
 auto filter_pair=[&](const T* from,T* to_a,T* to_b,const auto& a,const auto& b){
  auto* first_a=a.size()%2 ? to_a : scratch;
  auto* first_b=b.size()%2 ? to_b : other;
  if(a.front().derivative)convolve(from,first_b,count,shape,b.front(),workers,first_a,&a.front());
  else convolve(from,first_a,count,shape,a.front(),workers,first_b,&b.front());
  if(!bounded && (!finite(first_a,count) || !finite(first_b,count)))return false;
  // Each chain has its own temporary buffer. Alternate with its eventual
  // output so the remaining passes leave the other intermediate intact.
  auto finish=[&](T* current,T* target,T* temporary,const auto& passes){
   for(std::size_t i=1;i<passes.size();++i){
    auto* next=current==target ? temporary : target;
    convolve(current,next,count,shape,passes[i],workers);
    if(!bounded && !finite(next,count))return false;
    current=next;
   }
   return true;
  };
  return finish(first_a,to_a,scratch,a) && finish(first_b,to_b,other,b);
 };
 auto filter_group=[&](const T* from,std::size_t begin,auto axis_for,auto target_for){
  std::array<bool,5> ready{};
  for(auto i=begin;i<n;++i){
   if(ready[i])continue;
   const auto& a=plans[axis_for(i)];
   auto j=i+1;
   while(j<n && (ready[j] || !can_pair(a,plans[axis_for(j)])))++j;
   if(j==n){
    if(!filter(from,target_for(i),a))return false;
   }else{
    if(!filter_pair(from,target_for(i),target_for(j),a,plans[axis_for(j)]))return false;
    ready[j]=true;
   }
  }
  return true;
 };
 if(gaussian){
  if(!filter_group(prepared,0,[](auto i){return i;},[&](auto i){return derivatives+i*count;}))return false;
 }
 else{
  std::vector<GaussianPass> smoothing;for(const auto& pass:plans[0])smoothing.push_back(pass.filter);
  gaussian_nd_into(prepared,other,type,type,shape,smoothing,true);
  if(!finite(other,count))return false;
  std::size_t stride=count;for(std::size_t axis=0;axis<n;++axis){stride/=shape[axis];gradient(other,derivatives+axis*count,count,shape[axis],stride,workers);}
 }
 std::size_t field=0;
 for(std::size_t r=0;r<n;++r){
  const auto axis_for=[&](auto i){return xy ? n-1-i : i;};
  const auto target_for=[&](auto c){return static_cast<T*>(output[field+c-r]);};
  const auto* from=derivatives+axis_for(r)*count;
  if(gaussian){
   if(!filter_group(from,r,axis_for,target_for))return false;
  }else for(auto c=r;c<n;++c){
   const auto second=axis_for(c);
   std::size_t stride=1;for(std::size_t a=second+1;a<n;++a)stride*=shape[a];
   auto* target=target_for(c);
   gradient(from,target,count,shape[second],stride,workers);
   if(!finite(target,count))return false;
  }
  field+=n-r;
 }
 return true;
}
}}
namespace skimagexx {
bool hessian_matrix_into(const void* input,PixelType source,void* const* output,const std::vector<std::size_t>& shape,const std::vector<std::vector<HessianPass>>& plans,bool gaussian,bool xy){
 if(shape.empty() || shape.size()>5 || (xy && shape.size()!=2) || plans.size()!=(gaussian ? shape.size() : 1))throw std::invalid_argument("invalid Hessian dimensions/plan");
 const auto bytes=std::max(sizeof(double),pixel_size(source));std::size_t count=1;
 for(auto length:shape){if(!length)return true;if(!gaussian && length<2)throw std::invalid_argument("finite Hessian axes require two samples");if(length>std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/bytes/count)throw std::overflow_error("Hessian dimensions overflow");count*=length;}
 // Two chains of at most eight passes start within 2^32. With the gain
 // and tap-count bounds below, each pass (including double rounding and
 // the final cast) grows that bound by less than 16, ending below 2^96.
 bool bounded_plan=gaussian;
 for(const auto& list:plans){
 bounded_plan=bounded_plan && list.size()<=8;
 for(const auto& plan:list){const auto& p=plan.filter;
  if(p.axis>=shape.size() || p.weights.empty() || p.weights.size()%2==0 || int(p.border)<0 || int(p.border)>4)throw std::invalid_argument("invalid Hessian pass");
  const auto constant_limit=source==PixelType::float32 ? double(std::numeric_limits<float>::max())/16 : std::numeric_limits<double>::max()/16;
  if(!std::isfinite(p.cval) || std::abs(p.cval)>constant_limit)return false;
  double sum=0.;for(auto weight:p.weights){if(!std::isfinite(weight))return false;sum+=std::abs(weight);}
  if(sum>1e6)return false;
  bounded_plan=bounded_plan && sum<=4. && p.weights.size()<=(1u<<24) && std::abs(p.cval)<=bounded_input_limit;
  for(std::size_t i=0;i<p.weights.size();++i)if(std::abs(p.weights[i]-(plan.derivative ? -1 : 1)*p.weights[p.weights.size()-1-i])>16*std::numeric_limits<double>::epsilon()*std::max(1.,sum))throw std::invalid_argument("Hessian kernel symmetry mismatch");
 }}
 if(!input || !output)throw std::invalid_argument("nonnull Hessian buffers required");
 for(std::size_t i=0;i<shape.size()*(shape.size()+1)/2;++i)if(!output[i])throw std::invalid_argument("nonnull Hessian output required");
 return source==PixelType::float32 ? execute<float>(input,source,output,shape,plans,gaussian,xy,count,bounded_plan) : execute<double>(input,source,output,shape,plans,gaussian,xy,count,bounded_plan);
}
}
