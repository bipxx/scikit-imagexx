#include "skimagexx/structure_tensor.hpp"
#include "skimagexx/derivatives.hpp"
#include "skimagexx/image.hpp"
#include "structure_tensor_detail.hpp"
#include "execution.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace skimagexx { namespace {
std::ptrdiff_t neighbor(std::size_t coordinate,std::size_t length,bool next,GaussianBorder border){
    if(next && coordinate+1<length)return std::ptrdiff_t(coordinate+1);
    if(!next && coordinate)return std::ptrdiff_t(coordinate-1);
    if(border==GaussianBorder::constant)return -1;
    if(border==GaussianBorder::wrap)return next ? 0 : std::ptrdiff_t(length-1);
    if(border==GaussianBorder::mirror && length>1)return next ? std::ptrdiff_t(length-2) : 1;
    return std::ptrdiff_t(coordinate);
}
template<class T> void sobel_pass(const T* input,T* output,std::size_t count,
    std::size_t length,std::size_t stride,GaussianBorder border,double cval,bool derivative,int workers){
    detail::partition(count,workers,[&](std::size_t begin,std::size_t end,auto){
        while(begin<end){
            const auto coordinate=(begin/stride)%length,inner=begin%stride;
            const auto block=begin-coordinate*stride-inner;
            auto stop=std::min(end,begin+stride-inner);
            // Adjacent last-axis pixels have adjacent neighbors. Keep their
            // interior together, mapping only the two boundary pixels.
            if(stride==1 && coordinate && coordinate+1<length)stop=std::min(end,block+length-1);
            const auto left_index=neighbor(coordinate,length,false,border),right_index=neighbor(coordinate,length,true,border);
            const auto* left=left_index<0 ? nullptr : input+block+std::size_t(left_index)*stride+inner;
            const auto* right=right_index<0 ? nullptr : input+block+std::size_t(right_index)*stride+inner;
            std::size_t done=0;
#ifdef SKIMAGEXX_AVX2
            if(has_avx2())done=simd::structure_pass(left,input+begin,right,output+begin,stop-begin,std::is_same_v<T,float>,derivative,cval);
#endif
            for(;done<stop-begin;++done){
                const double a=left ? double(left[done]) : cval,b=right ? double(right[done]) : cval;
                // scipy.ndimage's symmetric/antisymmetric correlate1d order,
                // including the signed-zero contribution from the center.
                double value=double(input[begin+done])*(derivative ? 0. : 2.);
                value+=derivative ? (a-b)*-1. : a+b;
                output[begin+done]=T(value);
            }
            begin=stop;
        }
    });
}
template<class T> bool execute(const void* image,PixelType source,void* const* output,
    const std::vector<std::size_t>& shape,const std::vector<GaussianPass>& smoothing,
    const std::vector<GaussianBorder>& borders,double cval,bool xy,std::size_t count){
    const auto type=std::is_same_v<T,float> ? PixelType::float32 : PixelType::float64;
    const auto bound=std::sqrt(double(std::numeric_limits<T>::max()))/(16*std::pow(4.,double(shape.size()-1)));
    if(!std::isfinite(cval) || std::abs(cval)>bound)return false;
    for(const auto& pass:smoothing)if(!std::isfinite(pass.cval) || std::abs(pass.cval)>double(std::numeric_limits<T>::max())/4)return false;
    std::vector<T> converted;
    const T* input;
    if(source==type)input=static_cast<const T*>(image);
    else{converted.resize(count);gaussian_nd_into(image,converted.data(),source,type,shape,{},false);input=converted.data();}
    std::size_t checked=0;bool valid=true;
#ifdef SKIMAGEXX_AVX2
    if(has_avx2())checked=simd::structure_validate(input,count,std::is_same_v<T,float>,bound,valid);
#endif
    if(!valid)return false;
    for(;checked<count;++checked)if(!std::isfinite(input[checked]) || std::abs(double(input[checked]))>bound)return false;
    const auto n=shape.size();
    std::vector<std::size_t> strides(n,1);
    for(std::size_t axis=n-1;axis;--axis)strides[axis-1]=strides[axis]*shape[axis];
    std::vector<std::vector<T>> derivatives(n,std::vector<T>(count));
    std::vector<T> scratch(count);
    const auto workers=detail::active_workers(count>std::numeric_limits<std::size_t>::max()/3 ? count : count*3);
    for(std::size_t axis=0;axis<n;++axis){
        auto* current=n%2 ? derivatives[axis].data() : scratch.data();
        sobel_pass(input,current,count,shape[axis],strides[axis],borders[axis],cval,true,workers);
        for(std::size_t smooth=0;smooth<n;++smooth)if(smooth!=axis){
            auto* destination=current==scratch.data() ? derivatives[axis].data() : scratch.data();
            sobel_pass(current,destination,count,shape[smooth],strides[smooth],borders[smooth],cval,false,workers);
            current=destination;
        }
    }
    std::size_t field=0;
    for(std::size_t r=0;r<n;++r)for(std::size_t c=r;c<n;++c){
        const auto* first=derivatives[xy ? n-1-r : r].data();
        const auto* second=derivatives[xy ? n-1-c : c].data();
        auto* destination=static_cast<T*>(output[field++]);
        auto* product=smoothing.empty() ? destination : scratch.data();
        detail::partition(count,workers,[&](std::size_t begin,std::size_t end,auto){
#ifdef SKIMAGEXX_AVX2
            if(has_avx2())begin+=simd::structure_product(first+begin,second+begin,product+begin,end-begin,std::is_same_v<T,float>);
#endif
            for(;begin<end;++begin)product[begin]=T(first[begin]*second[begin]);
        });
        if(!smoothing.empty())gaussian_nd_into(product,destination,type,type,shape,smoothing,true);
    }
    return true;
}
}}
namespace skimagexx {
void sobel_derivative_2d_into(const void* input,void* output,void* scratch,std::size_t height,std::size_t width,bool single,std::size_t axis,const std::array<GaussianBorder,2>& borders,double cval){
    if(axis>1 || int(borders[0])<0 || int(borders[0])>4 || int(borders[1])<0 || int(borders[1])>4)throw std::invalid_argument("invalid Sobel derivative plan");
    if(!height || !width)return;
    if(height>std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/sizeof(double)/width)throw std::overflow_error("Sobel derivative dimensions overflow");
    if(!input || !output || !scratch)throw std::invalid_argument("nonnull Sobel derivative buffers required");
    const auto count=height*width;const auto workers=detail::active_workers(count*3);
    auto run=[&](auto dummy){using T=decltype(dummy);
        sobel_pass(static_cast<const T*>(input),static_cast<T*>(scratch),count,axis ? width : height,axis ? 1 : width,borders[axis],cval,true,workers);
        sobel_pass(static_cast<const T*>(scratch),static_cast<T*>(output),count,axis ? height : width,axis ? width : 1,borders[1-axis],cval,false,workers);
    };
    if(single)run(float{});else run(double{});
}
bool structure_tensor_into(const void* image,PixelType source,void* const* output,
    const std::vector<std::size_t>& shape,const std::vector<GaussianPass>& smoothing,
    const std::vector<GaussianBorder>& borders,double cval,bool xy){
    if(shape.size()<2 || shape.size()>5 || borders.size()!=shape.size() || (xy && shape.size()!=2))throw std::invalid_argument("invalid structure tensor dimensions");
    const auto bytes=std::max(sizeof(double),pixel_size(source));
    std::size_t count=1;for(auto length:shape){
        if(!length)return true;
        if(length>std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/bytes/count)throw std::overflow_error("structure tensor dimensions overflow");
        count*=length;
    }
    for(auto border:borders)if(int(border)<0 || int(border)>4)throw std::invalid_argument("invalid structure tensor border");
    for(const auto& pass:smoothing){
        if(pass.axis>=shape.size() || pass.weights.empty() || pass.weights.size()%2==0 || int(pass.border)<0 || int(pass.border)>4)throw std::invalid_argument("invalid structure tensor smoothing pass");
        double sum=0.;for(auto weight:pass.weights){if(!std::isfinite(weight) || weight<0.)throw std::invalid_argument("nonnegative finite smoothing required");sum+=weight;}
        if(std::abs(sum-1.)>8*pass.weights.size()*std::numeric_limits<double>::epsilon())throw std::invalid_argument("normalized smoothing required");
    }
    if(!image || !output)throw std::invalid_argument("nonnull structure tensor storage required");
    for(std::size_t field=0;field<shape.size()*(shape.size()+1)/2;++field)if(!output[field])throw std::invalid_argument("nonnull structure tensor output required");
    return source==PixelType::float32 ? execute<float>(image,source,output,shape,smoothing,borders,cval,xy,count)
                                     : execute<double>(image,source,output,shape,smoothing,borders,cval,xy,count);
}
}
