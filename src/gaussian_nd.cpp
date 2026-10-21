#include "skimagexx/gaussian_nd.hpp"
#include "skimagexx/image.hpp"
#include "execution.hpp"
#include "simd.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
namespace skimagexx {
namespace {
std::ptrdiff_t index_at(std::ptrdiff_t x,std::size_t length,GaussianBorder mode) {
    const auto n=std::ptrdiff_t(length);
    if(x>=0 && x<n)return x;
    if(mode==GaussianBorder::constant)return -1;
    if(mode==GaussianBorder::nearest)return std::clamp(x,std::ptrdiff_t(0),n-1);
    if(mode==GaussianBorder::wrap) {x%=n;return x<0 ? x+n : x;}
    if(mode==GaussianBorder::mirror && n==1)return 0;
    const auto period=mode==GaussianBorder::reflect ? 2*n : 2*(n-1);
    x%=period;if(x<0)x+=period;
    return x<n ? x : mode==GaussianBorder::reflect ? period-1-x : period-x;
}
std::size_t elements(const std::vector<std::size_t>& shape,std::size_t bytes) {
    if(std::find(shape.begin(),shape.end(),0)!=shape.end())return 0;
    std::size_t count=1;const auto limit=std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/bytes;
    for(auto length:shape) {if(length>limit/count)throw std::overflow_error("Gaussian dimensions overflow");count*=length;}
    return count;
}
template<class T,bool Boolean=false> void prepare(const T* input,double* output,std::size_t count,bool preserve,int workers) {
    detail::partition(count,workers,[&](auto begin,auto end,auto) {
#ifdef SKIMAGEXX_AVX2
        if(has_avx2())begin+=simd::gaussian_prepare(input+begin,output+begin,end-begin,
            Boolean ? PixelType::boolean : sizeof(T)==1 ? (std::is_signed_v<T> ? PixelType::int8 : PixelType::uint8) :
            sizeof(T)==2 ? (std::is_signed_v<T> ? PixelType::int16 : PixelType::uint16) :
            sizeof(T)==4 ? (std::is_signed_v<T> ? PixelType::int32 : PixelType::uint32) :
            std::is_signed_v<T> ? PixelType::int64 : PixelType::uint64,preserve);
#endif
        for(auto i=begin;i<end;++i) {
            if constexpr(Boolean)output[i]=input[i]!=0 ? 1.0 : 0.0;
            else if(preserve)output[i]=double(input[i]);
            else {const auto value=double(input[i])*(1.0/double(std::numeric_limits<T>::max()));output[i]=std::is_signed_v<T> ? std::max(value,-1.0) : value;}
        }
    });
}
template<class Input,class Output> void convolve(const Input* input,Output* output,
    const std::vector<std::size_t>& shape,const GaussianPass& pass,std::size_t count,int workers) {
    const auto radius=pass.weights.size()/2,length=shape[pass.axis];
    std::size_t stride=1;for(std::size_t d=pass.axis+1;d<shape.size();++d)stride*=shape[d];
    const auto center_weight=pass.weights[radius];
    detail::partition(count,workers,[&](auto begin,auto end,auto) {
        std::vector<const void*> left(radius),right(radius);
        while(begin<end) {
            const auto coordinate=(begin/stride)%length;
            const auto block=begin/(stride*length)*(stride*length);
            auto scalar=[&](std::size_t position) {
                const auto x=(position/stride)%length,inner=position%stride;
                double sum=double(input[position])*center_weight;
                for(std::size_t j=0;j<radius;++j) {
                    const auto distance=std::ptrdiff_t(radius-j);
                    const auto lo=index_at(std::ptrdiff_t(x)-distance,length,pass.border),hi=index_at(std::ptrdiff_t(x)+distance,length,pass.border);
                    const double a=lo<0 ? pass.cval : double(input[block+std::size_t(lo)*stride+inner]);
                    const double b=hi<0 ? pass.cval : double(input[block+std::size_t(hi)*stride+inner]);
                    sum+=(a+b)*pass.weights[j];
                }
                output[position]=Output(sum);
            };
            if(stride==1) {
                auto stop=std::min(end,block+length),position=begin;
                const auto inside_begin=block+std::min(radius,length);
                const auto inside_end=block+(length>radius ? length-radius : 0);
                for(;position<stop && position<inside_begin;++position)scalar(position);
                const auto vector_end=std::min(stop,inside_end);
                if(position<vector_end) {
                    for(std::size_t j=0;j<radius;++j) {
                        left[j]=input+position-(radius-j);right[j]=input+position+(radius-j);
                    }
#ifdef SKIMAGEXX_AVX2
                    if(has_avx2())position+=simd::gaussian_rows(left.data(),right.data(),input+position,output+position,vector_end-position,
                        radius,pass.weights.data(),center_weight,pass.cval,std::is_same_v<Input,float>,std::is_same_v<Output,float>);
#endif
                    for(;position<vector_end;++position)scalar(position);
                }
                for(;position<stop;++position)scalar(position);
                begin=stop;
            } else {
                const auto inner=begin%stride;
                // Away from the boundary, every neighbor is a fixed linear
                // offset. Span adjacent axis coordinates too: channel-last
                // images otherwise expose only three pixels per SIMD call.
                const bool interior=coordinate>=radius && radius<length-coordinate;
                const auto span=interior ? std::min(end,block+(length-radius)*stride)-begin : std::min(end-begin,stride-inner);
                for(std::size_t j=0;j<radius;++j) {
                    if(interior) {
                        const auto distance=(radius-j)*stride;
                        left[j]=input+begin-distance;right[j]=input+begin+distance;
                    } else {
                        const auto distance=std::ptrdiff_t(radius-j);
                        const auto lo=index_at(std::ptrdiff_t(coordinate)-distance,length,pass.border),hi=index_at(std::ptrdiff_t(coordinate)+distance,length,pass.border);
                        left[j]=lo<0 ? nullptr : input+block+std::size_t(lo)*stride+inner;
                        right[j]=hi<0 ? nullptr : input+block+std::size_t(hi)*stride+inner;
                    }
                }
                std::size_t done=0;
#ifdef SKIMAGEXX_AVX2
                if(has_avx2())done=simd::gaussian_rows(left.data(),right.data(),input+begin,output+begin,span,radius,
                    pass.weights.data(),center_weight,pass.cval,std::is_same_v<Input,float>,std::is_same_v<Output,float>);
#endif
                for(auto i=done;i<span;++i)scalar(begin+i);
                begin+=span;
            }
        }
    });
}
template<class Input,class Output> ConversionInfo filter(const Input* input,Output* output,PixelType source,PixelType target,
    const std::vector<std::size_t>& shape,const std::vector<GaussianPass>& passes,std::size_t count,int workers) {
    if(passes.empty())return convert_into(input,output,count,source,target);
    std::unique_ptr<Output[]> temporary(passes.size()>1 ? new Output[count] : nullptr);
    const Output* current=nullptr;
    for(std::size_t index=0;index<passes.size();++index) {
        auto* destination=(passes.size()-index)%2 ? output : temporary.get();
        if(index==0)convolve(input,destination,shape,passes[index],count,workers);
        else convolve(current,destination,shape,passes[index],count,workers);
        current=destination;
    }
    return {};
}
}
ConversionInfo gaussian_nd_into(const void* input,void* output,PixelType source,PixelType target,
    const std::vector<std::size_t>& shape,const std::vector<GaussianPass>& passes,bool preserve) {
    if(target!=PixelType::float32 && target!=PixelType::float64)throw std::invalid_argument("floating Gaussian output required");
    const auto count=elements(shape,std::max(sizeof(double),std::max(pixel_size(source),pixel_size(target))));
    for(const auto& pass:passes) {
        if(pass.axis>=shape.size() || pass.weights.empty() || pass.weights.size()%2==0 || int(pass.border)<0 || int(pass.border)>4)
            throw std::invalid_argument("invalid Gaussian pass");
        for(std::size_t j=0;j<pass.weights.size()/2;++j)
            if(std::abs(pass.weights[j]-pass.weights[pass.weights.size()-1-j])>std::numeric_limits<double>::epsilon())
                throw std::invalid_argument("symmetric Gaussian coefficients required");
    }
    if(!count)return {};
    if(!input || !output)throw std::invalid_argument("nonnull Gaussian storage required");
    const auto workers=detail::active_workers(count);
    std::unique_ptr<double[]> prepared;
    if(source!=PixelType::float32 && source!=PixelType::float64) {
        if(!passes.empty() || target!=PixelType::float64)prepared.reset(new double[count]);
        auto* destination=prepared ? prepared.get() : static_cast<double*>(output);
#define PREPARE(tag,type) case PixelType::tag:prepare(static_cast<const type*>(input),destination,count,preserve,workers);break;
        switch(source) {
        case PixelType::boolean:prepare<std::uint8_t,true>(static_cast<const std::uint8_t*>(input),destination,count,preserve,workers);break;
        PREPARE(uint8,std::uint8_t) PREPARE(int8,std::int8_t) PREPARE(uint16,std::uint16_t) PREPARE(int16,std::int16_t)
        PREPARE(uint32,std::uint32_t) PREPARE(int32,std::int32_t) PREPARE(uint64,std::uint64_t) PREPARE(int64,std::int64_t)
        default:throw std::invalid_argument("unsupported Gaussian input");
        }
#undef PREPARE
        if(!prepared)return {};
        input=prepared.get();source=PixelType::float64;
    }
    if(source==PixelType::float32) {
        if(target==PixelType::float32)return filter(static_cast<const float*>(input),static_cast<float*>(output),source,target,shape,passes,count,workers);
        return filter(static_cast<const float*>(input),static_cast<double*>(output),source,target,shape,passes,count,workers);
    }
    if(target==PixelType::float32)return filter(static_cast<const double*>(input),static_cast<float*>(output),source,target,shape,passes,count,workers);
    return filter(static_cast<const double*>(input),static_cast<double*>(output),source,target,shape,passes,count,workers);
}
namespace {
template<class T> void difference(const T* first,const T* second,T* output,std::size_t count) {
    detail::partition(count,detail::active_workers(count),[&](std::size_t begin,std::size_t end,auto) {
#ifdef SKIMAGEXX_AVX2
        if(has_avx2())begin+=simd::gaussian_difference(first+begin,second+begin,output+begin,end-begin,std::is_same_v<T,float>);
#endif
        for(auto i=begin;i<end;++i)output[i]=T(first[i]-second[i]);
    });
}
template<class Input,class Output> void sharpen(const Input* image,const Input* blur,Output* output,std::size_t count,
    double amount,bool wide,bool clip,double lower) {
    detail::partition(count,detail::active_workers(count),[&](std::size_t begin,std::size_t end,auto) {
#ifdef SKIMAGEXX_AVX2
        if(has_avx2())begin+=simd::gaussian_unsharp(image+begin,blur+begin,output+begin,end-begin,
            std::is_same_v<Input,float>,std::is_same_v<Output,float>,amount,wide,clip,lower);
#endif
        for(auto i=begin;i<end;++i) {
            const Input delta=Input(image[i]-blur[i]);
            if(wide || std::is_same_v<Input,double>) {
                double value=double(image[i])+double(delta)*amount;
                if(clip)value=value<lower ? lower : value>1. ? 1. : value;
                output[i]=Output(value);
            } else {
                Input value=Input(image[i]+Input(delta*Input(amount)));
                if(clip)value=value<Input(lower) ? Input(lower) : value>Input(1) ? Input(1) : value;
                output[i]=Output(value);
            }
        }
    });
}
}
void gaussian_difference_into(const void* first,const void* second,void* output,std::size_t count,PixelType type) {
    if(type!=PixelType::float32 && type!=PixelType::float64)throw std::invalid_argument("floating difference buffers required");
    if(count>std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/pixel_size(type))throw std::overflow_error("difference size overflow");
    if(!count)return;
    if(!first || !second || !output)throw std::invalid_argument("nonnull difference storage required");
    if(type==PixelType::float32)difference(static_cast<const float*>(first),static_cast<const float*>(second),static_cast<float*>(output),count);
    else difference(static_cast<const double*>(first),static_cast<const double*>(second),static_cast<double*>(output),count);
}
void gaussian_unsharp_into(const void* image,const void* blur,void* output,std::size_t count,PixelType source,PixelType target,
    double amount,bool wide,bool clip,double lower) {
    if((source!=PixelType::float32 && source!=PixelType::float64) || (target!=PixelType::float32 && target!=PixelType::float64))throw std::invalid_argument("floating unsharp buffers required");
    if(count>std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/std::max(pixel_size(source),pixel_size(target)))throw std::overflow_error("unsharp size overflow");
    if(!count)return;
    if(!image || !blur || !output)throw std::invalid_argument("nonnull unsharp storage required");
    if(source==PixelType::float32) {
        if(target==PixelType::float32)sharpen(static_cast<const float*>(image),static_cast<const float*>(blur),static_cast<float*>(output),count,amount,wide,clip,lower);
        else sharpen(static_cast<const float*>(image),static_cast<const float*>(blur),static_cast<double*>(output),count,amount,wide,clip,lower);
    } else {
        if(target==PixelType::float32)sharpen(static_cast<const double*>(image),static_cast<const double*>(blur),static_cast<float*>(output),count,amount,wide,clip,lower);
        else sharpen(static_cast<const double*>(image),static_cast<const double*>(blur),static_cast<double*>(output),count,amount,wide,clip,lower);
    }
}

}
