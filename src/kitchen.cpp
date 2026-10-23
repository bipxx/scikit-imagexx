#include "skimagexx/kitchen.hpp"
#include "skimagexx/image.hpp"
#include "structure_tensor_detail.hpp"
#include "kitchen_detail.hpp"
#include "execution.hpp"
#include <atomic>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>

namespace skimagexx { namespace {
template<class T> bool run(const void* image,void* destination,PixelType source,std::size_t height,std::size_t width,const std::array<GaussianBorder,2>& borders,double cval){
    const auto count=height*width;const auto bound=std::cbrt(double(std::numeric_limits<T>::max()))/128;
    if(!std::isfinite(cval) || std::abs(cval)>bound)return false;
    constexpr bool single=std::is_same_v<T,float>;const auto type=single ? PixelType::float32 : PixelType::float64;
    std::unique_ptr<T[]> converted;const T* input;
    if(type==source)input=static_cast<const T*>(image);
    else{converted.reset(new T[count]);gaussian_nd_into(image,converted.get(),source,type,{height,width},{},true);input=converted.get();}
    std::size_t checked=0;bool valid=true;
#ifdef SKIMAGEXX_AVX2
    if(has_avx2())checked=simd::structure_validate(input,count,single,bound,valid);
#endif
    if(!valid)return false;
    for(;checked<count;++checked)if(!std::isfinite(input[checked]) || std::abs(double(input[checked]))>bound)return false;
    std::unique_ptr<T[]> gy(new T[count]),gx(new T[count]),gxy(new T[count]),gxx(new T[count]),gyy(new T[count]),scratch(new T[count]);
    sobel_derivative_2d_into(input,gy.get(),scratch.get(),height,width,single,0,borders,cval);
    sobel_derivative_2d_into(input,gx.get(),scratch.get(),height,width,single,1,borders,cval);
    sobel_derivative_2d_into(gx.get(),gxy.get(),scratch.get(),height,width,single,0,borders,cval);
    sobel_derivative_2d_into(gx.get(),gxx.get(),scratch.get(),height,width,single,1,borders,cval);
    sobel_derivative_2d_into(gy.get(),gyy.get(),scratch.get(),height,width,single,0,borders,cval);
    const auto* y=gy.get();const auto* x=gx.get();const auto* xy=gxy.get();const auto* xx=gxx.get();const auto* yy=gyy.get();
    auto* output=static_cast<T*>(destination);std::atomic<bool> supported{true};
    detail::partition(count,detail::active_workers(count*12),[&](auto begin,auto end,auto){
        bool ok=true;
#ifdef SKIMAGEXX_AVX2
        const void* fields[]={y+begin,x+begin,xy+begin,xx+begin,yy+begin};
        if(has_avx2())begin+=simd::kitchen_response(fields,output+begin,end-begin,single,ok);
#endif
        if(!ok){supported.store(false,std::memory_order_relaxed);return;}
        for(;begin<end;++begin){
            const T y2=T(y[begin]*y[begin]),x2=T(x[begin]*x[begin]);
            const T sum=T(T(xx[begin]*y2)+T(yy[begin]*x2));
            const T cross=T(T(T(T(2)*xy[begin])*x[begin])*y[begin]);
            const T numerator=T(sum-cross),denominator=T(x2+y2);
            const T value=denominator!=T(0) ? T(numerator/denominator) : T(0);
            if(!std::isfinite(value)){supported.store(false,std::memory_order_relaxed);return;}
            output[begin]=value;
        }
    });
    return supported.load(std::memory_order_relaxed);
}
}}
namespace skimagexx {
bool kitchen_rosenfeld_into(const void* image,void* output,PixelType source,std::size_t height,std::size_t width,const std::array<GaussianBorder,2>& borders,double cval){
    pixel_size(source);
    for(auto border:borders)if(int(border)<0 || int(border)>4)throw std::invalid_argument("invalid Kitchen-Rosenfeld border");
    if(!height || !width)return true;
    if(height>std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/sizeof(double)/width)throw std::overflow_error("Kitchen-Rosenfeld dimensions overflow");
    if(!image || !output)throw std::invalid_argument("nonnull Kitchen-Rosenfeld buffers required");
    return source==PixelType::float32 ? run<float>(image,output,source,height,width,borders,cval) : run<double>(image,output,source,height,width,borders,cval);
}
}
