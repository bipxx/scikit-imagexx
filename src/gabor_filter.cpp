#include "skimagexx/gabor_filter.hpp"
#include "execution.hpp"
#include "simd.hpp"
#include "gabor_filter_detail.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace skimagexx { namespace {
std::size_t multiply(std::size_t a,std::size_t b) {
    const auto limit=std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/sizeof(double);
    if(b && a>limit/b)throw std::overflow_error("Gabor filter dimensions overflow");return a*b;
}
std::ptrdiff_t mapped(std::ptrdiff_t coordinate,std::size_t length,GaussianBorder border) {
    const auto n=std::ptrdiff_t(length);
    if(coordinate>=0 && coordinate<n)return coordinate;
    if(border==GaussianBorder::constant)return -1;
    if(border==GaussianBorder::nearest)return std::clamp(coordinate,std::ptrdiff_t(0),n-1);
    if(border==GaussianBorder::wrap){coordinate%=n;return coordinate<0 ? coordinate+n : coordinate;}
    if(n==1)return 0;
    const auto period=border==GaussianBorder::reflect ? 2*n : 2*(n-1);
    coordinate%=period;if(coordinate<0)coordinate+=period;
    return coordinate<n ? coordinate : border==GaussianBorder::reflect ? period-1-coordinate : period-coordinate;
}
// ndimage's integer stores truncate and wrap; never invoke undefined C++
// floating-to-integer conversion for nonfinite or out-of-range intermediates.
std::int32_t truncate32(double x) {return x>=-2147483648. && x<2147483648. ? std::int32_t(x) : std::numeric_limits<std::int32_t>::min();}
std::int64_t truncate64(double x) {return x>=-9223372036854775808. && x<9223372036854775808. ? std::int64_t(x) : std::numeric_limits<std::int64_t>::min();}
template<class T> T cast(double x) {
    if constexpr(std::is_floating_point_v<T>)return T(x);
    else {
        using U=std::make_unsigned_t<T>;U bits;
        if constexpr(sizeof(T)<4)bits=U(std::uint32_t(truncate32(x)));
        else if constexpr(sizeof(T)==4 && std::is_signed_v<T>)bits=U(truncate32(x));
        else if constexpr(sizeof(T)==4)bits=U(truncate64(x));
        else if constexpr(std::is_signed_v<T>)bits=U(truncate64(x));
        else bits=x>=9223372036854775808. ? U(truncate64(x-9223372036854775808.))+U(1ULL<<63) : U(truncate64(x));
        T result;std::memcpy(&result,&bits,sizeof(T));return result;
    }
}
template<class T> void store(void* destination,std::size_t offset,const double* values,std::size_t width) {
    auto* out=static_cast<T*>(destination)+offset;std::size_t x=0;
#ifdef SKIMAGEXX_AVX2
    if constexpr(std::is_same_v<T,float>)if(has_avx2())x=simd::gabor_filter_store_float(values,out,width);
#endif
    for(;x<width;++x){
        const auto v=values[x];
        if constexpr(std::is_unsigned_v<T>)out[x]=v>-1 ? cast<T>(v) : T(T(0)-cast<T>(-v));
        else out[x]=cast<T>(v);
    }
}
template<class T> void prepare(const T* image,double* padded,std::size_t height,std::size_t width,
    std::size_t pitch,const std::vector<std::ptrdiff_t>& xmap,PixelType type,double cval,int workers) {
    const auto radius=(pitch-width)/2;
    detail::partition(height,workers,[&](auto begin,auto end,auto){
        for(auto y=begin;y<end;++y){auto* row=padded+y*pitch;const auto* source=image+y*width;
            std::size_t x=0;
#ifdef SKIMAGEXX_AVX2
            if(has_avx2()){
                if constexpr(std::is_same_v<T,float>)x=simd::gabor_filter_widen_float(source,row+radius,width);
                else x=simd::gaussian_prepare(source,row+radius,width,type==PixelType::boolean ? PixelType::uint8 : type,true);
            }
#endif
            if constexpr(std::is_same_v<T,double>){std::copy_n(source,width,row+radius);x=width;}
            for(;x<width;++x)row[radius+x]=double(source[x]);
            for(x=0;x<radius;++x)row[x]=xmap[x]<0 ? cval : double(source[std::size_t(xmap[x])]);
            for(x=radius+width;x<pitch;++x)row[x]=xmap[x]<0 ? cval : double(source[std::size_t(xmap[x])]);
        }
    });
}
template<class T> void run(const void* image,void* real,void* imaginary,PixelType type,
    std::size_t height,std::size_t width,const void* kernel,bool single,
    std::size_t kh,std::size_t kw,GaussianBorder border,double cval,bool collision) {
    const auto count=multiply(height,width),pitch=width+kw-1;
    const auto padded_count=multiply(height,pitch);
    struct Weight {std::size_t y,x;double real,imaginary;};std::vector<Weight> weights;
    const auto kc=multiply(kh,kw);weights.reserve(kc);
    const auto weight=[&](std::size_t index){return single ? double(static_cast<const float*>(kernel)[index]) : static_cast<const double*>(kernel)[index];};
    for(std::size_t y=0;y<kh;++y)for(std::size_t x=0;x<kw;++x){
        const auto offset=2*(kc-1-(y*kw+x));auto r=weight(offset),i=weight(offset+1);
        r=std::abs(r)>std::numeric_limits<double>::epsilon() ? r : 0.;
        i=std::abs(i)>std::numeric_limits<double>::epsilon() ? i : 0.;
        if(r!=0 || i!=0)weights.push_back({y,x,r,i});
    }
    const auto work=count>std::numeric_limits<std::size_t>::max()/std::max(std::size_t(1),weights.size()) ? std::numeric_limits<std::size_t>::max() : count*std::max(std::size_t(1),weights.size());
    const auto workers=detail::active_workers(work);
    std::vector<std::ptrdiff_t> ymap;std::vector<double> padded;
    const auto representative=[](std::size_t coordinate,std::size_t length,std::size_t kernel_size){
        const auto radius=kernel_size/2;
        if(kernel_size<=length && coordinate>=radius && coordinate<=length-1-radius)return radius;
        return coordinate;
    };
    if(collision){
        // Offsets in ndimage are shared across the interior region. With
        // zero strides the sentinel mask is therefore based on its stored
        // representative coordinate, not each output's absolute coordinate.
        // Precompute contiguous sample fields and keep the same SIMD row loop.
        padded.resize(multiply(kw,width));const auto value=double(static_cast<const T*>(image)[0]);
        for(std::size_t kx=0;kx<kw;++kx)for(std::size_t x=0;x<width;++x){
            const auto coordinate=mapped(std::ptrdiff_t(representative(x,width,kw))+std::ptrdiff_t(kx)-std::ptrdiff_t(kw/2),width,border);
            padded[kx*width+x]=coordinate<0 || coordinate==1 ? cval : value;
        }
    }else{
        std::vector<std::ptrdiff_t> xmap(pitch);ymap.resize(height+kh-1);
        for(std::size_t x=0;x<pitch;++x)xmap[x]=mapped(std::ptrdiff_t(x)-std::ptrdiff_t(kw/2),width,border);
        for(std::size_t y=0;y<ymap.size();++y)ymap[y]=mapped(std::ptrdiff_t(y)-std::ptrdiff_t(kh/2),height,border);
        padded.resize(padded_count);
        prepare(static_cast<const T*>(image),padded.data(),height,width,pitch,xmap,type,cval,workers);
    }
    detail::partition(height,workers,[&](auto begin,auto end,auto){
        std::vector<detail::GaborFilterTap> taps(weights.size());
        std::vector<double> rbuffer,iBuffer;
        if constexpr(!std::is_same_v<T,double>){rbuffer.resize(width);iBuffer.resize(width);}
        for(auto y=begin;y<end;++y){
            for(std::size_t t=0;t<weights.size();++t){const auto& w=weights[t];
                if(collision){const auto row=mapped(std::ptrdiff_t(representative(y,height,kh))+std::ptrdiff_t(w.y)-std::ptrdiff_t(kh/2),height,border);
                    taps[t]={row<0 || row==1 ? nullptr : padded.data()+w.x*width,w.real,w.imaginary};
                }else{const auto row=ymap[y+w.y];taps[t]={row<0 ? nullptr : padded.data()+std::size_t(row)*pitch+w.x,w.real,w.imaginary};}
            }
            auto* r=std::is_same_v<T,double> ? static_cast<double*>(real)+y*width : rbuffer.data();
            auto* i=std::is_same_v<T,double> ? static_cast<double*>(imaginary)+y*width : iBuffer.data();
            std::size_t x=0;
#ifdef SKIMAGEXX_AVX2
            if(has_avx2())x=simd::gabor_filter_row(taps.data(),taps.size(),r,i,width,cval);
#endif
            for(;x<width;++x){double re=0.,im=0.;
                for(const auto& t:taps){const auto value=t.row ? t.row[x] : cval;
                    if(t.real!=0)re+=t.real*value;if(t.imaginary!=0)im+=t.imaginary*value;}
                r[x]=re;i[x]=im;
            }
            if constexpr(!std::is_same_v<T,double>){store<T>(real,y*width,r,width);store<T>(imaginary,y*width,i,width);}
        }
    });
}
}}
namespace skimagexx {
void gabor_filter_into(const void* image,PixelType type,void* real,void* imaginary,
    std::size_t height,std::size_t width,const void* kernel,bool single,
    std::size_t kh,std::size_t kw,GaussianBorder border,double cval,bool collision) {
    const auto limit=std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/sizeof(double);
    if(!height || !width || !kh || !kw || kh%2!=1 || kw%2!=1 || unsigned(border)>4 ||
        width>limit || height>limit || kw>limit-width+1 || kh>limit-height+1 ||
        !image || !real || !imaginary || !kernel)throw std::invalid_argument("invalid Gabor filter buffers or geometry");
    multiply(height,width);multiply(multiply(kh,kw),2);
    switch(type){
#define GABOR_CASE(tag,T) case PixelType::tag:run<T>(image,real,imaginary,type,height,width,kernel,single,kh,kw,border,cval,collision);return;
        GABOR_CASE(boolean,std::uint8_t) GABOR_CASE(uint8,std::uint8_t) GABOR_CASE(int8,std::int8_t)
        GABOR_CASE(uint16,std::uint16_t) GABOR_CASE(int16,std::int16_t) GABOR_CASE(uint32,std::uint32_t)
        GABOR_CASE(int32,std::int32_t) GABOR_CASE(uint64,std::uint64_t) GABOR_CASE(int64,std::int64_t)
        GABOR_CASE(float32,float) GABOR_CASE(float64,double)
#undef GABOR_CASE
    }
    throw std::invalid_argument("unsupported Gabor pixel type");
}
}
