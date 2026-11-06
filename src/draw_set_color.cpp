// Derived operation order from scikit-image0.26 draw.set_color (BSD-3-Clause).
// Attribution: third_party/scikit-image-LICENSE.txt.
#include <skimagexx/draw_set_color.hpp>
#include "execution.hpp"
#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
#include <xmmintrin.h>
#endif
#pragma STDC FENV_ACCESS ON
namespace skimagexx::detail {
namespace {
bool ordinary(double x) {std::uint64_t bits;std::memcpy(&bits,&x,8);bits&=0x7fffffffffffffffull;return bits<=0x4030000000000000ull&&(bits==0||bits>=0x37f0000000000000ull);}
bool integral_bounded(double x) {
    // ordinary() already established zero or finite magnitude<=16.
    std::uint64_t bits;std::memcpy(&bits,&x,8);bits&=0x7fffffffffffffffull;
    if(bits==0)return true;
    const int exponent=int((bits>>52)&0x7ffu)-1023;
    return exponent>=0 && (bits&((std::uint64_t{1}<<(52-exponent))-1))==0;
}
double load(const char* p){double x;std::memcpy(&x,p,8);return x;}
}
bool draw_set_color(const SetColorView& v,int workers) {
    if(workers<1||workers>1024||!v.image||!v.color||!v.shape[2]||v.shape[2]>1024)return false;
    if(v.count&&(!v.rows||!v.columns||!v.alpha))return false;
    if(std::fegetround()!=FE_TONEAREST)return false;
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
    if(_mm_getcsr()&((1u<<15)|(1u<<6)|(3u<<13)))return false;
#endif
    const std::size_t bytes=v.image_float32?4:8;
    const auto limit=std::size_t(std::numeric_limits<std::ptrdiff_t>::max());
    std::size_t pos=0,neg=0;std::array<std::size_t,3> axes{0,1,2};
    std::array<std::size_t,3> magnitude{};
    for(std::size_t a=0;a<3;++a){
        if(v.shape[a]>limit||v.strides[a]==std::numeric_limits<std::ptrdiff_t>::min())return false;
        magnitude[a]=std::size_t(v.strides[a]<0?-v.strides[a]:v.strides[a]);
        auto& span=v.strides[a]<0?neg:pos;
        if(v.shape[a]&&magnitude[a]&&(v.shape[a]-1)>(limit-span)/magnitude[a])return false;
        if(v.shape[a])span+=(v.shape[a]-1)*magnitude[a];
    }
    if(pos>limit-(bytes-1)||neg>limit-(bytes-1)-pos)return false;
    std::sort(axes.begin(),axes.end(),[&](auto a,auto b){return magnitude[a]<magnitude[b];});
    std::size_t extent=bytes;
    for(auto a:axes)if(v.shape[a]>1){if(magnitude[a]<extent)return false;extent+=(v.shape[a]-1)*magnitude[a];}
    if(v.count>limit/v.shape[2]/sizeof(double))return false;
    for(std::size_t c=0;c<v.shape[2];++c) {
        if(!ordinary(v.color[c]))return false;
        if(v.integer_product&&!integral_bounded(v.color[c]))return false;
    }
    std::vector<std::ptrdiff_t> offsets;std::vector<double> alphas;
    offsets.reserve(v.count);alphas.reserve(v.count);
    for(std::size_t i=0;i<v.count;++i){
        const auto r=v.rows[i],c=v.columns[i];
        if(r<0||c<0||std::size_t(r)>=v.shape[0]||std::size_t(c)>=v.shape[1])continue;
        if(!ordinary(v.alpha[i]))return false;
        if(v.integer_product&&!integral_bounded(v.alpha[i]))return false;
        offsets.push_back(r*v.strides[0]+c*v.strides[1]);alphas.push_back(v.alpha[i]);
    }
    const auto channels=v.shape[2],n=offsets.size(),size=n*channels;
    std::vector<double> original(size),colors(size),values(size),result(size);
    auto* base=static_cast<char*>(v.image);
    for(std::size_t i=0;i<n;++i)for(std::size_t c=0;c<channels;++c){
        const auto* pointer=base+offsets[i]+std::ptrdiff_t(c)*v.strides[2];
        double x;
        if(v.image_float32) {
            std::uint32_t bits;std::memcpy(&bits,pointer,4);const auto magnitude=bits&0x7fffffffu;
            if(magnitude>0x41800000u||(magnitude!=0&&magnitude<0x00800000u))return false;
            float value;std::memcpy(&value,pointer,4);x=double(value);
        } else {
            x=load(pointer);if(!ordinary(x))return false;
        }
        original[i*channels+c]=x;
    }
    // Bounded dyadic operands make every stage normal/finite (including exact
    // cancellation), so these operations produce no NumPy FP policy events.
    // Onlyf32canrejectafterarithmetic: preserveentryFPcontext onthatrefusal.
    std::fenv_t before_arithmetic;
    if(v.image_float32&&std::fegetenv(&before_arithmetic))return false;
    const int team=size>=parallel_min?workers:1;
    partition(size,team,[&](auto begin,auto end,auto){
        if(v.integer_product) {
            for(auto k=begin;k<end;++k)
                colors[k]=double(std::int64_t(v.color[k%channels])*std::int64_t(alphas[k/channels]));
        } else {
            for(auto k=begin;k<end;++k)colors[k]=v.color[k%channels]*alphas[k/channels];
        }
        for(auto k=begin;k<end;++k)values[k]=original[k]*(1.0-alphas[k/channels]);
        for(auto k=begin;k<end;++k)result[k]=values[k]+colors[k];
    });
    if(v.image_float32)for(double value:result) {
        std::uint64_t bits;std::memcpy(&bits,&value,8);bits&=0x7fffffffffffffffull;
        if(bits!=0&&bits<0x3810000000000000ull) {
            std::fesetenv(&before_arithmetic);return false;
        }
    }
    for(std::size_t i=0;i<n;++i)for(std::size_t c=0;c<channels;++c) {
        auto* pointer=base+offsets[i]+std::ptrdiff_t(c)*v.strides[2];
        if(v.image_float32) {const float value=float(result[i*channels+c]);std::memcpy(pointer,&value,4);}
        else std::memcpy(pointer,&result[i*channels+c],8);
    }
    return true;
}
}
