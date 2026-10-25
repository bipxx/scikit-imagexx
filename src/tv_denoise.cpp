#include "skimagexx/tv_denoise.hpp"
#include "tv_denoise_internal.hpp"
#include "execution.hpp"
#include <array>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
#include <xmmintrin.h>
#endif

namespace skimagexx {
namespace {
struct Interval { std::uintptr_t low, high; };
struct InputLayout {
    std::size_t rank=0,n=1,width=0;
    std::array<std::size_t,4> shape{},steps{};
    std::array<std::ptrdiff_t,4> stride{};
    bool contiguous=true;
};
bool interval(const void* pointer,std::size_t bytes,Interval& result) {
    auto address=reinterpret_cast<std::uintptr_t>(pointer);
    if(!pointer || bytes>std::numeric_limits<std::uintptr_t>::max()-address)return false;
    result={address,address+bytes};return true;
}
bool intersects(Interval a,Interval b) {return a.low<b.high && b.low<a.high;}
std::size_t magnitude(std::ptrdiff_t value) {return value<0?std::size_t(-(value+1))+1:std::size_t(value);}
bool input_layout(const TvConstView& input,InputLayout& layout,Interval& span) {
    layout.rank=input.shape.size();
    if(input.type==TvScalarType::float32)layout.width=4;
    else if(input.type==TvScalarType::float64)layout.width=8;
    else return false;
    if(layout.rank<1 || layout.rank>4 || input.strides.size()!=layout.rank || !input.data)return false;
    for(std::size_t axis=layout.rank;axis-->0;) {
        const auto size=input.shape[axis];
        if(!size || size>(std::size_t{1}<<30)/layout.n)return false;
        layout.shape[axis]=size;layout.steps[axis]=layout.n;layout.stride[axis]=input.strides[axis];
        if(size>1 && input.strides[axis]!=std::ptrdiff_t(layout.n*layout.width))layout.contiguous=false;
        layout.n*=size;
    }
    std::size_t below=0,above=0;
    const auto limit=std::size_t(std::numeric_limits<std::ptrdiff_t>::max());
    for(std::size_t axis=0;axis<layout.rank;++axis) {
        const auto stride=magnitude(layout.stride[axis]);
        if(stride && layout.shape[axis]-1>limit/stride)return false;
        const auto delta=(layout.shape[axis]-1)*stride;
        auto& extent=layout.stride[axis]<0?below:above;
        if(delta>limit-extent)return false;
        extent+=delta;
    }
    if(below>limit-above)return false;
    const auto address=reinterpret_cast<std::uintptr_t>(input.data);
    if(address<below || above>std::numeric_limits<std::uintptr_t>::max()-address ||
       layout.width>std::numeric_limits<std::uintptr_t>::max()-address-above)return false;
    span={address-below,address+above+layout.width};return true;
}
std::ptrdiff_t offset(const InputLayout& layout,std::size_t index) {
    if(layout.contiguous)return std::ptrdiff_t(index*layout.width);
    std::ptrdiff_t result=0;
    for(std::size_t axis=0;axis<layout.rank;++axis)
        result+=std::ptrdiff_t((index/layout.steps[axis])%layout.shape[axis])*layout.stride[axis];
    return result;
}
// Advance through the same logical C-order indices without dividing at every
// load. Extent validation precedes every use. Carry subtracts the current
// coordinate contribution before changing an outer axis, so no transient
// one-past byte offset can overflow for a certified negative/large stride.
struct InputCursor {
    const InputLayout& layout;
    std::array<std::size_t,4> coordinate{};
    std::ptrdiff_t byte;
    InputCursor(const InputLayout& value,std::size_t index):layout(value),byte(offset(value,index)) {
        for(std::size_t a=0;a<layout.rank;++a)
            coordinate[a]=(index/layout.steps[a])%layout.shape[a];
    }
    void advance() {
        for(std::size_t a=layout.rank;a-->0;) {
            if(coordinate[a]+1<layout.shape[a]) {
                ++coordinate[a];byte+=layout.stride[a];return;
            }
            byte-=std::ptrdiff_t(coordinate[a])*layout.stride[a];coordinate[a]=0;
        }
    }
};
template<class T> T load(const void* pointer,std::ptrdiff_t bytes) {
    T value;std::memcpy(&value,static_cast<const char*>(pointer)+bytes,sizeof(T));return value;
}
template<class T> std::uint64_t bits(T value) {
    if constexpr(sizeof(T)==4) {std::uint32_t raw;std::memcpy(&raw,&value,4);return raw;}
    else {std::uint64_t raw;std::memcpy(&raw,&value,8);return raw;}
}
bool zero_bytes(const void* pointer,std::size_t bytes) {
    const auto* data=static_cast<const unsigned char*>(pointer);
    // Integer loads preserve the exact positive-zero byte contract, including
    // arbitrary alignment and nonfinite bit patterns. Each load is wholly
    // inside the certified extent; the remainder tests cannot overflow.
    while(bytes>=64) {
        std::uint64_t a,b,c,d,e,f,g,h;
        std::memcpy(&a,data,8);std::memcpy(&b,data+8,8);
        std::memcpy(&c,data+16,8);std::memcpy(&d,data+24,8);
        std::memcpy(&e,data+32,8);std::memcpy(&f,data+40,8);
        std::memcpy(&g,data+48,8);std::memcpy(&h,data+56,8);
        if(a|b|c|d|e|f|g|h)return false;
        data+=64;bytes-=64;
    }
    while(bytes>=8) {
        std::uint64_t word;std::memcpy(&word,data,8);
        if(word)return false;
        data+=8;bytes-=8;
    }
    for(std::size_t i=0;i<bytes;++i)if(data[i])return false;
    return true;
}
bool spatial(const TvView& view,const TvConstView& input,const InputLayout& layout,Interval& span) {
    if(view.type!=input.type || view.shape!=input.shape || view.strides.size()!=layout.rank ||
       reinterpret_cast<std::uintptr_t>(view.data)%layout.width)return false;
    for(std::size_t axis=0;axis<layout.rank;++axis)
        if(view.strides[axis]!=std::ptrdiff_t(layout.steps[axis]*layout.width))return false;
    return interval(view.data,layout.n*layout.width,span);
}
bool planes(const TvPlanes& view,const TvConstView& input,const InputLayout& layout,Interval& span) {
    return view.type==input.type && view.shape==input.shape &&
        reinterpret_cast<std::uintptr_t>(view.data)%layout.width==0 &&
        interval(view.data,layout.n*layout.rank*layout.width,span);
}
bool alias(const TvView& view,const TvConstView& read) {
    return view.data==read.data && view.type==read.type && view.shape==read.shape && view.strides==read.strides;
}
bool alias(const TvPlanes& view,const TvConstPlanes& read) {
    return view.data==read.data && view.type==read.type && view.shape==read.shape;
}
template<class T> bool bounded_input(const TvConstView& input,const InputLayout& layout) {
    constexpr std::uint64_t maximum=sizeof(T)==4?0x49800000ull:0x4130000000000000ull;
    constexpr std::uint64_t sign=sizeof(T)==4?0x80000000ull:0x8000000000000000ull;
    if(layout.contiguous) {
        for(std::size_t i=0;i<layout.n;++i)
            if((bits(load<T>(input.data,offset(layout,i)))&~sign)>maximum)return false;
    } else {
        InputCursor cursor(layout,0);
        for(std::size_t i=0;i<layout.n;++i) {
            if((bits(load<T>(input.data,cursor.byte))&~sign)>maximum)return false;
            if(i+1<layout.n)cursor.advance();
        }
    }
    return true;
}
template<class T> T scalar(std::uint64_t raw) {
    T value;
    if constexpr(sizeof(T)==4) {auto word=std::uint32_t(raw);std::memcpy(&value,&word,4);}
    else std::memcpy(&value,&raw,8);
    return value;
}

template<class T,class Energy> TvDenoiseResult run(const TvConstView& input,
    const TvDenoiseWorkspace& w,const TvDenoiseParameters& parameters,TvDenoiseTrace* trace) {
    InputLayout layout;Interval ignored;input_layout(input,layout,ignored);
    const auto type=sizeof(T)==4?TvReductionType::float32:TvReductionType::float64;
    auto* d=static_cast<T*>(w.d.data);auto* out=static_cast<T*>(w.out.data);
    auto* norm=static_cast<T*>(w.norm.data);
    const double tau=1.0/(2.0*double(layout.rank));
    const T norm_coefficient=T(tau/parameters.weight);
    const Energy weight=Energy(parameters.weight),eps=Energy(parameters.eps),size=Energy(double(layout.n));
    Energy initial=0,previous=0;
    for(std::size_t iteration=0;iteration<parameters.max_iterations;++iteration) {
        if(iteration) {
            tv_chambolle_divergence_into(w.p_read,w.d);
            detail::partition(layout.n,detail::active_workers(layout.n),[&](std::size_t begin,std::size_t end,std::size_t){
                if(layout.contiguous) {
                    for(std::size_t i=begin;i<end;++i)out[i]=T(load<T>(input.data,offset(layout,i))+d[i]);
                } else if(begin<end) {
                    InputCursor cursor(layout,begin);
                    for(std::size_t i=begin;i<end;++i) {
                        out[i]=T(load<T>(input.data,cursor.byte)+d[i]);
                        if(i+1<end)cursor.advance();
                    }
                }
            });
        }
        tv_square_contiguous_into({w.d.data,layout.n,type},w.squared.data);
        const auto squared_sum=tv_sum_contiguous({w.squared.data,layout.n,type},parameters.reduction);
        tv_chambolle_gradient_into(iteration?w.out_read:input,w.g);
        tv_chambolle_norm_into(w.g_read,w.norm);
        const auto norm_sum=tv_sum_contiguous({w.norm.data,layout.n,type},parameters.reduction);
        const Energy weighted=Energy(weight*Energy(scalar<T>(norm_sum.bits)));
        Energy energy=Energy(Energy(scalar<T>(squared_sum.bits))+weighted);
        detail::partition(layout.n,detail::active_workers(layout.n),[&](std::size_t begin,std::size_t end,std::size_t){
            for(std::size_t i=begin;i<end;++i)norm[i]=T(T(norm[i]*norm_coefficient)+T(1));
        });
        tv_chambolle_dual_into(w.p,w.g_read,w.norm_read,tau);
        energy=Energy(energy/size);
        Energy difference=0,threshold=0;bool stopped=false;
        if(!iteration)initial=previous=energy;
        else {
            difference=std::abs(Energy(previous-energy));
            threshold=Energy(eps*initial);
            stopped=difference<threshold;
            if(!stopped)previous=energy;
        }
        if(trace)trace[iteration]={squared_sum.bits,norm_sum.bits,bits(energy),bits(difference),bits(threshold),sizeof(Energy)==4,stopped};
        if(stopped)return {iteration+1,false};
    }
    return {parameters.max_iterations,parameters.max_iterations==1};
}
} // namespace

const char* tv_denoise_ineligibility(const TvConstView& input,const TvDenoiseWorkspace& w,
    const TvDenoiseParameters& parameters,TvDenoiseTrace* trace,std::size_t capacity) noexcept {
    InputLayout layout;std::array<Interval,8> spans{};
    if(!input_layout(input,layout,spans[0]))return "TV finite input requires rank1..4, valid strides and 1..2^30 elements";
    const auto weight=bits(parameters.weight),epsilon=bits(parameters.eps);
    if(weight<0x3eb0000000000000ull || weight>0x4130000000000000ull)return "TV finite weight requires positive [2^-20,2^20]";
    const auto eps_abs=epsilon&0x7fffffffffffffffull;
    if(eps_abs>0x4130000000000000ull || ((epsilon>>63)&&eps_abs))return "TV finite eps requires [0,2^20]";
    if(!parameters.max_iterations || parameters.max_iterations>20000)return "TV finite iterations require [1,20000]";
    if(parameters.reduction.profile!=TvReductionProfile::numpy_1244_buffered &&
       parameters.reduction.profile!=TvReductionProfile::numpy_254_aligned_contiguous)return "TV finite requires audited NumPy1.24.4 or2.5.4 profile";
    if(!parameters.reduction.buffer_elements)return "TV finite reduction buffer must be positive";
    if(std::fegetround()!=FE_TONEAREST)return "TV finite requires nearest rounding";
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
    if(_mm_getcsr()&((1u<<15)|(1u<<6)))return "TV finite requires gradual underflow";
#endif
    if(!planes(w.p,input,layout,spans[1]) || !planes(w.g,input,layout,spans[2]) ||
       !spatial(w.d,input,layout,spans[3]) || !spatial(w.out,input,layout,spans[4]) ||
       !spatial(w.squared,input,layout,spans[5]) || !spatial(w.norm,input,layout,spans[6]))return "TV finite requires matching aligned C workspaces";
    if(!alias(w.p,w.p_read)||!alias(w.g,w.g_read)||!alias(w.out,w.out_read)||!alias(w.norm,w.norm_read))return "TV finite readonly workspace aliases must match";
    std::size_t count=7;
    if(trace) {
        if(capacity<parameters.max_iterations || reinterpret_cast<std::uintptr_t>(trace)%alignof(TvDenoiseTrace) ||
           !interval(trace,parameters.max_iterations*sizeof(TvDenoiseTrace),spans[7]))return "TV finite trace must hold all iteration records";
        count=8;
    }
    for(std::size_t a=0;a<count;++a)for(std::size_t b=0;b<a;++b)
        if(intersects(spans[a],spans[b]))return "TV finite buffers and input must be disjoint";
    if(input.type==TvScalarType::float32?!bounded_input<float>(input,layout):!bounded_input<double>(input,layout))return "TV finite input must be finite with abs<=2^20";
    if(!zero_bytes(w.p.data,layout.n*layout.rank*layout.width) ||
       !zero_bytes(w.g.data,layout.n*layout.rank*layout.width) || !zero_bytes(w.d.data,layout.n*layout.width))return "TV finite p/g/d must start positive zero";
    return nullptr;
}

const char* detail::tv_denoise_try_finite_into(const TvConstView& input,
    const TvDenoiseWorkspace& workspace,const TvDenoiseParameters& parameters,
    TvDenoiseResult& result,TvDenoiseTrace* trace,std::size_t capacity) {
    if(const char* reason=tv_denoise_ineligibility(input,workspace,parameters,trace,capacity))return reason;
    if(input.type==TvScalarType::float64)result=run<double,double>(input,workspace,parameters,trace);
    else if(parameters.reduction.profile==TvReductionProfile::numpy_1244_buffered)result=run<float,double>(input,workspace,parameters,trace);
    else result=run<float,float>(input,workspace,parameters,trace);
    return nullptr;
}

TvDenoiseResult tv_denoise_finite_into(const TvConstView& input,const TvDenoiseWorkspace& workspace,
    const TvDenoiseParameters& parameters,TvDenoiseTrace* trace,std::size_t capacity) {
    TvDenoiseResult result{};
    if(const char* reason=detail::tv_denoise_try_finite_into(input,workspace,parameters,result,trace,capacity))throw std::invalid_argument(reason);
    return result;
}
} // namespace skimagexx
