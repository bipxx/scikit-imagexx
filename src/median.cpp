#include "skimagexx/median.hpp"
#include "execution.hpp"
#include "simd.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <type_traits>
namespace skimagexx { namespace {
std::size_t product(const std::vector<std::size_t>& shape) {
    std::size_t n=1;const auto limit=std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/sizeof(double);
    for(auto d:shape){if(d>limit || (d && n>limit/d))throw std::overflow_error("median dimensions overflow");n*=d;}return n;
}
std::vector<std::size_t> pitches(const std::vector<std::size_t>& shape) {
    std::vector<std::size_t> p(shape.size(),1);for(auto d=shape.size();d-->1;)p[d-1]=p[d]*shape[d];return p;
}
std::ptrdiff_t mapped(std::ptrdiff_t x,std::size_t length,GaussianBorder border) {
    const auto n=std::ptrdiff_t(length);if(x>=0 && x<n)return x;
    if(border==GaussianBorder::constant)return -1;
    if(border==GaussianBorder::nearest)return std::clamp(x,std::ptrdiff_t(0),n-1);
    if(border==GaussianBorder::wrap){x%=n;return x<0 ? x+n : x;}
    if(n==1)return 0;
    const auto period=border==GaussianBorder::reflect ? 2*n : 2*(n-1);
    x%=period;if(x<0)x+=period;
    return x<n ? x : border==GaussianBorder::reflect ? period-1-x : period-x;
}
std::int32_t truncate32(double x) {
    return x>=-2147483648. && x<2147483648. ? std::int32_t(x) : std::numeric_limits<std::int32_t>::min();
}
std::int64_t truncate64(double x) {
    return x>=-9223372036854775808. && x<9223372036854775808. ? std::int64_t(x) : std::numeric_limits<std::int64_t>::min();
}
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
template<class T> void save(void* output,std::size_t i,double value) {
    if constexpr(std::is_unsigned_v<T>)static_cast<T*>(output)[i]=value>-1 ? cast<T>(value) : T(T(0)-cast<T>(-value));
    else static_cast<T*>(output)[i]=cast<T>(value);
}
void write(void* output,std::size_t i,double value,PixelType type) {
    switch(type){
#define SAVE(tag,T) case PixelType::tag:save<T>(output,i,value);return;
    SAVE(boolean,std::uint8_t) SAVE(uint8,std::uint8_t) SAVE(int8,std::int8_t)
    SAVE(uint16,std::uint16_t) SAVE(int16,std::int16_t) SAVE(uint32,std::uint32_t) SAVE(int32,std::int32_t)
    SAVE(uint64,std::uint64_t) SAVE(int64,std::int64_t) SAVE(float32,float) SAVE(float64,double)
#undef SAVE
    }
}
template<class T> T select(T* values,std::size_t count,std::size_t rank) {
    // Hoare's first-pivot partition preserves the comparison/tie order used
    // by SciPy's N-D selector, including its unordered floating comparisons.
    std::ptrdiff_t low=0,high=std::ptrdiff_t(count)-1;
    while(low<high){const auto pivot=values[low];auto left=low-1,right=high+1;
        for(;;){do --right;while(values[right]>pivot);do ++left;while(values[left]<pivot);
            if(left>=right)break;std::swap(values[left],values[right]);}
        if(rank<=std::size_t(right))high=right;else low=right+1;
    }return values[rank];
}
template<class T> class RankWindow {
    std::vector<T> values;
    std::vector<std::size_t> lower,upper,position;
    std::vector<unsigned char> side;
    bool better(std::size_t a,std::size_t b,bool maximum) const {
        return maximum ? values[a]>values[b] : values[a]<values[b];
    }
    void exchange(std::vector<std::size_t>& heap,std::size_t a,std::size_t b) {
        std::swap(heap[a],heap[b]);position[heap[a]]=a;position[heap[b]]=b;
    }
    void repair(std::vector<std::size_t>& heap,std::size_t i,bool maximum) {
        if(i && better(heap[i],heap[(i-1)/2],maximum)) {
            do {const auto parent=(i-1)/2;if(!better(heap[i],heap[parent],maximum))break;exchange(heap,i,parent);i=parent;}while(i);
        } else {
            while(2*i+1<heap.size()){auto child=2*i+1;
                if(child+1<heap.size() && better(heap[child+1],heap[child],maximum))++child;
                if(!better(heap[child],heap[i],maximum))break;exchange(heap,i,child);i=child;
            }
        }
    }
public:
    RankWindow(const T* first,std::size_t count,std::size_t rank):values(first,first+count),position(count),side(count) {
        std::vector<std::size_t> indices(count);std::iota(indices.begin(),indices.end(),std::size_t(0));
        std::nth_element(indices.begin(),indices.begin()+rank,indices.end(),[&](auto a,auto b){return values[a]<values[b];});
        lower.assign(indices.begin(),indices.begin()+rank+1);upper.assign(indices.begin()+rank+1,indices.end());
        std::make_heap(lower.begin(),lower.end(),[&](auto a,auto b){return values[a]<values[b];});
        std::make_heap(upper.begin(),upper.end(),[&](auto a,auto b){return values[a]>values[b];});
        for(std::size_t i=0;i<lower.size();++i){position[lower[i]]=i;side[lower[i]]=0;}
        for(std::size_t i=0;i<upper.size();++i){position[upper[i]]=i;side[upper[i]]=1;}
    }
    void replace(std::size_t slot,T value) {
        values[slot]=value;
        if(side[slot])repair(upper,position[slot],false);else repair(lower,position[slot],true);
        if(!upper.empty() && values[lower[0]]>values[upper[0]]) {
            const auto a=lower[0],b=upper[0];lower[0]=b;upper[0]=a;
            position[a]=position[b]=0;side[a]=1;side[b]=0;
            repair(lower,0,true);repair(upper,0,false);
        }
    }
    T value() const {return values[lower[0]];}
};
template<class T,class W> MedianInfo run(const T* input,void* output,PixelType target,
    const std::vector<std::size_t>& shape,const std::vector<std::size_t>& window,
    const std::vector<std::vector<std::size_t>>& coordinates,std::size_t rank,GaussianBorder border,
    double cval,std::int64_t integer_cval,bool nd_rank,bool fast_1d,bool zero_strides,bool source_cval) {
    const auto n=product(shape);if(!n)return {};
    if(!input || !output)throw std::invalid_argument("nonnull median buffers required");
    W constant;
    if constexpr(std::is_same_v<W,std::int64_t>)constant=integer_cval;
    else constant=source_cval ? double(cast<T>(cval)) : cval;
    bool unordered=false,negative_zero=false;
    if constexpr(std::is_floating_point_v<T>){
        for(std::size_t i=0;i<n;++i){unordered|=std::isnan(input[i]);negative_zero|=input[i]==0 && std::signbit(input[i]);}
    }
    if constexpr(std::is_floating_point_v<W>){if(border==GaussianBorder::constant){unordered|=std::isnan(constant);negative_zero|=constant==0 && std::signbit(constant);}}
    if(fast_1d && unordered)return {true};
    const bool vectorize=!unordered && !negative_zero && !zero_strides && coordinates.size()<=32;
    auto padded_shape=shape;for(std::size_t d=0;d<shape.size();++d){if(window[d]-1>std::size_t(std::numeric_limits<std::ptrdiff_t>::max())-shape[d])throw std::overflow_error("median padding overflow");padded_shape[d]+=window[d]-1;}
    const auto padded_count=product(padded_shape);const auto ip=pitches(shape),pp=pitches(padded_shape);const auto workers=detail::active_workers(n);
    std::vector<W> padded(padded_count);
    const auto width=padded_shape.back(),left=window.back()/2,right=window.back()-left-1;
    std::vector<std::ptrdiff_t> before(left),after(right);
    for(std::size_t x=0;x<left;++x)before[x]=mapped(std::ptrdiff_t(x)-std::ptrdiff_t(left),shape.back(),border);
    for(std::size_t x=0;x<right;++x)after[x]=mapped(std::ptrdiff_t(shape.back()+x),shape.back(),border);
    detail::partition(padded_count/width,workers,[&](auto begin,auto end,auto){
        for(auto row=begin;row<end;++row){auto remainder=row*width;std::size_t base=0;bool missing=false;
            for(std::size_t d=0;d+1<shape.size();++d){const auto x=mapped(std::ptrdiff_t(remainder/pp[d])-std::ptrdiff_t(window[d]/2),shape[d],border);remainder%=pp[d];if(x<0){missing=true;break;}base+=std::size_t(x)*ip[d];}
            auto* dst=padded.data()+row*width;
            if(missing){std::fill_n(dst,width,constant);continue;}
            for(std::size_t x=0;x<left;++x)dst[x]=before[x]<0 ? constant : W(input[base+std::size_t(before[x])]);
            if constexpr(std::is_same_v<T,W>)std::copy_n(input+base,shape.back(),dst+left);
            else for(std::size_t x=0;x<shape.back();++x)dst[left+x]=W(input[base+x]);
            for(std::size_t x=0;x<right;++x)dst[left+shape.back()+x]=after[x]<0 ? constant : W(input[base+std::size_t(after[x])]);
        }
    });
    std::vector<std::size_t> offsets(coordinates.size(),0);
    for(std::size_t t=0;t<coordinates.size();++t)for(std::size_t d=0;d<shape.size();++d)offsets[t]+=coordinates[t][d]*pp[d];
    bool dense_line=fast_1d && shape.size()==1 && coordinates.size()==window[0] && coordinates.size()>9;
    if(dense_line)for(std::size_t t=0;t<offsets.size();++t)if(offsets[t]!=t){dense_line=false;break;}
    if(dense_line) {
        // The rank partition retains k elements in fixed slots. Updating one
        // slot repairs only its heap and, if necessary, exchanges the roots.
        detail::partition(n,workers,[&](auto begin,auto end,auto){
            if(begin==end)return;
            RankWindow<W> state(padded.data()+begin,window[0],rank);
            for(auto i=begin;i<end;++i){
                if(i!=begin)state.replace((i-begin-1)%window[0],padded[i+window[0]-1]);
                if constexpr(std::is_same_v<W,std::int64_t>)static_cast<std::int64_t*>(output)[i]=state.value();
                else write(output,i,double(state.value()),target);
            }
        });return {};
    }
    detail::partition(n,workers,[&](auto begin,auto end,auto){
        std::vector<W> values(coordinates.size()),selected(std::min(std::size_t(1024),end-begin));
        for(auto position=begin;position<end;){const auto span=std::min({end-position,shape.back()-position%shape.back(),selected.size()});
            auto remainder=position;std::size_t base=0;for(std::size_t d=0;d<shape.size();++d){base+=(remainder/ip[d])*pp[d];remainder%=ip[d];}
            std::size_t done=0;
#ifdef SKIMAGEXX_AVX2
            if(vectorize && has_avx2())done=simd::median_small(padded.data()+base,selected.data(),span,offsets.data(),offsets.size(),rank,std::is_same_v<W,std::int64_t>);
#endif
            for(auto k=done;k<span;++k){
                if(zero_strides){for(std::size_t t=0;t<coordinates.size();++t){auto i=position+k;bool missing=false;
                    for(std::size_t d=0;d<shape.size();++d){const auto x=i/ip[d];i%=ip[d];const auto radius=window[d]/2,tail=window[d]-radius-1;
                        const auto representative=window[d]>=shape[d] || x<radius || x>=shape[d]-tail ? x : radius;
                        const auto c=mapped(std::ptrdiff_t(representative)+std::ptrdiff_t(coordinates[t][d])-std::ptrdiff_t(radius),shape[d],border);
                        if(c<0 || c==1){missing=true;break;}}
                    values[t]=missing ? constant : W(input[0]);
                }} else for(std::size_t t=0;t<offsets.size();++t)values[t]=padded[base+k+offsets[t]];
                selected[k]=select(values.data(),values.size(),rank);
            }
            for(std::size_t k=0;k<span;++k){
                if constexpr(std::is_same_v<W,std::int64_t>)static_cast<std::int64_t*>(output)[position+k]=selected[k];
                else write(output,position+k,nd_rank ? double(cast<T>(selected[k])) : double(selected[k]),target);
            }
            position+=span;
        }
    });return {};
}
}}
namespace skimagexx {
MedianInfo median_filter_into(const void* input,void* output,PixelType source,PixelType target,
    const std::vector<std::size_t>& shape,const std::vector<std::size_t>& window,
    const std::vector<std::vector<std::size_t>>& coordinates,std::size_t rank,GaussianBorder border,
    double cval,std::int64_t integer_cval,bool nd_rank,bool fast_1d,bool zero_strides,bool source_cval) {
    if(shape.empty() || shape.size()!=window.size() || coordinates.empty() || rank>=coordinates.size() || unsigned(border)>4)throw std::invalid_argument("invalid median geometry");
    product(shape);product(window);for(auto w:window)if(!w)throw std::invalid_argument("empty median window");
    for(const auto& c:coordinates){if(c.size()!=shape.size())throw std::invalid_argument("median coordinate rank mismatch");for(std::size_t d=0;d<c.size();++d)if(c[d]>=window[d])throw std::invalid_argument("median coordinate outside window");}
    if(fast_1d && source==PixelType::int64){if(target!=source)throw std::invalid_argument("exact int64 median target required");return run<std::int64_t,std::int64_t>(static_cast<const std::int64_t*>(input),output,target,shape,window,coordinates,rank,border,cval,integer_cval,false,true,false,false);}
    switch(source){
#define RUN(tag,T) case PixelType::tag:return run<T,double>(static_cast<const T*>(input),output,target,shape,window,coordinates,rank,border,cval,integer_cval,nd_rank,fast_1d,zero_strides,source_cval);
    RUN(boolean,std::uint8_t) RUN(uint8,std::uint8_t) RUN(int8,std::int8_t) RUN(uint16,std::uint16_t) RUN(int16,std::int16_t)
    RUN(uint32,std::uint32_t) RUN(int32,std::int32_t) RUN(uint64,std::uint64_t) RUN(int64,std::int64_t) RUN(float32,float) RUN(float64,double)
#undef RUN
    }
    throw std::invalid_argument("unsupported median input");
}
}
