#include "skimagexx/contrast.hpp"
#include "execution.hpp"
#include "half.hpp"
#include "simd.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace skimagexx {
namespace {
struct Axis {std::size_t size;std::ptrdiff_t stride;};
struct Plan {
    const unsigned char* origin=nullptr;
    std::vector<Axis> axes;
    std::size_t count=1,repeat=1,total=1;
    bool dense=true;
    const unsigned char* at(std::size_t index) const {
        auto pointer=origin;
        for(std::size_t j=axes.size();j--;) {
            const auto& axis=axes[j];
            pointer+=std::ptrdiff_t(index%axis.size)*axis.stride;
            index/=axis.size;
        }
        return pointer;
    }
};
Plan plan(const ContrastView& view) {
    if(view.shape.size()!=view.strides.size())throw std::invalid_argument("one contrast stride per axis required");
    if(view.half && view.type!=PixelType::uint16)throw std::invalid_argument("binary16 contrast storage required");
    const auto bytes=pixel_size(view.type),limit=std::size_t(std::numeric_limits<std::ptrdiff_t>::max());
    Plan p;p.origin=static_cast<const unsigned char*>(view.data);
    std::size_t positive=bytes-1,negative=0;
    for(std::size_t j=0;j<view.shape.size();++j) {
        const auto length=view.shape[j];const auto step=view.strides[j];
        if(!length)throw std::invalid_argument("nonempty contrast input required");
        if(length>limit/p.total)throw std::overflow_error("contrast dimensions overflow");
        p.total*=length;
        if(length<=1)continue;
        if(step==std::numeric_limits<std::ptrdiff_t>::min())throw std::overflow_error("contrast stride overflow");
        const auto magnitude=std::size_t(step<0 ? -step : step);
        auto& span=step<0 ? negative : positive;
        if(magnitude>(limit-span)/(length-1))throw std::overflow_error("contrast input span overflow");
        span+=magnitude*(length-1);
        if(!step)p.repeat*=length;
        else {p.axes.push_back({length,step});p.count*=length;}
    }
    if(!p.origin)throw std::invalid_argument("nonnull contrast input required");
    if(p.count>limit/(view.half ? sizeof(float) : bytes))throw std::overflow_error("contrast working storage overflow");
    auto magnitude=[](const Axis& axis){return std::size_t(axis.stride<0 ? -axis.stride : axis.stride);};
    std::sort(p.axes.begin(),p.axes.end(),[&](const auto& a,const auto& b){return magnitude(a)>magnitude(b);});
    std::size_t extent=bytes;
    for(std::size_t j=p.axes.size();j--;) {
        if(magnitude(p.axes[j])!=extent)p.dense=false;
        extent*=p.axes[j].size;
    }
    if(p.dense)p.origin-=negative;
    return p;
}

template<class T> T read(const unsigned char* pointer,bool swap) {
    T result;
    if(swap && sizeof(T)>1) {
        unsigned char bytes[sizeof(T)];
        for(std::size_t i=0;i<sizeof(T);++i)bytes[i]=pointer[sizeof(T)-1-i];
        std::memcpy(&result,bytes,sizeof(T));
    } else std::memcpy(&result,pointer,sizeof(T));
    return result;
}

struct Query {std::size_t lower,upper;double weight;};
std::array<Query,2> queries(const Plan& p,const std::array<double,2>& positions) {
    std::array<Query,2> output;
    for(std::size_t j=0;j<2;++j) {
        const auto position=positions[j];
        if(!std::isfinite(position) || position<0 || position>double(p.total-1))
            throw std::invalid_argument("contrast rank outside input");
        if(position>=double(p.total-1))output[j]={p.count-1,p.count-1,0};
        else {
            const auto lower=std::size_t(std::floor(position));
            output[j]={lower/p.repeat,(lower+1)/p.repeat,position-std::floor(position)};
        }
    }
    return output;
}

template<class T> double difference(T lower,T upper) {
    if constexpr(std::is_integral_v<T>) {
        using Word=std::make_unsigned_t<T>;
        const auto bits=Word(Word(upper)-Word(lower));
        T result;std::memcpy(&result,&bits,sizeof(T));return double(result);
    } else return double(T(upper-lower));
}

template<class T> void select(std::vector<T>& data,const std::vector<std::size_t>& ranks,
                             std::size_t first,std::size_t last,std::size_t begin,std::size_t end) {
    if(first==last)return;
    const auto middle=first+(last-first)/2,rank=ranks[middle];
    auto start=data.begin()+std::ptrdiff_t(begin),stop=data.begin()+std::ptrdiff_t(end),pivot=data.begin()+std::ptrdiff_t(rank);
    std::nth_element(start,pivot,stop);
    select(data,ranks,first,middle,begin,rank);
    select(data,ranks,middle+1,last,rank+1,end);
}

template<class Input,class Value=Input,bool Half=false> bool run(const ContrastView& view,const Plan& p,
    const std::array<Query,2>& wanted,std::array<double,2>& output) {
    std::vector<Value> values(p.count);
    const auto workers=detail::active_workers(p.count);std::vector<unsigned char> valid(std::size_t(workers),1);
    detail::partition(p.count,workers,[&](auto begin,auto end,auto worker) {
        std::size_t done=begin;
        if(p.dense) {
            if constexpr(Half) {
#ifdef SKIMAGEXX_AVX2
                if(has_avx2() && !view.swap)done+=simd::edge_half(reinterpret_cast<const std::uint16_t*>(p.origin)+begin,values.data()+begin,end-begin);
#endif
            } else if(view.swap && sizeof(Input)>1) {
#ifdef SKIMAGEXX_AVX2
                if(has_avx2())done+=simd::edge_swap(p.origin+begin*sizeof(Input),reinterpret_cast<unsigned char*>(values.data()+begin),end-begin,sizeof(Input));
#endif
            } else {
                std::memcpy(values.data()+begin,p.origin+begin*sizeof(Input),(end-begin)*sizeof(Input));done=end;
            }
        }
        for(auto i=done;i<end;++i) {
            const auto* pointer=p.dense ? p.origin+i*sizeof(Input) : p.at(i);
            const auto value=read<Input>(pointer,view.swap);
            if constexpr(Half)values[i]=detail::decode_half(std::uint16_t(value));
            else values[i]=value;
        }
        if constexpr(std::is_floating_point_v<Value>) {
#ifdef SKIMAGEXX_AVX2
            if(has_avx2()) {
                if(!simd::contrast_finite(values.data()+begin,end-begin,sizeof(Value)==4))valid[worker]=0;
                return;
            }
#endif
            for(auto i=begin;i<end;++i)if(!std::isfinite(values[i])) {valid[worker]=0;break;}
        }
    });
    if(std::find(valid.begin(),valid.end(),0)!=valid.end())return false;
    std::vector<std::size_t> ranks;
    for(const auto& q:wanted) {ranks.push_back(q.lower);ranks.push_back(q.upper);}
    std::sort(ranks.begin(),ranks.end());ranks.erase(std::unique(ranks.begin(),ranks.end()),ranks.end());
    select(values,ranks,0,ranks.size(),0,values.size());
    for(std::size_t j=0;j<2;++j) {
        const auto& q=wanted[j];const auto lower=values[q.lower],upper=values[q.upper];
        double delta=difference(lower,upper);
        if constexpr(Half)delta=detail::decode_half(detail::encode_half(float(delta)));
        if(!std::isfinite(delta))return false;
        output[j]=q.weight>=.5 ? double(upper)-delta*(1-q.weight) : double(lower)+delta*q.weight;
        if(!std::isfinite(output[j]))return false;
    }
    return true;
}
}

bool contrast_boolean_constant(const ContrastView& view) {
    if(view.type!=PixelType::boolean || view.half)throw std::invalid_argument("boolean contrast storage required");
    const auto p=plan(view);const auto workers=detail::active_workers(p.count);
    std::vector<unsigned char> flags(std::size_t(workers),0);
    detail::partition(p.count,workers,[&](auto begin,auto end,auto worker) {
        unsigned char found=0;
#ifdef SKIMAGEXX_AVX2
        if(p.dense && has_avx2()) {
            flags[worker]=simd::contrast_truth(p.origin+begin,end-begin);return;
        }
#endif
        for(auto i=begin;i<end && found!=3;++i)found|=*(p.dense ? p.origin+i : p.at(i)) ? 1 : 2;
        flags[worker]=found;
    });
    unsigned char found=0;for(auto value:flags)found|=value;return found!=3;
}

bool contrast_percentiles(const ContrastView& view,const std::array<double,2>& positions,std::array<double,2>& output) {
    const auto p=plan(view);const auto wanted=queries(p,positions);
    if(view.half)return run<std::uint16_t,float,true>(view,p,wanted,output);
    switch(view.type) {
    case PixelType::uint8:return run<std::uint8_t>(view,p,wanted,output);
    case PixelType::int8:return run<std::int8_t>(view,p,wanted,output);
    case PixelType::uint16:return run<std::uint16_t>(view,p,wanted,output);
    case PixelType::int16:return run<std::int16_t>(view,p,wanted,output);
    case PixelType::uint32:return run<std::uint32_t>(view,p,wanted,output);
    case PixelType::int32:return run<std::int32_t>(view,p,wanted,output);
    case PixelType::uint64:return run<std::uint64_t>(view,p,wanted,output);
    case PixelType::int64:return run<std::int64_t>(view,p,wanted,output);
    case PixelType::float32:return run<float>(view,p,wanted,output);
    case PixelType::float64:return run<double>(view,p,wanted,output);
    case PixelType::boolean:throw std::invalid_argument("real percentile storage required");
    }
    throw std::invalid_argument("unsupported contrast storage");
}
}
