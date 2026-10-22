#include "skimagexx/histogram_matching.hpp"
#include "skimagexx/image.hpp"
#include "execution.hpp"
#include "half.hpp"
#include "simd.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace skimagexx {
namespace {
constexpr auto sign64=UINT64_C(1)<<63;
template<class T> std::uint64_t read_key(const void* p,std::size_t i) {
    const auto value=static_cast<const T*>(p)[i];
    if constexpr(std::is_floating_point_v<T>) {
        if(std::isnan(value))return UINT64_MAX;
        using U=std::conditional_t<sizeof(T)==4,std::uint32_t,std::uint64_t>;
        U bits;std::memcpy(&bits,&value,sizeof(value));
        if(value==0)bits=0;
        constexpr U sign=U(1)<<(8*sizeof(T)-1);
        return bits&sign ? U(~bits) : U(bits^sign);
    } else if constexpr(std::is_signed_v<T>)return std::uint64_t(std::int64_t(value))^sign64;
    else return std::uint64_t(value);
}
std::uint64_t read_bool(const void* p,std::size_t i){return static_cast<const unsigned char*>(p)[i]!=0;}
using KeyReader=std::uint64_t(*)(const void*,std::size_t);
KeyReader reader(PixelType type) {
    switch(type) {
    case PixelType::boolean:return read_bool;
#define KEY(tag,T) case PixelType::tag:return read_key<T>;
    KEY(uint8,std::uint8_t) KEY(int8,std::int8_t) KEY(uint16,std::uint16_t) KEY(int16,std::int16_t)
    KEY(uint32,std::uint32_t) KEY(int32,std::int32_t) KEY(uint64,std::uint64_t) KEY(int64,std::int64_t)
    KEY(float32,float) KEY(float64,double)
#undef KEY
    default:throw std::invalid_argument("unsupported histogram-matching dtype");
    }
}
double value_from_key(std::uint64_t key,PixelType type) {
    if(type==PixelType::float32 || type==PixelType::float64) {
        if(key==UINT64_MAX)return std::numeric_limits<double>::quiet_NaN();
        if(type==PixelType::float32) {
            auto bits=std::uint32_t(key);bits=bits&0x80000000U ? bits^0x80000000U : ~bits;
            float value;std::memcpy(&value,&bits,4);return value;
        }
        auto bits=key&sign64 ? key^sign64 : ~key;double value;std::memcpy(&value,&bits,8);return value;
    }
    if(type==PixelType::int8 || type==PixelType::int16 || type==PixelType::int32 || type==PixelType::int64) {
        key^=sign64;std::int64_t value;std::memcpy(&value,&key,8);return double(value);
    }
    return double(key);
}
struct Distribution {
    std::vector<std::uint64_t> keys;
    std::vector<std::size_t> counts;
    std::vector<double> values;
    std::vector<std::size_t> order;
    std::vector<std::size_t> ends;
    std::uint64_t low=0;
    bool dense=false;
};
Distribution distribution(const void* input,std::size_t count,std::size_t channels,std::size_t channel,
    PixelType type,bool source,bool require_nonnegative,int workers) {
    Distribution result;
    result.keys.resize(count);
    const auto read=reader(type);
    detail::partition(count,workers,[&](auto begin,auto end,auto) {
        for(auto i=begin;i<end;++i)result.keys[i]=read(input,i*channels+channel);
    });
    if(!count)return result;
    const auto bounds=std::minmax_element(result.keys.begin(),result.keys.end());
    if(require_nonnegative && value_from_key(*bounds.first,type)<0)
        throw std::invalid_argument("'list' argument must have no negative elements");
    result.low=*bounds.first;
    const auto span=*bounds.second-result.low;
    // Bounded direct tables cover quantized floating data as well as integers.
    // Key spans for most floating images are large, selecting the sorted path.
    if(span<65536) {
        result.dense=true;
        std::vector<std::size_t> histogram(std::size_t(span)+1);
        for(auto key:result.keys)++histogram[std::size_t(key-result.low)];
        for(std::size_t i=0;i<histogram.size();++i)if(histogram[i]) {
            result.counts.push_back(histogram[i]);result.values.push_back(value_from_key(result.low+i,type));
            result.ends.push_back(i);
        }
        return result;
    }
    if(source) {
        result.order.resize(count);std::iota(result.order.begin(),result.order.end(),0);
        std::sort(result.order.begin(),result.order.end(),[&](auto a,auto b){return result.keys[a]<result.keys[b];});
        for(std::size_t i=0;i<count;) {
            const auto start=i,key=result.keys[result.order[i]];
            while(i<count && result.keys[result.order[i]]==key)++i;
            result.counts.push_back(i-start);result.values.push_back(value_from_key(key,type));result.ends.push_back(i);
        }
    } else {
        std::sort(result.keys.begin(),result.keys.end());
        for(std::size_t i=0;i<count;) {
            const auto start=i,key=result.keys[i];
            while(i<count && result.keys[i]==key)++i;
            result.counts.push_back(i-start);result.values.push_back(value_from_key(key,type));
        }
    }
    return result;
}

// Direct binary64 -> binary16 rounding avoids a binary32 double-rounding step.
std::uint16_t half64(double value) {
    std::uint64_t bits;std::memcpy(&bits,&value,8);
    const auto sign=std::uint16_t((bits>>48)&0x8000);
    const auto exponent=int((bits>>52)&2047);
    const auto fraction=bits&UINT64_C(0xfffffffffffff);
    if(exponent==2047)return std::uint16_t(sign|0x7c00|(fraction ? 0x200 : 0));
    if(exponent>1038)return std::uint16_t(sign|0x7c00);
    if(exponent<998)return sign;
    const auto mantissa=fraction|UINT64_C(0x10000000000000);
    const int shift=exponent<1009 ? 1051-exponent : 42;
    const auto truncated=mantissa>>shift,rest=mantissa&((UINT64_C(1)<<shift)-1),tie=UINT64_C(1)<<(shift-1);
    auto rounded=truncated+std::uint64_t(rest>tie || (rest==tie && (truncated&1)));
    if(exponent>=1009)rounded+=std::uint64_t(exponent-1009)*1024;
    return std::uint16_t(sign|std::uint16_t(rounded));
}
template<class T> unsigned store(void* output,std::size_t index,double value,bool half,bool swap) {
    unsigned flags=0;T result;
    if constexpr(std::is_floating_point_v<T>) {
        result=T(half ? detail::decode_half(half64(value)) : value);
        if(std::isfinite(value)) {
            if(std::isinf(result))flags|=2;
            const double normal=half ? 0x1p-14 : double(std::numeric_limits<T>::min());
            if(value!=0 && std::abs(value)<normal && double(result)!=value)flags|=4;
        }
    } else if constexpr(std::is_same_v<T,bool>)result=value!=0;
    else {
        // NumPy's small integer casts use a signed native conversion followed
        // by narrowing. Bound the conversion explicitly to avoid C++ UB.
        using Intermediate=std::conditional_t<(sizeof(T)<=4 && std::is_signed_v<T>) || sizeof(T)<4,std::int32_t,std::int64_t>;
        constexpr double limit=sizeof(Intermediate)==4 ? 2147483648. : 9223372036854775808.;
        Intermediate integer;
        if(!std::isfinite(value) || value < -limit || value>=limit) {
            integer=std::numeric_limits<Intermediate>::min();flags|=1;
        } else integer=Intermediate(value);
        using U=std::make_unsigned_t<T>;
        const U narrowed=U(integer);std::memcpy(&result,&narrowed,sizeof(T));
    }
    auto* target=static_cast<unsigned char*>(output)+index*sizeof(T);
    if(!swap || sizeof(T)==1)std::memcpy(target,&result,sizeof(T));
    else {
        const auto* bytes=reinterpret_cast<const unsigned char*>(&result);
        std::reverse_copy(bytes,bytes+sizeof(T),target);
    }
    return flags;
}
using Writer=unsigned(*)(void*,std::size_t,double,bool,bool);
Writer writer(PixelType type) {
    switch(type) {
#define WRITE(tag,T) case PixelType::tag:return store<T>;
    WRITE(boolean,bool) WRITE(uint8,std::uint8_t) WRITE(int8,std::int8_t) WRITE(uint16,std::uint16_t) WRITE(int16,std::int16_t)
    WRITE(uint32,std::uint32_t) WRITE(int32,std::int32_t) WRITE(uint64,std::uint64_t) WRITE(int64,std::int64_t)
    WRITE(float32,float) WRITE(float64,double)
#undef WRITE
    default:throw std::invalid_argument("unsupported histogram-matching output type");
    }
}
}
unsigned match_histograms_into(const void* source,const void* reference,void* output,std::size_t count,std::size_t reference_count,
    std::size_t channels,PixelType st,PixelType rt,PixelType ot,bool unsigned_source,bool half_round,bool swap_output,unsigned* channel_flags) {
    (void)reader(st);(void)reader(rt);const auto write=writer(ot);
    const auto limit=std::size_t(std::numeric_limits<std::ptrdiff_t>::max());
    const auto bytes=std::max({pixel_size(st),pixel_size(rt),pixel_size(ot)});
    if(channels && (channels>limit/bytes || std::max(count,reference_count)>limit/(channels*bytes)))
        throw std::overflow_error("histogram-matching dimensions overflow");
    if(!channels || !count)return 0;
    if(!reference_count)throw std::invalid_argument("array of sample points is empty");
    if(!source || !reference || !output)throw std::invalid_argument("histogram-matching buffers must be nonnull");
    const int workers=detail::active_workers(std::max(count,reference_count));
    unsigned flags=0;
    for(std::size_t channel=0;channel<channels;++channel) {
        auto a=distribution(source,count,channels,channel,st,true,false,workers);
        auto b=distribution(reference,reference_count,channels,channel,rt,false,unsigned_source,workers);
        std::vector<double> cdf(b.counts.size());std::size_t total=0;
        for(std::size_t i=0;i<cdf.size();++i){total+=b.counts[i];cdf[i]=double(total)/double(reference_count);}
        std::vector<double> mapped(a.counts.size());std::size_t position=0;total=0;
        for(std::size_t i=0;i<mapped.size();++i) {
            total+=a.counts[i];const double q=double(total)/double(count);
            while(position+1<cdf.size() && q>=cdf[position+1])++position;
            if(q<=cdf.front())mapped[i]=b.values.front();
            else if(position+1==cdf.size() || q==cdf[position])mapped[i]=b.values[position];
            else {
                const auto slope=(b.values[position+1]-b.values[position])/(cdf[position+1]-cdf[position]);
                auto value=slope*(q-cdf[position])+b.values[position];
                if(std::isnan(value)) {
                    value=slope*(q-cdf[position+1])+b.values[position+1];
                    if(std::isnan(value) && b.values[position]==b.values[position+1])value=b.values[position];
                }
                mapped[i]=value;
            }
        }
        std::vector<unsigned> notices(std::size_t(workers),0);
        if(a.dense) {
            std::vector<double> table(std::size_t(*std::max_element(a.keys.begin(),a.keys.end())-a.low)+1);
            for(std::size_t i=0;i<mapped.size();++i)table[a.ends[i]]=mapped[i];
            detail::partition(count,workers,[&](auto begin,auto end,auto worker) {
                unsigned local=0;for(auto i=begin;i<end;++i)local|=write(output,i*channels+channel,table[std::size_t(a.keys[i]-a.low)],half_round,swap_output);
                notices[worker]=local;
            });
        } else {
            detail::partition(mapped.size(),workers,[&](auto begin,auto end,auto worker) {
                unsigned local=0;
                for(auto group=begin;group<end;++group)for(auto i=group ? a.ends[group-1] : 0;i<a.ends[group];++i)
                    local|=write(output,a.order[i]*channels+channel,mapped[group],half_round,swap_output);
                notices[worker]=local;
            });
        }
        unsigned channel_notice=0;
        for(auto notice:notices)channel_notice|=notice;
        if(channel_flags)channel_flags[channel]=channel_notice;
        flags|=channel_notice;
    }
    return flags;
}
}
