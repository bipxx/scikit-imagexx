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
        if(std::isnan(value))return sizeof(T)==4 ? UINT32_MAX : UINT64_MAX;
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
    std::uint64_t high=0;
    bool dense=false;
    std::vector<std::uint64_t> hash_keys;
    std::vector<std::size_t> hash_positions;
};
std::size_t hash_slot(std::uint64_t key,const std::vector<std::uint64_t>& keys,const std::vector<std::size_t>& positions) {
    auto hash=key;hash^=hash>>33;hash*=UINT64_C(0xff51afd7ed558ccd);hash^=hash>>33;
    hash*=UINT64_C(0xc4ceb9fe1a85ec53);hash^=hash>>33;
    auto slot=std::size_t(hash)&(keys.size()-1);
    while(positions[slot] && keys[slot]!=key)slot=(slot+1)&(keys.size()-1);
    return slot;
}
template<class T,class Key> void radix_sort(std::vector<T>& values,int workers,Key key,int bytes=8) {
    if(values.size()<2048) {
        std::sort(values.begin(),values.end(),[&](const auto& a,const auto& b){return key(a)<key(b);});return;
    }
    std::vector<T> temporary(values.size());
    std::vector<std::size_t> counts(std::size_t(workers)*256),offsets(counts.size());
    std::vector<std::pair<std::size_t,std::size_t>> segments(std::size_t(workers),{0,0});
    for(int shift=0;shift<bytes*8;shift+=8) {
        std::fill(counts.begin(),counts.end(),0);
        std::fill(segments.begin(),segments.end(),std::pair<std::size_t,std::size_t>{0,0});
        detail::partition(values.size(),workers,[&](auto begin,auto end,auto worker) {
            segments[worker]={begin,end};auto* histogram=counts.data()+worker*256;
            for(auto i=begin;i<end;++i)++histogram[(key(values[i])>>shift)&255];
        });
        std::size_t position=0,occupied=0;
        for(std::size_t bin=0;bin<256;++bin) {
            const auto start=position;
            for(int worker=0;worker<workers;++worker) {
                const auto index=std::size_t(worker)*256+bin;
                offsets[index]=position;position+=counts[index];
            }
            occupied+=position!=start;
        }
        if(occupied<2)continue;
        // Scatter the recorded histogram segments, rather than assuming the
        // next OpenMP team has the same size (OMP_DYNAMIC may change it).
        detail::partition(std::size_t(workers),workers,[&](auto begin,auto end,auto) {
            for(auto segment=begin;segment<end;++segment) {
                auto* next=offsets.data()+segment*256;
                for(auto i=segments[segment].first;i<segments[segment].second;++i)
                    temporary[next[(key(values[i])>>shift)&255]++]=values[i];
            }
        });
        values.swap(temporary);
    }
}
Distribution distribution(const void* input,std::size_t count,std::size_t channels,std::size_t channel,
    PixelType type,bool source,bool require_nonnegative,int workers) {
    Distribution result;
    result.keys.resize(count);
    const auto read=reader(type);
    detail::partition(count,workers,[&](auto begin,auto end,auto) {
        auto i=begin;
#ifdef SKIMAGEXX_AVX2
        if(has_avx2())i+=simd::histogram_keys(static_cast<const unsigned char*>(input)+(begin*channels+channel)*pixel_size(type),
            result.keys.data()+begin,end-begin,type,channels);
#endif
        for(;i<end;++i)result.keys[i]=read(input,i*channels+channel);
    });
    if(!count)return result;
    const bool small=type==PixelType::boolean || type==PixelType::uint8 || type==PixelType::int8 || type==PixelType::uint16 || type==PixelType::int16;
    if(small && count>=4096) {
        result.low=type==PixelType::int8 ? sign64-128 : type==PixelType::int16 ? sign64-32768 : 0;
        result.high=type==PixelType::boolean ? 1 : type==PixelType::uint8 ? 255 : type==PixelType::int8 ? sign64+127 :
            type==PixelType::int16 ? sign64+32767 : 65535;
    } else {
        const auto bounds=std::minmax_element(result.keys.begin(),result.keys.end());
        result.low=*bounds.first;result.high=*bounds.second;
    }
    if(require_nonnegative && (type==PixelType::int8 || type==PixelType::int16 || type==PixelType::int32 || type==PixelType::int64)
            && *std::min_element(result.keys.begin(),result.keys.end())<sign64)
        throw std::invalid_argument("'list' argument must have no negative elements");
    const auto span=result.high-result.low;
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
    if(count>=4096) {
        result.hash_keys.resize(8192);result.hash_positions.assign(8192,0);
        bool complete=true;
        for(auto key:result.keys) {
            const auto slot=hash_slot(key,result.hash_keys,result.hash_positions);
            if(!result.hash_positions[slot]) {
                if(result.counts.size()==4096){complete=false;break;}
                result.hash_keys[slot]=key;result.hash_positions[slot]=result.counts.size()+1;
                result.counts.push_back(0);
            }
            ++result.counts[result.hash_positions[slot]-1];
        }
        if(complete) {
            std::vector<std::uint64_t> unique;
            for(std::size_t i=0;i<result.hash_keys.size();++i)if(result.hash_positions[i])unique.push_back(result.hash_keys[i]);
            std::sort(unique.begin(),unique.end());
            auto previous=std::move(result.counts);result.counts.reserve(unique.size());
            for(std::size_t i=0;i<unique.size();++i) {
                const auto slot=hash_slot(unique[i],result.hash_keys,result.hash_positions);
                result.counts.push_back(previous[result.hash_positions[slot]-1]);
                result.values.push_back(value_from_key(unique[i],type));result.hash_positions[slot]=i+1;
            }
            return result;
        }
        result.hash_keys.clear();result.hash_positions.clear();result.counts.clear();
    }
    if(source) {
        if(count<=UINT32_MAX && span<=UINT32_MAX) {
            std::vector<std::uint64_t> packed(count);
            for(std::size_t i=0;i<count;++i)packed[i]=((result.keys[i]-result.low)<<32)|std::uint64_t(i);
            std::vector<std::uint64_t>().swap(result.keys);
            radix_sort(packed,workers,[](auto value){return value>>32;},4);
            result.order.resize(count);
            for(std::size_t i=0;i<count;) {
                const auto start=i,key=packed[i]>>32;
                while(i<count && (packed[i]>>32)==key){result.order[i]=std::size_t(packed[i]&UINT32_MAX);++i;}
                result.counts.push_back(i-start);result.values.push_back(value_from_key(result.low+key,type));result.ends.push_back(i);
            }
        } else {
            struct Entry{std::uint64_t key;std::size_t index;};
            std::vector<Entry> entries(count);
            for(std::size_t i=0;i<count;++i)entries[i]={result.keys[i],i};
            std::vector<std::uint64_t>().swap(result.keys);
            radix_sort(entries,workers,[](const auto& value){return value.key;});
            result.order.resize(count);
            for(std::size_t i=0;i<count;) {
                const auto start=i,key=entries[i].key;
                while(i<count && entries[i].key==key){result.order[i]=entries[i].index;++i;}
                result.counts.push_back(i-start);result.values.push_back(value_from_key(key,type));result.ends.push_back(i);
            }
        }
    } else {
        radix_sort(result.keys,workers,[](auto key){return key;},result.high<=UINT32_MAX ? 4 : 8);
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
        unsigned mapped_notices=0;
        if(ot==PixelType::float32 && !half_round && !swap_output)for(auto value:mapped) {
            float converted;mapped_notices|=write(&converted,0,value,false,false);
        }
        if(a.dense) {
            std::vector<double> table(std::size_t(a.high-a.low)+1);
            for(std::size_t i=0;i<mapped.size();++i)table[a.ends[i]]=mapped[i];
            detail::partition(count,workers,[&](auto begin,auto end,auto worker) {
                unsigned local=0;auto i=begin;
#ifdef SKIMAGEXX_AVX2
                if(has_avx2() && !half_round && !swap_output && (ot==PixelType::float64 || (ot==st && ot==rt &&
                    (ot==PixelType::float32 || ot==PixelType::uint8 || ot==PixelType::uint16 || ot==PixelType::int8 || ot==PixelType::int16 || ot==PixelType::boolean))))
                    i+=simd::histogram_lookup(a.keys.data()+begin,static_cast<unsigned char*>(output)+(begin*channels+channel)*pixel_size(ot),
                        end-begin,a.low,table.data(),ot,channels);
#endif
                for(;i<end;++i)local|=write(output,i*channels+channel,table[std::size_t(a.keys[i]-a.low)],half_round,swap_output);
                notices[worker]=local;
            });
        } else if(!a.hash_positions.empty()) {
            detail::partition(count,workers,[&](auto begin,auto end,auto worker) {
                unsigned local=0;
                for(auto i=begin;i<end;++i) {
                    const auto slot=hash_slot(a.keys[i],a.hash_keys,a.hash_positions);
                    local|=write(output,i*channels+channel,mapped[a.hash_positions[slot]-1],half_round,swap_output);
                }
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
        unsigned channel_notice=mapped_notices;
        for(auto notice:notices)channel_notice|=notice;
        if(channel_flags)channel_flags[channel]=channel_notice;
        flags|=channel_notice;
    }
    return flags;
}
}
