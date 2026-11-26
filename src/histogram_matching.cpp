#include "skimagexx/histogram_matching.hpp"
#include "skimagexx/rank_order.hpp"
#include "skimagexx/image.hpp"
#include "execution.hpp"
#include "half.hpp"
#include "simd.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace skimagexx {
#ifdef SKIMAGEXX_AVX2
namespace simd {
std::size_t histogram_keys(const void*,std::uint64_t*,std::size_t,PixelType,std::size_t,unsigned&);
void histogram_key_bounds(const std::uint64_t*,std::size_t,std::uint64_t&,std::uint64_t&);
std::size_t rank_dense_labels(const void*,void*,std::size_t,std::uint64_t,const std::size_t*,PixelType,std::size_t);
}
#endif
namespace {
constexpr auto sign64=UINT64_C(1)<<63;
template<class T,bool DetectRankEdges=false>
std::uint64_t read_key(const void* p,std::ptrdiff_t i,unsigned* flags=nullptr) {
    const auto value=static_cast<const T*>(p)[i];
    if constexpr(std::is_floating_point_v<T>) {
        using U=std::conditional_t<sizeof(T)==4,std::uint32_t,std::uint64_t>;
        constexpr U sign=U(1)<<(8*sizeof(T)-1);
        U bits;
        std::memcpy(&bits,&value,sizeof(bits));
        if constexpr(DetectRankEdges) {
            constexpr U infinity=sizeof(T)==4 ? U(0x7f800000) : U(UINT64_C(0x7ff0000000000000));
            const auto magnitude=bits&(sign-1);
            if(magnitude>infinity) {
                *flags|=1;
                return std::numeric_limits<U>::max();
            }
            if(!magnitude) {
                *flags|=bits&sign ? 2u : 4u;
                bits=0;
            }
        } else {
            if(std::isnan(value))return std::numeric_limits<U>::max();
            if(value==0)bits=0;
        }
        return bits&sign ? U(~bits) : U(bits^sign);
    } else if constexpr(std::is_signed_v<T>)return std::uint64_t(std::int64_t(value))^sign64;
    else return std::uint64_t(value);
}
std::uint64_t read_bool(const void* p,std::ptrdiff_t i){return static_cast<const unsigned char*>(p)[i]!=0;}
using KeyReader=std::uint64_t(*)(const void*,std::ptrdiff_t);
KeyReader reader(PixelType type) {
    switch(type) {
    case PixelType::boolean:return read_bool;
#define KEY(tag,T) case PixelType::tag:return [](const void* p,std::ptrdiff_t i){return read_key<T>(p,i);};
    KEY(uint8,std::uint8_t) KEY(int8,std::int8_t) KEY(uint16,std::uint16_t) KEY(int16,std::int16_t)
    KEY(uint32,std::uint32_t) KEY(int32,std::int32_t) KEY(uint64,std::uint64_t) KEY(int64,std::int64_t)
    KEY(float32,float) KEY(float64,double)
#undef KEY
    default:throw std::invalid_argument("unsupported histogram-matching dtype");
    }
}
// Three naturally aligned words keep the record compact on both endian orders.
struct Packed {
    std::array<std::uint32_t,2> words;
    std::uint32_t position;
    std::uint64_t key() const {
        std::uint64_t value;
        std::memcpy(&value,words.data(),sizeof(value));
        return value;
    }
};
static_assert(sizeof(Packed)==12);
struct WideEntry {
    std::uint64_t bits;
    std::size_t position;
    std::uint64_t key() const {return bits;}
};
struct CompactEntry {
    std::uint32_t bits;
    std::uint32_t position;
    std::uint64_t key() const {return bits;}
};
static_assert(sizeof(CompactEntry)==8);
bool compact_keys(PixelType type) {
    return type==PixelType::float32 || type==PixelType::uint32 || type==PixelType::uint16 ||
        type==PixelType::uint8 || type==PixelType::boolean;
}
template<class Entry> Entry indexed_key(std::uint64_t key,std::size_t position) {
    if constexpr(std::is_same_v<Entry,Packed>) {
        Packed entry;
        std::memcpy(entry.words.data(),&key,sizeof(key));
        entry.position=std::uint32_t(position);
        return entry;
    } else return {decltype(Entry::bits)(key),decltype(Entry::position)(position)};
}
constexpr auto entry_key=[](const auto& entry){return entry.key();};
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
    std::vector<std::uint16_t> slots;
    std::vector<std::size_t> hash_positions;
};

template<class T,class Visit>
void visit_unsigned(const T* input,std::ptrdiff_t stride,std::size_t begin,std::size_t end,Visit visit) {
    if(stride==1) {
        for(auto i=begin;i<end;++i)visit(i,input[i]);
    } else {
        for(auto i=begin;i<end;++i)visit(i,input[std::ptrdiff_t(i)*stride]);
    }
}

template<class T>
Distribution unsigned_distribution(const T* input,std::size_t count,std::ptrdiff_t stride,
    bool source,bool values,int workers) {
    Distribution result;
    result.dense=true;
    result.high=std::numeric_limits<T>::max();
    if constexpr(sizeof(T)==2) {
        // Avoid clearing and scanning the full uint16 domain for small images
        // with a narrow range, without allocating one key per pixel.
        if(count<4096) {
            T low=input[0],high=low;
            visit_unsigned(input,stride,1,count,[&](auto,T value) {
                low=std::min(low,value);high=std::max(high,value);
            });
            result.low=low;result.high=high;
        }
    }
    const auto bins=std::size_t(result.high-result.low)+1;
    const auto low=std::size_t(result.low);
    // Bound the replicated histograms by the number of input samples.
    const auto teams=std::min(std::size_t(workers),std::max(std::size_t(1),count/bins));
    const auto lanes=count/teams/bins>=8 ? std::size_t(4) : std::size_t(1);
    std::array<std::size_t,sizeof(T)==1 ? 4*256 : 0> small{};
    std::vector<std::size_t> large;
    auto* histogram=small.data();
    if(lanes*bins>small.size() || teams>1) {
        large.resize(teams*lanes*bins);
        histogram=large.data();
    }
    detail::partition(count,int(teams),[&](auto begin,auto end,auto worker) {
        auto* local=histogram+worker*lanes*bins;
        const auto lane_stride=lanes==4 ? bins : 0;
        const auto accumulate=[&](auto step) {
            for(;end-begin>=4;begin+=4) {
                const auto* p=input+std::ptrdiff_t(begin)*step;
                ++local[p[0]-low];
                ++local[lane_stride+p[step]-low];
                ++local[2*lane_stride+p[2*step]-low];
                ++local[3*lane_stride+p[3*step]-low];
            }
        };
        if(stride==1)accumulate(std::integral_constant<std::ptrdiff_t,1>{});
        else accumulate(stride);
        visit_unsigned(input,stride,begin,end,[&](auto,T value){++local[value-low];});
    });
    for(std::size_t lane=1;lane<teams*lanes;++lane)
        for(std::size_t i=0;i<bins;++i)histogram[i]+=histogram[lane*bins+i];

    const auto capacity=std::min(count,bins);
    result.counts.reserve(capacity);
    if(source)result.ends.reserve(capacity);
    else if(values)result.values.reserve(capacity);
    for(std::size_t i=0;i<bins;++i)if(histogram[i]) {
        result.counts.push_back(histogram[i]);
        if(source)result.ends.push_back(i);
        else if(values)result.values.push_back(double(low+i));
    }
    return result;
}

template<class Keys,class Positions>
std::size_t hash_slot(std::uint64_t key,const Keys& keys,const Positions& positions) {
    auto hash=key;hash^=hash>>33;hash*=UINT64_C(0xff51afd7ed558ccd);hash^=hash>>33;
    hash*=UINT64_C(0xc4ceb9fe1a85ec53);hash^=hash>>33;
    auto slot=std::size_t(hash)&(keys.size()-1);
    while(positions[slot] && keys[slot]!=key)slot=(slot+1)&(keys.size()-1);
    return slot;
}
template<class KeyAt> bool high_cardinality(KeyAt key_at,bool require_wide_span=false) {
    std::array<std::uint64_t,512> keys;
    std::array<unsigned char,512> occupied{};
    unsigned unique=0;
    std::uint64_t low=UINT64_MAX,high=0;
    for(std::size_t i=0;i<256;++i) {
        const auto key=key_at(i);
        const auto slot=hash_slot(key,keys,occupied);
        if(!occupied[slot]) {
            ++unique;
            if(require_wide_span) {
                low=std::min(low,key);high=std::max(high,key);
            }
            if(unique>200 && (!require_wide_span || high-low>=65536))return true;
            keys[slot]=key;
            occupied[slot]=1;
        }
    }
    return false;
}

template<unsigned DigitBits,class Offset,std::size_t... Pass>
void count_radix_digits(std::uint64_t key,std::uint64_t different,
    std::array<std::array<Offset,std::size_t(1)<<DigitBits>,sizeof...(Pass)>& histograms,
    std::index_sequence<Pass...>) {
    constexpr auto mask=(std::size_t(1)<<DigitBits)-1;
    const auto count_digit=[&](auto pass) {
        constexpr auto shift=decltype(pass)::value*DigitBits;
        if((different>>shift)&mask)++histograms[decltype(pass)::value][(key>>shift)&mask];
    };
    (count_digit(std::integral_constant<std::size_t,Pass>{}),...);
}

template<unsigned DigitBits,unsigned KeyBits,class T,class Key>
void radix_serial(std::vector<T>& values,Key key) {
    constexpr std::size_t bins=std::size_t(1)<<DigitBits;
    constexpr auto mask=bins-1;
    constexpr auto max_passes=(KeyBits+DigitBits-1)/DigitBits;
    using Offset=std::conditional_t<DigitBits==11,std::uint32_t,std::size_t>;
    const auto count=values.size();
    const auto first=key(values.front());
    auto previous=first;
    std::uint64_t different=0;
    bool unordered=false;
    for(const auto& entry:values) {
        const auto current=key(entry);
        different|=first^current;
        unordered|=current<previous;
        previous=current;
    }
    if(!unordered)return;

    // All digit histograms are invariant under permutation. Count them from
    // the input once and omit digits that cannot change the order.
    std::array<std::array<Offset,bins>,max_passes> histograms{};
    for(const auto& entry:values)
        count_radix_digits<DigitBits>(key(entry),different,histograms,std::make_index_sequence<max_passes>{});

    std::unique_ptr<T[]> temporary(new T[count]);
    auto* source=values.data();
    auto* target=temporary.get();
    for(unsigned shift=0;shift<KeyBits && different>>shift;shift+=DigitBits) {
        if(!((different>>shift)&mask))continue;
        auto& offsets=histograms[shift/DigitBits];
        Offset position=0;
        for(auto& offset:offsets) {
            const auto frequency=offset;
            offset=position;
            position+=frequency;
        }
        for(std::size_t i=0;i<count;++i)target[offsets[(key(source[i])>>shift)&mask]++]=source[i];
        std::swap(source,target);
    }
    if(source!=values.data())std::copy_n(source,count,values.data());
}

template<class T,class Key> void radix_sort(std::vector<T>& values,int workers,Key key,unsigned bits=64) {
    if(values.size()<2)return;
    if(workers==1) {
        if(bits<=32) {
            if(values.size()>=4096 && values.size()<=UINT32_MAX)radix_serial<11,32>(values,key);
            else radix_serial<8,32>(values,key);
        } else if(values.size()>=4096 && values.size()<65536)radix_serial<11,64>(values,key);
        else radix_serial<8,64>(values,key);
        return;
    }
    std::vector<T> temporary(values.size());
    std::vector<std::size_t> counts(std::size_t(workers)*256),offsets(counts.size());
    std::vector<std::pair<std::size_t,std::size_t>> segments(std::size_t(workers),{0,0});
    for(unsigned shift=0;shift<bits;shift+=8) {
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
void fill_keys(const void* input,std::uint64_t* keys,std::size_t count,std::ptrdiff_t pixels,
    std::ptrdiff_t offset,PixelType type,int workers) {
    const auto read=reader(type);
    detail::partition(count,workers,[&](auto begin,auto end,auto) {
        auto i=begin;
#ifdef SKIMAGEXX_AVX2
        if(has_avx2() && pixels>=0)i+=simd::histogram_keys(static_cast<const unsigned char*>(input)+
            (std::ptrdiff_t(begin)*pixels+offset)*std::ptrdiff_t(pixel_size(type)),keys+begin,end-begin,type,std::size_t(pixels));
#endif
        for(;i<end;++i)keys[i]=read(input,std::ptrdiff_t(i)*pixels+offset);
    });
}
bool signed_integer(PixelType type) {
    return type==PixelType::int8 || type==PixelType::int16 || type==PixelType::int32 || type==PixelType::int64;
}
void require_nonnegative_keys(const std::vector<std::uint64_t>& keys,PixelType type,bool required) {
    if(required && signed_integer(type) && *std::min_element(keys.begin(),keys.end())<sign64)
        throw std::invalid_argument("'list' argument must have no negative elements");
}
Distribution distribution(const void* input,std::size_t count,std::ptrdiff_t pixels,std::ptrdiff_t component,std::size_t channel,
    PixelType type,bool source,bool require_nonnegative,int workers,bool values=true) {
    if(!count)return {};
    const auto offset=std::ptrdiff_t(channel)*component;
    if(type==PixelType::uint8)
        return unsigned_distribution(static_cast<const std::uint8_t*>(input)+offset,count,pixels,source,values,workers);
    if(type==PixelType::uint16)
        return unsigned_distribution(static_cast<const std::uint16_t*>(input)+offset,count,pixels,source,values,workers);
    Distribution result;
    result.keys.resize(count);
    fill_keys(input,result.keys.data(),count,pixels,offset,type,workers);
    const bool small=type==PixelType::boolean || type==PixelType::int8 || type==PixelType::int16;
    if(small && count>=4096) {
        result.low=type==PixelType::int8 ? sign64-128 : type==PixelType::int16 ? sign64-32768 : 0;
        result.high=type==PixelType::boolean ? 1 : type==PixelType::int8 ? sign64+127 : sign64+32767;
    } else {
#ifdef SKIMAGEXX_AVX2
        if(count>=32 && has_avx2())simd::histogram_key_bounds(result.keys.data(),count,result.low,result.high);
        else
#endif
        {
            const auto bounds=std::minmax_element(result.keys.begin(),result.keys.end());
            result.low=*bounds.first;result.high=*bounds.second;
        }
    }
    require_nonnegative_keys(result.keys,type,require_nonnegative);
    const auto span=result.high-result.low;
    // Bounded direct tables cover quantized floating data as well as integers.
    // Key spans for most floating images are large, selecting the sorted path.
    if(span<65536) {
        result.dense=true;
        std::vector<std::size_t> histogram(std::size_t(span)+1);
        const auto capacity=std::min(count,histogram.size());
        result.counts.reserve(capacity);
        if(source)result.ends.reserve(capacity);else if(values)result.values.reserve(capacity);
        for(auto key:result.keys)++histogram[std::size_t(key-result.low)];
        for(std::size_t i=0;i<histogram.size();++i)if(histogram[i]) {
            result.counts.push_back(histogram[i]);
            if(source)result.ends.push_back(i);else if(values)result.values.push_back(value_from_key(result.low+i,type));
        }
        return result;
    }
    if(count>=4096 && !high_cardinality([&](auto i){return result.keys[i];})) {
        std::vector<std::uint64_t> hash_keys(8192);
        std::vector<std::size_t> positions(8192);
        if(source)result.slots.resize(count);
        bool complete=true;
        for(std::size_t i=0;i<count;++i) {
            const auto key=result.keys[i];
            const auto slot=hash_slot(key,hash_keys,positions);
            if(!positions[slot]) {
                if(result.counts.size()==4096){complete=false;break;}
                hash_keys[slot]=key;positions[slot]=result.counts.size()+1;
                result.counts.push_back(0);
            }
            ++result.counts[positions[slot]-1];
            if(source)result.slots[i]=std::uint16_t(slot);
        }
        if(complete) {
            std::vector<std::uint64_t> unique;
            unique.reserve(result.counts.size());
            for(std::size_t i=0;i<hash_keys.size();++i)if(positions[i])unique.push_back(hash_keys[i]);
            radix_sort(unique,1,[](auto key){return key;});
            auto previous=std::move(result.counts);result.counts.clear();result.counts.reserve(unique.size());
            if(!source && values)result.values.reserve(unique.size());
            for(std::size_t i=0;i<unique.size();++i) {
                const auto slot=hash_slot(unique[i],hash_keys,positions);
                result.counts.push_back(previous[positions[slot]-1]);
                if(!source && values)result.values.push_back(value_from_key(unique[i],type));
                positions[slot]=i+1;
            }
            if(source)result.hash_positions=std::move(positions);
            std::vector<std::uint64_t>().swap(result.keys);
            return result;
        }
        std::vector<std::uint16_t>().swap(result.slots);
        result.counts.clear();
    }
    if(source) {
        result.counts.reserve(count);result.ends.reserve(count);
        if(count<=UINT32_MAX && span<=UINT32_MAX) {
            std::vector<std::uint64_t> packed(count);
            for(std::size_t i=0;i<count;++i)packed[i]=((result.keys[i]-result.low)<<32)|std::uint64_t(i);
            std::vector<std::uint64_t>().swap(result.keys);
            radix_sort(packed,workers,[](auto value){return value>>32;},32);
            result.order.resize(count);
            for(std::size_t i=0;i<count;) {
                const auto start=i,key=packed[i]>>32;
                while(i<count && (packed[i]>>32)==key){result.order[i]=std::size_t(packed[i]&UINT32_MAX);++i;}
                result.counts.push_back(i-start);result.ends.push_back(i);
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
                result.counts.push_back(i-start);result.ends.push_back(i);
            }
        }
    } else {
        result.counts.reserve(count);if(values)result.values.reserve(count);
        radix_sort(result.keys,workers,[](auto key){return key;},result.high<=UINT32_MAX ? 32 : 64);
        for(std::size_t i=0;i<count;) {
            const auto start=i,key=result.keys[i];
            while(i<count && result.keys[i]==key)++i;
            result.counts.push_back(i-start);if(values)result.values.push_back(value_from_key(key,type));
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
template<class T,class Output>
unsigned apply_unsigned(const T* source,std::ptrdiff_t stride,const Distribution& distribution,
    const std::vector<double>& mapped,void* output,std::size_t count,std::size_t channels,
    std::size_t channel,int workers,Writer write,bool half_round,bool swap_output) {
    const auto bins=std::size_t(distribution.high-distribution.low)+1;
    const auto low=std::size_t(distribution.low);
    std::unique_ptr<Output[]> table(new Output[bins]);
    unsigned flags=0;
    for(std::size_t i=0;i<mapped.size();++i) {
        const auto bin=distribution.ends[i];
        if constexpr(std::is_same_v<Output,double>)table[bin]=mapped[i];
        else flags|=write(table.get(),bin,mapped[i],half_round,swap_output);
    }
    auto* target=static_cast<unsigned char*>(output);
    detail::partition(count,workers,[&](auto begin,auto end,auto) {
        visit_unsigned(source,stride,begin,end,[&](auto i,T value) {
            std::memcpy(target+(i*channels+channel)*sizeof(Output),&table[value-low],sizeof(Output));
        });
    });
    return flags;
}

template<class T>
unsigned apply_unsigned(const T* source,std::ptrdiff_t stride,const Distribution& distribution,
    const std::vector<double>& mapped,void* output,std::size_t count,std::size_t channels,
    std::size_t channel,int workers,Writer write,PixelType type,bool half_round,bool swap_output) {
    const auto emit=[&](auto tag,Writer convert) {
        return apply_unsigned<T,decltype(tag)>(source,stride,distribution,mapped,output,count,
            channels,channel,workers,convert,half_round,swap_output);
    };
    switch(pixel_size(type)) {
    case 1:return emit(std::uint8_t{},write);
    case 2:return emit(std::uint16_t{},write);
    case 4:return emit(std::uint32_t{},write);
    default:
        if(type==PixelType::float64 && !half_round && !swap_output)return emit(double{},nullptr);
        return emit(std::uint64_t{},write);
    }
}

template<class Next>
auto quantile_interpolator(std::size_t count,Next next) {
    auto [end,first_value]=next();
    const double first=double(end)/double(count);
    double lower_q=first,upper_q=first,lower_value=first_value,upper_value=first_value,slope=0.;
    bool has_upper=end<count,slope_ready=false;
    if(has_upper) {
        const auto group=next();
        end=group.first;upper_value=group.second;
        upper_q=double(end)/double(count);
    }
    return [=](double q) mutable {
        while(has_upper && q>=upper_q) {
            lower_q=upper_q;lower_value=upper_value;
            has_upper=end<count;
            if(has_upper) {
                const auto group=next();
                end=group.first;upper_value=group.second;
                upper_q=double(end)/double(count);
            }
            slope_ready=false;
        }
        if(q<=first)return first_value;
        if(!has_upper || q==lower_q)return lower_value;
        if(!slope_ready) {
            slope=(upper_value-lower_value)/(upper_q-lower_q);
            slope_ready=true;
        }
        auto value=slope*(q-lower_q)+lower_value;
        if(std::isnan(value)) {
            value=slope*(q-upper_q)+upper_value;
            if(std::isnan(value) && lower_value==upper_value)value=lower_value;
        }
        return value;
    };
}

std::vector<double> interpolate_quantiles(const Distribution& source,const Distribution& reference,
    std::size_t count,std::size_t reference_count) {
    auto interpolate=quantile_interpolator(reference_count,[&,group=std::size_t(0),end=std::size_t(0)]() mutable {
        end+=reference.counts[group];
        return std::pair{end,reference.values[group++]};
    });
    std::vector<double> mapped(source.counts.size());
    std::size_t total=0;
    for(std::size_t i=0;i<mapped.size();++i) {
        total+=source.counts[i];
        mapped[i]=interpolate(double(total)/double(count));
    }
    return mapped;
}

template<class Entry>
unsigned match_sorted(const void* source,const void* reference,void* output,std::size_t count,std::size_t reference_count,
    std::size_t channels,std::size_t channel,PixelType st,PixelType rt,HistogramMatchingStrides strides,
    bool unsigned_source,Writer write,bool half_round,bool swap_output) {
    std::vector<Entry> entries;
    entries.reserve(count);
    std::uint64_t source_keys[1024];
    const auto source_offset=std::ptrdiff_t(channel)*strides.source_channels;
    for(std::size_t begin=0;begin<count;) {
        const auto size=std::min(std::size(source_keys),count-begin);
        fill_keys(source,source_keys,size,strides.source_pixels,
            std::ptrdiff_t(begin)*strides.source_pixels+source_offset,st,1);
        for(std::size_t i=0;i<size;++i)entries.push_back(indexed_key<Entry>(source_keys[i],begin+i));
        begin+=size;
    }
    radix_sort(entries,1,entry_key,std::is_same_v<Entry,CompactEntry> ? 32 : 64);

    std::vector<std::uint64_t> keys(reference_count);
    fill_keys(reference,keys.data(),reference_count,strides.reference_pixels,
        std::ptrdiff_t(channel)*strides.reference_channels,rt,1);
    require_nonnegative_keys(keys,rt,unsigned_source);
    radix_sort(keys,1,[](auto key){return key;},compact_keys(rt) ? 32 : 64);
    auto interpolate=quantile_interpolator(reference_count,[&,end=std::size_t(0)]() mutable {
        const auto key=keys[end++];
        while(end<keys.size() && keys[end]==key)++end;
        return std::pair{end,value_from_key(key,rt)};
    });

    unsigned flags=0;
    for(std::size_t begin=0;begin<count;) {
        auto end=begin+1;
        const auto key=entries[begin].key();
        while(end<count && entries[end].key()==key)++end;
        const auto value=interpolate(double(end)/double(count));
        if(write) {
            for(;begin<end;++begin)
                flags|=write(output,std::size_t(entries[begin].position)*channels+channel,value,half_round,swap_output);
        } else {
            for(;begin<end;++begin)
                static_cast<double*>(output)[std::size_t(entries[begin].position)*channels+channel]=value;
        }
    }
    return flags;
}

template<class T> void apply_floating(const Distribution& source,const T* mapped,std::size_t groups,
    T* output,std::size_t count,std::size_t channels,std::size_t channel,int workers) {
    if(source.dense) {
        // Only populated bins are read by source.keys; no image-sized cast
        // callback or zero-filled lookup table is needed for native floats.
        std::unique_ptr<double[]> table(new double[std::size_t(source.high-source.low)+1]);
        for(std::size_t i=0;i<groups;++i)table[source.ends[i]]=double(mapped[i]);
        detail::partition(count,workers,[&](auto begin,auto end,auto) {
            auto i=begin;
#ifdef SKIMAGEXX_AVX2
            if(has_avx2())i+=simd::histogram_lookup(source.keys.data()+begin,output+begin*channels+channel,
                end-begin,source.low,table.get(),std::is_same_v<T,double> ? PixelType::float64 : PixelType::float32,channels);
#endif
            for(;i<end;++i)output[i*channels+channel]=T(table[std::size_t(source.keys[i]-source.low)]);
        });
    } else if(!source.hash_positions.empty()) {
        detail::partition(count,workers,[&](auto begin,auto end,auto) {
            for(auto i=begin;i<end;++i)
                output[i*channels+channel]=mapped[source.hash_positions[source.slots[i]]-1];
        });
    } else {
        detail::partition(groups,workers,[&](auto begin,auto end,auto) {
            for(auto group=begin;group<end;++group) {
                const auto value=mapped[group];
                for(auto i=group ? source.ends[group-1] : 0;i<source.ends[group];++i)
                    output[source.order[i]*channels+channel]=value;
            }
        });
    }
}
}
namespace {
void validate_histogram_view(std::size_t count,std::size_t channels,PixelType type,
    std::ptrdiff_t pixels,std::ptrdiff_t component) {
    const auto limit=std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/pixel_size(type);
    const auto magnitude=[](std::ptrdiff_t value){return value<0 ? std::size_t(-(value+1))+1 : std::size_t(value);};
    const auto p=count>1 ? magnitude(pixels) : 0,c=channels>1 ? magnitude(component) : 0;
    if((count>1 && p>limit/(count-1)) || (channels>1 && c>(limit-(count-1)*p)/(channels-1)))
        throw std::overflow_error("histogram-matching strides overflow");
}
}
unsigned match_histograms_into(const void* source,const void* reference,void* output,std::size_t count,std::size_t reference_count,
    std::size_t channels,PixelType st,PixelType rt,PixelType ot,bool unsigned_source,bool half_round,bool swap_output,unsigned* channel_flags) {
    if(channels>std::size_t(std::numeric_limits<std::ptrdiff_t>::max()))
        throw std::overflow_error("histogram-matching dimensions overflow");
    return match_histograms_strided_into(source,reference,output,count,reference_count,channels,st,rt,ot,
        {std::ptrdiff_t(channels),1,std::ptrdiff_t(channels),1},unsigned_source,half_round,swap_output,channel_flags);
}
std::vector<std::size_t> histogram_sorted_counts(const void* input,std::size_t count,PixelType type,int workers) {
    if(workers<1 || workers>1024)throw std::invalid_argument("entropy worker snapshot outside [1,1024]");
#ifndef _OPENMP
    if(workers!=1)throw std::invalid_argument("this build has no OpenMP support");
#endif
    if(count>std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/pixel_size(type) || (count&&!input))
        throw std::invalid_argument("entropy input extent overflow or null storage");
    if(!count)return {};
    return distribution(input,count,1,0,0,type,false,false,count>=detail::parallel_min?workers:1,false).counts;
}

unsigned match_histograms_strided_into(const void* source,const void* reference,void* output,std::size_t count,std::size_t reference_count,
    std::size_t channels,PixelType st,PixelType rt,PixelType ot,HistogramMatchingStrides strides,
    bool unsigned_source,bool half_round,bool swap_output,unsigned* channel_flags) {
    (void)reader(st);(void)reader(rt);const auto write=writer(ot);
    const auto limit=std::size_t(std::numeric_limits<std::ptrdiff_t>::max());
    const auto bytes=std::max({pixel_size(st),pixel_size(rt),pixel_size(ot)});
    if(channels && (channels>limit/bytes || std::max(count,reference_count)>limit/(channels*bytes)))
        throw std::overflow_error("histogram-matching dimensions overflow");
    if(!channels || !count)return 0;
    if(!reference_count)throw std::invalid_argument("array of sample points is empty");
    validate_histogram_view(count,channels,st,strides.source_pixels,strides.source_channels);
    validate_histogram_view(reference_count,channels,rt,strides.reference_pixels,strides.reference_channels);
    if(!source || !reference || !output)throw std::invalid_argument("histogram-matching buffers must be nonnull");
    const int workers=detail::active_workers(std::max(count,reference_count));
    unsigned flags=0;
    for(std::size_t channel=0;channel<channels;++channel) {
        const auto source_offset=std::ptrdiff_t(channel)*strides.source_channels;
        if(workers==1 && count>=4096 && count<=UINT32_MAX && pixel_size(st)>=4) {
            std::uint64_t sample[256];
            fill_keys(source,sample,std::size(sample),strides.source_pixels,source_offset,st,1);
            if(high_cardinality([&](auto i){return sample[i];},true)) {
                const auto sorted_write=ot==PixelType::float64 && !half_round && !swap_output ? nullptr : write;
                const auto match=compact_keys(st) ? match_sorted<CompactEntry> : match_sorted<Packed>;
                const auto notice=match(source,reference,output,count,reference_count,channels,channel,st,rt,
                    strides,unsigned_source,sorted_write,half_round,swap_output);
                if(channel_flags)channel_flags[channel]=notice;
                flags|=notice;
                continue;
            }
        }
        auto a=distribution(source,count,strides.source_pixels,strides.source_channels,channel,st,true,false,workers);
        auto b=distribution(reference,reference_count,strides.reference_pixels,strides.reference_channels,channel,rt,false,unsigned_source,workers);
        std::vector<std::uint64_t>().swap(b.keys);
        auto mapped=interpolate_quantiles(a,b,count,reference_count);
        std::vector<std::size_t>().swap(b.counts);
        std::vector<std::size_t>().swap(a.counts);
        std::vector<double>().swap(b.values);
        if(st==PixelType::uint8 || st==PixelType::uint16) {
            const auto apply=[&](const auto* data) {
                return apply_unsigned(data+source_offset,strides.source_pixels,a,mapped,output,count,channels,
                    channel,workers,write,ot,half_round,swap_output);
            };
            const auto notice=st==PixelType::uint8 ? apply(static_cast<const std::uint8_t*>(source)) :
                apply(static_cast<const std::uint16_t*>(source));
            if(channel_flags)channel_flags[channel]=notice;
            flags|=notice;
            continue;
        }
        if(!half_round && !swap_output && (ot==PixelType::float64 || ot==PixelType::float32)) {
            unsigned channel_notice=0;
            if(ot==PixelType::float64)
                apply_floating(a,mapped.data(),mapped.size(),static_cast<double*>(output),count,channels,channel,workers);
            else {
                const auto groups=mapped.size();
                std::unique_ptr<float[]> converted(new float[groups]);
                for(std::size_t i=0;i<groups;++i)channel_notice|=write(converted.get(),i,mapped[i],false,false);
                std::vector<double>().swap(mapped);
                apply_floating(a,converted.get(),groups,static_cast<float*>(output),count,channels,channel,workers);
            }
            if(channel_flags)channel_flags[channel]=channel_notice;
            flags|=channel_notice;
            continue;
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
                    local|=write(output,i*channels+channel,mapped[a.hash_positions[a.slots[i]]-1],half_round,swap_output);
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

namespace {
template<class Assign> void with_rank_labels(PixelType type,void* output,Assign assign) {
    switch(type) {
    case PixelType::uint8:assign(static_cast<std::uint8_t*>(output));break;
    case PixelType::uint16:assign(static_cast<std::uint16_t*>(output));break;
    case PixelType::uint32:assign(static_cast<std::uint32_t*>(output));break;
    case PixelType::uint64:assign(static_cast<std::uint64_t*>(output));break;
    default:throw std::invalid_argument("unsigned rank labels required");
    }
}

template<class Entry>
void rank_sort_and_emit(std::vector<Entry>& entries,int workers,PixelType label_type,void* output,
    std::vector<std::size_t>& representatives) {
    constexpr auto bits=std::is_same_v<Entry,CompactEntry> ? 32 : 64;
    radix_sort(entries,workers,entry_key,bits);
    const auto count=entries.size();
    if(workers==1) {
        representatives.reserve(count);
        with_rank_labels(label_type,output,[&](auto* labels) {
            using U=std::remove_pointer_t<decltype(labels)>;
            auto previous=entries.front().key();
            representatives.push_back(entries.front().position);
            for(const auto& entry:entries) {
                const auto key=entry.key();
                if(key!=previous) {
                    representatives.push_back(entry.position);
                    previous=key;
                }
                labels[entry.position]=U(representatives.size()-1);
            }
        });
        return;
    }

    // Keep the original segment boundaries if OpenMP resizes its next team.
    std::vector<std::pair<std::size_t,std::size_t>> segments(std::size_t(workers),{0,0});
    std::vector<std::size_t> groups(std::size_t(workers),0),prefix(std::size_t(workers),0);
    const auto starts_group=[&](auto i){return !i || entries[i].key()!=entries[i-1].key();};
    detail::partition(count,workers,[&](auto begin,auto end,auto worker) {
        segments[worker]={begin,end};
        for(auto i=begin;i<end;++i)groups[worker]+=starts_group(i);
    });
    std::size_t total=0;
    for(std::size_t worker=0;worker<groups.size();++worker){prefix[worker]=total;total+=groups[worker];}
    representatives.resize(total);
    with_rank_labels(label_type,output,[&](auto* labels) {
        using U=std::remove_pointer_t<decltype(labels)>;
        detail::partition(std::size_t(workers),workers,[&](auto begin,auto end,auto) {
            for(auto worker=begin;worker<end;++worker) {
                auto rank=prefix[worker];
                for(auto i=segments[worker].first;i<segments[worker].second;++i) {
                    const auto original=entries[i].position;
                    if(starts_group(i))representatives[rank++]=original;
                    labels[original]=U(rank-1);
                }
            }
        });
    });
}

template<class Entry>
bool rank_entries(const void* input,std::size_t count,PixelType type,KeyReader read,int workers,void* output,
    PixelType label_type,std::vector<std::size_t>& representatives) {
    std::vector<Entry> entries(count);
    std::vector<unsigned> notices(std::size_t(workers),0);
    std::vector<std::pair<std::uint64_t,std::uint64_t>> bounds(std::size_t(workers),{UINT64_MAX,0});
    const auto floating=type==PixelType::float32 || type==PixelType::float64;
    const auto rank_read=type==PixelType::float32 ? read_key<float,true> : read_key<double,true>;
    detail::partition(count,workers,[&](auto begin,auto end,auto worker) {
        unsigned flags=0;auto i=begin;std::uint64_t low=UINT64_MAX,high=0;
        const auto put=[&](std::size_t index,std::uint64_t key) {
            entries[index]=indexed_key<Entry>(key,index);
            low=std::min(low,key);high=std::max(high,key);
        };
#ifdef SKIMAGEXX_AVX2
        if(has_avx2()) {
            std::uint64_t keys[1024];
            while(i<end) {
                const auto n=std::min(std::size_t(1024),end-i);
                const auto done=simd::histogram_keys(static_cast<const unsigned char*>(input)+i*pixel_size(type),keys,n,type,1,flags);
                if(!done)break;
                for(std::size_t k=0;k<done;++k)put(i+k,keys[k]);
                i+=done;
            }
        }
#endif
        if(floating) {
            for(;i<end;++i)put(i,rank_read(input,std::ptrdiff_t(i),&flags));
        } else if(type==PixelType::boolean) {
            const auto* data=static_cast<const unsigned char*>(input);
            for(;i<end;++i) {
                if(data[i]>1)flags|=8;
                put(i,data[i]!=0);
            }
        } else {
            for(;i<end;++i)put(i,read(input,std::ptrdiff_t(i)));
        }
        notices[worker]=flags;
        bounds[worker]={low,high};
    });
    unsigned flags=0;for(auto notice:notices)flags|=notice;
    if((flags&9) || (flags&6)==6)return false;
    representatives.clear();
    auto emit_table=[&](auto rank_at,const std::size_t* dense=nullptr,std::uint64_t low=0) {
        with_rank_labels(label_type,output,[&](auto* labels) {
            using U=std::remove_pointer_t<decltype(labels)>;
            detail::partition(count,workers,[&](auto begin,auto end,auto) {
#ifdef SKIMAGEXX_AVX2
                if(dense && has_avx2())begin+=simd::rank_dense_labels(entries.data()+begin,labels+begin,end-begin,low,dense,label_type,sizeof(Entry));
#else
                (void)dense;(void)low;
#endif
                for(auto i=begin;i<end;++i)labels[i]=U(rank_at(i));
            });
        });
    };
    if(count>=4096) {
        std::uint64_t low=UINT64_MAX,high=0;
        for(const auto& bound:bounds){low=std::min(low,bound.first);high=std::max(high,bound.second);}
        const auto span=high-low;
        if(span<65536) {
            std::vector<std::size_t> table(std::size_t(span)+1,count);
            for(const auto& entry:entries) {
                auto& position=table[std::size_t(entry.key()-low)];
                if(position==count)position=entry.position;
            }
            for(auto& position:table)if(position!=count){const auto original=position;position=representatives.size();representatives.push_back(original);}
            emit_table([&](auto i){return table[std::size_t(entries[i].key()-low)];},table.data(),low);
            return true;
        }
        if(!high_cardinality([&](auto i){return entries[i].key();})) {
            std::vector<std::uint64_t> keys(8192);
            std::vector<std::size_t> positions(8192,0);
            std::size_t unique_count=0,visited=0;
            for(;visited<count;++visited) {
                auto& entry=entries[visited];
                const auto key=entry.key();
                const auto slot=hash_slot(key,keys,positions);
                if(!positions[slot]) {
                    if(unique_count==4096)break;
                    keys[slot]=key;positions[slot]=visited+1;++unique_count;
                }
                entry.position=decltype(entry.position)(slot);
            }
            if(visited==count) {
                std::vector<Entry> unique;unique.reserve(unique_count);
                for(std::size_t slot=0;slot<keys.size();++slot)
                    if(positions[slot])unique.push_back(indexed_key<Entry>(keys[slot],positions[slot]-1));
                radix_sort(unique,1,entry_key,std::is_same_v<Entry,CompactEntry> ? 32 : 64);
                for(const auto& entry:unique) {
                    representatives.push_back(entry.position);
                    positions[hash_slot(entry.key(),keys,positions)]=representatives.size();
                }
                emit_table([&](auto i){return positions[entries[i].position]-1;});
                return true;
            }
            // The sample underestimated cardinality. Restore the original
            // positions before passing the records to the stable sort.
            for(std::size_t i=0;i<visited;++i)entries[i].position=decltype(entries[i].position)(i);
        }
    }
    rank_sort_and_emit(entries,workers,label_type,output,representatives);
    return true;
}
}

bool rank_order_into(const void* input,std::size_t count,PixelType type,void* output,
    PixelType label_type,std::vector<std::size_t>& representatives) {
    if(!count || !input || !output)throw std::invalid_argument("nonempty rank-order buffers required");
    const auto maximum=label_type==PixelType::uint8 ? UINT64_C(255) : label_type==PixelType::uint16 ? UINT64_C(65535) :
        label_type==PixelType::uint32 ? UINT64_C(4294967295) : label_type==PixelType::uint64 ? UINT64_MAX : 0;
    if(!maximum || count-1>maximum)throw std::invalid_argument("rank label type is too narrow");
    if(count>std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/sizeof(WideEntry))
        throw std::overflow_error("rank-order dimensions overflow");
    const auto read=reader(type);const auto workers=detail::active_workers(count);
    if(type==PixelType::boolean && *static_cast<const unsigned char*>(input)>1)return false;
    if(type==PixelType::float32 && std::isnan(*static_cast<const float*>(input)))return false;
    if(type==PixelType::float64 && std::isnan(*static_cast<const double*>(input)))return false;
    const auto bytes=pixel_size(type);bool uniform=true;
#ifdef SKIMAGEXX_AVX2
    if(has_avx2())uniform=simd::rank_uniform_bits(input,count,bytes);
    else
#endif
    for(std::size_t i=1;i<count;++i)if(std::memcmp(input,static_cast<const unsigned char*>(input)+i*bytes,bytes)){uniform=false;break;}
    if(uniform){std::memset(output,0,count*pixel_size(label_type));representatives.assign(1,0);return true;}
    if(count<=UINT32_MAX) {
        if(compact_keys(type))return rank_entries<CompactEntry>(input,count,type,read,workers,output,label_type,representatives);
        return rank_entries<Packed>(input,count,type,read,workers,output,label_type,representatives);
    }
    return rank_entries<WideEntry>(input,count,type,read,workers,output,label_type,representatives);
}
}
