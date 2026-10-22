#include "skimagexx/label_color.hpp"
#include "execution.hpp"
#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace skimagexx {
namespace {
template<class T> T load(const unsigned char* source,bool swap) {
    T result;
    if(!swap || sizeof(T)==1)std::memcpy(&result,source,sizeof(T));
    else {
        std::array<unsigned char,sizeof(T)> bytes;
        std::reverse_copy(source,source+sizeof(T),bytes.begin());
        std::memcpy(&result,bytes.data(),sizeof(T));
    }
    return result;
}
template<class T> bool finite(T value) {
    if constexpr(std::is_floating_point_v<T>) {
        using U=std::conditional_t<sizeof(T)==4,std::uint32_t,std::uint64_t>;
        U bits;std::memcpy(&bits,&value,sizeof(T));
        const U exponent=sizeof(T)==4 ? U(0x7f800000u) : U(UINT64_C(0x7ff0000000000000));
        return (bits&exponent)!=exponent;
    } else return true;
}
template<class T> std::uint64_t ordered_key(T value) {
    if constexpr(std::is_floating_point_v<T>) {
        using U=std::conditional_t<sizeof(T)==4,std::uint32_t,std::uint64_t>;
        U bits;std::memcpy(&bits,&value,sizeof(T));
        const U sign=U(1)<<(sizeof(T)*8-1);
        if(value==T(0))bits=0; // NumPy unique merges positive and negative zero.
        return std::uint64_t(bits&sign ? U(~bits) : U(bits^sign));
    } else if constexpr(std::is_signed_v<T>) {
        using U=std::make_unsigned_t<T>;
        return std::uint64_t(U(value)^(U(1)<<(sizeof(T)*8-1)));
    } else return std::uint64_t(value);
}
template<class T,bool Boolean=false> bool map(const void* source,std::size_t count,bool swap,
    std::uint64_t background,bool overlay,std::uint64_t* output,LabelSummary& summary) {
    const auto* bytes=static_cast<const unsigned char*>(source);
    const int workers=detail::active_workers(count);
    std::vector<std::uint64_t> values(count);
    std::uint64_t low=std::numeric_limits<std::uint64_t>::max(),high=0,bg_key=background;
    bool can_match=true;
    for(std::size_t i=0;i<count;++i) {
        T value=load<T>(bytes+i*sizeof(T),swap);
        if constexpr(Boolean)value=T(value!=0);
        if(!finite(value))return false;
        if(overlay) {
            if constexpr(std::is_floating_point_v<T>) {
                if(background || value<T(0) || double(value)>=18446744073709551616.0)return false;
            } else if constexpr(std::is_signed_v<T>)if(value<0)return false;
            values[i]=std::uint64_t(value);
            low=std::min(low,values[i]);high=std::max(high,values[i]);
        } else values[i]=ordered_key(value);
    }
    if(!overlay) {
        if constexpr(std::is_floating_point_v<T>) {
            // Keep Python-int comparison promotion exact in both NumPy1/2.
            const auto limit=sizeof(T)==4 ? UINT64_C(16777216) : UINT64_C(9007199254740992);
            if(background>limit)return false;
            bg_key=ordered_key(T(background));
        } else {
            const auto maximum=Boolean ? std::uint64_t(1) : std::uint64_t(std::numeric_limits<T>::max());
            can_match=background<=maximum;
            if(can_match)bg_key=ordered_key(T(background));
        }
    }
    std::uint64_t mask=std::numeric_limits<std::uint64_t>::max();
    if(overlay) {
        const auto offset=std::min(low,background);
        if(offset) {
            if constexpr(Boolean || std::is_floating_point_v<T>)return false;
            else {
                if(background>std::uint64_t(std::numeric_limits<T>::max()) || background>UINT64_C(9007199254740992))return false;
                for(auto& value:values)value-=offset;
                high-=offset;background-=offset;bg_key=background;
            }
        }
        if(high<=UINT8_MAX)mask=UINT8_MAX;
        else if(high<=UINT16_MAX)mask=UINT16_MAX;
        else if(high<=UINT32_MAX)mask=UINT32_MAX;
    }
    auto unique=values;
    std::sort(unique.begin(),unique.end());
    unique.erase(std::unique(unique.begin(),unique.end()),unique.end());
    const auto bg=std::lower_bound(unique.begin(),unique.end(),bg_key);
    const bool present=can_match && bg!=unique.end() && *bg==bg_key;
    const auto bg_rank=std::size_t(bg-unique.begin());
    if(unique.size()==std::numeric_limits<std::size_t>::max())throw std::overflow_error("too many label regions");
    summary={unique.size()+(present ? 0 : 1),background,present};
    detail::partition(count,workers,[&](std::size_t begin,std::size_t end,std::size_t) {
        for(auto i=begin;i<end;++i) {
            const auto position=std::size_t(std::lower_bound(unique.begin(),unique.end(),values[i])-unique.begin());
            const auto rank=present && position==bg_rank ? 0 : position+(!present || position<bg_rank ? 1 : 0);
            output[i]=std::uint64_t(rank)&mask;
        }
    });
    return true;
}
}
bool label_ranks(const void* input,std::size_t count,PixelType type,bool swap,
    std::uint64_t background,bool overlay,std::uint64_t* ranks,LabelSummary& summary) {
    if(count && (!input || !ranks))throw std::invalid_argument("nonnull label storage required");
    if(!count) {summary={1,background,false};return true;}
    if(count>std::numeric_limits<std::size_t>::max()/sizeof(std::uint64_t))throw std::overflow_error("label storage too large");
    switch(type) {
    case PixelType::boolean:return map<std::uint8_t,true>(input,count,swap,background,overlay,ranks,summary);
#define LABEL_CASE(kind,T) case PixelType::kind:return map<T>(input,count,swap,background,overlay,ranks,summary)
    LABEL_CASE(uint8,std::uint8_t);LABEL_CASE(int8,std::int8_t);
    LABEL_CASE(uint16,std::uint16_t);LABEL_CASE(int16,std::int16_t);
    LABEL_CASE(uint32,std::uint32_t);LABEL_CASE(int32,std::int32_t);
    LABEL_CASE(uint64,std::uint64_t);LABEL_CASE(int64,std::int64_t);
    LABEL_CASE(float32,float);LABEL_CASE(float64,double);
#undef LABEL_CASE
    }
    throw std::invalid_argument("unsupported label storage");
}
}
