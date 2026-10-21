#include "skimagexx/integrate.hpp"
#include "execution.hpp"
#include "simd.hpp"
#include "half.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
namespace skimagexx {
namespace {
double half_value(std::uint16_t h) {
    return double(detail::decode_half(h));
}
template<class T,bool Boolean,bool Half> double sample(const unsigned char* source,bool swap) {
    T value;
    if(swap && sizeof(T)>1) {
        std::array<unsigned char,sizeof(T)> bytes;
        for(std::size_t j=0;j<sizeof(T);++j)bytes[j]=source[sizeof(T)-1-j];
        std::memcpy(&value,bytes.data(),sizeof(T));
    } else std::memcpy(&value,source,sizeof(T));
    if constexpr(Boolean)return value!=0 ? 1. : 0.;
    else if constexpr(Half)return half_value(std::uint16_t(value));
    else return double(value);
}
void validate(const IntegralTableView& table,const std::int64_t* lower,const std::int64_t* upper,double* output,std::size_t windows) {
    const auto rank=table.shape.size();
    if(rank>=std::numeric_limits<std::size_t>::digits || table.strides.size()!=rank)throw std::invalid_argument("integral table rank/strides invalid");
    if(table.half && table.type!=PixelType::uint16)throw std::invalid_argument("binary16 integral table storage required");
    const auto item=pixel_size(table.type),limit=std::size_t(std::numeric_limits<std::ptrdiff_t>::max());
    if(windows>limit/sizeof(double) || (rank && windows>limit/(rank*sizeof(std::int64_t))))throw std::overflow_error("integral window dimensions overflow");
    if(!windows)return;
    if(!table.data || !output || (rank && (!lower || !upper)))throw std::invalid_argument("nonnull integral window storage required");
    std::size_t positive=0,negative=0;
    for(std::size_t d=0;d<rank;++d) {
        const auto length=table.shape[d];const auto step=table.strides[d];
        if(!length)throw std::invalid_argument("cannot query empty integral table axis");
        if(step==std::numeric_limits<std::ptrdiff_t>::min())throw std::overflow_error("integral table stride overflow");
        const auto magnitude=std::size_t(step<0 ? -step : step);
        if(length>1 && magnitude>limit/(length-1))throw std::overflow_error("integral table stride overflow");
        const auto extent=magnitude*(length-1);auto& total=step<0 ? negative : positive;
        if(extent>limit-total || (step>=0 && item-1>limit-total-extent))throw std::overflow_error("integral table span overflow");
        total+=extent;
    }
    for(std::size_t row=0;row<windows;++row)for(std::size_t d=0;d<rank;++d) {
        const auto lo=lower[row*rank+d],hi=upper[row*rank+d];
        if(hi<0 || std::uint64_t(hi)>=table.shape[d] || (lo>=0 && std::uint64_t(lo)>=table.shape[d]))throw std::out_of_range("integral window coordinate outside table");
    }
}
template<class T,bool Boolean=false,bool Half=false> bool run(const IntegralTableView& table,const std::int64_t* lower,
    const std::int64_t* upper,double* output,std::size_t windows) {
    const auto rank=table.shape.size(),corners=std::size_t(1)<<rank;
    const auto work=windows>std::numeric_limits<std::size_t>::max()/corners ? std::numeric_limits<std::size_t>::max() : windows*corners;
    const auto workers=detail::active_workers(work);std::vector<unsigned char> supported(std::size_t(workers),1);
    const auto* bytes=static_cast<const unsigned char*>(table.data);
    const bool vectorized=has_avx2();(void)vectorized;
    detail::partition(windows,workers,[&](auto begin,auto end,auto worker) {
        std::fill(output+begin,output+end,0.);
        std::array<double,256> values;
        for(std::size_t corner=0;corner<corners;++corner) {
            unsigned parity=0;for(auto mask=corner;mask;mask>>=1)parity^=unsigned(mask&1);
            for(auto first=begin;first<end;) {
                const auto count=std::min(values.size(),end-first);
                for(std::size_t j=0;j<count;++j) {
                    std::ptrdiff_t offset=0;bool bad=false;
                    for(std::size_t d=0;d<rank;++d) {
                        const bool selected=(corner&(std::size_t(1)<<(rank-1-d)))!=0;
                        const auto coordinate=selected ? lower[(first+j)*rank+d] : upper[(first+j)*rank+d];
                        if(selected && coordinate<0) {bad=true;break;}
                        offset+=std::ptrdiff_t(coordinate)*table.strides[d];
                    }
                    if(bad)values[j]=0.;
                    else {
                        const double value=sample<T,Boolean,Half>(bytes+offset,table.swap);
                        if constexpr(std::is_floating_point_v<T> || Half)if(!std::isfinite(value)) {supported[worker]=0;return;}
                        values[j]=parity ? -value : value;
                    }
                }
                std::size_t done=0;
#ifdef SKIMAGEXX_AVX2
                if(vectorized)done=simd::integrate_add(values.data(),output+first,count);
#endif
                for(auto j=done;j<count;++j)output[first+j]+=values[j];
                if constexpr(std::is_same_v<T,double>)for(std::size_t j=0;j<count;++j)if(!std::isfinite(output[first+j])) {supported[worker]=0;return;}
                first+=count;
            }
        }
    });
    return std::all_of(supported.begin(),supported.end(),[](auto value){return value!=0;});
}
}
bool integrate_windows_into(const IntegralTableView& table,const std::int64_t* lower,const std::int64_t* upper,double* output,std::size_t windows) {
    validate(table,lower,upper,output,windows);if(!windows)return true;
#define QUERY(tag,T) case PixelType::tag:return run<T>(table,lower,upper,output,windows);
    switch(table.type) {
    case PixelType::boolean:return run<std::uint8_t,true>(table,lower,upper,output,windows);
    case PixelType::uint16:if(table.half)return run<std::uint16_t,false,true>(table,lower,upper,output,windows);return run<std::uint16_t>(table,lower,upper,output,windows);
    QUERY(uint8,std::uint8_t) QUERY(int8,std::int8_t) QUERY(int16,std::int16_t)
    QUERY(uint32,std::uint32_t) QUERY(int32,std::int32_t) QUERY(uint64,std::uint64_t) QUERY(int64,std::int64_t)
    QUERY(float32,float) QUERY(float64,double)
    }
#undef QUERY
    throw std::invalid_argument("unsupported integral table storage");
}
}
