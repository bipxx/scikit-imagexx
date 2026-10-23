#include "resize_range.hpp"
#include "skimagexx/image.hpp"
#include "execution.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace skimagexx::detail {
namespace {
template<class T> struct alignas(64) Partial {T low{},high{};ResizeRangeInfo info;};
template<class T> void add(Partial<T>& state,T value) {
    if constexpr(std::is_floating_point_v<T>) {
        if(std::isnan(value)){state.info.has_nan=true;return;}
        if(value==0) {
            if(std::signbit(value))state.info.negative_zero=true;
            else state.info.positive_zero=true;
        }
    }
    if(!state.info.has_value){state.low=state.high=value;state.info.has_value=true;}
    else {if(value<state.low)state.low=value;if(value>state.high)state.high=value;}
}
template<class T,bool Boolean=false> ResizeRangeInfo range(const T* input,std::size_t count,PixelType type,T* lower,T* upper) {
    const auto workers=active_workers(count);
    // Allocate before entering OpenMP, with separate cache lines per worker.
    std::vector<Partial<T>> partials{std::size_t(workers)};
    partition(count,workers,[&](std::size_t begin,std::size_t end,std::size_t id) {
        auto& state=partials[id];
#ifdef SKIMAGEXX_AVX2
        if(has_avx2())begin+=simd::resize_range(input+begin,end-begin,type,&state.low,&state.high,state.info);
#else
        (void)type;
#endif
        for(auto i=begin;i<end;++i)add(state,Boolean ? T(input[i]!=0) : input[i]);
    });
    Partial<T> result;
    for(const auto& state:partials) {
        result.info.has_nan|=state.info.has_nan;
        result.info.positive_zero|=state.info.positive_zero;
        result.info.negative_zero|=state.info.negative_zero;
        if(state.info.has_value){add(result,state.low);add(result,state.high);}
    }
    if constexpr(std::is_floating_point_v<T>)if(!result.info.has_value)result.low=result.high=std::numeric_limits<T>::quiet_NaN();
    *lower=result.low;*upper=result.high;return result.info;
}
}
ResizeRangeInfo resize_range_into(const void* input,std::size_t count,PixelType type,void* lower,void* upper) {
    const auto bytes=pixel_size(type);
    if(count>std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/bytes)throw std::overflow_error("range dimensions overflow");
    if(!count)return {};
    if(!input || !lower || !upper)throw std::invalid_argument("nonnull range input and bounds required");
    switch(type) {
        case PixelType::boolean:return range<std::uint8_t,true>(static_cast<const std::uint8_t*>(input),count,type,static_cast<std::uint8_t*>(lower),static_cast<std::uint8_t*>(upper));
        #define RANGE(tag,T) case PixelType::tag:return range(static_cast<const T*>(input),count,type,static_cast<T*>(lower),static_cast<T*>(upper));
        RANGE(uint8,std::uint8_t) RANGE(int8,std::int8_t) RANGE(uint16,std::uint16_t) RANGE(int16,std::int16_t)
        RANGE(uint32,std::uint32_t) RANGE(int32,std::int32_t) RANGE(uint64,std::uint64_t) RANGE(int64,std::int64_t)
        RANGE(float32,float) RANGE(float64,double)
        #undef RANGE
        default:throw std::invalid_argument("unsupported range dtype");
    }
}
}
