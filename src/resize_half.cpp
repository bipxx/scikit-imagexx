#include "resize_half.hpp"
#include "skimagexx/image.hpp"
#include "execution.hpp"
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace skimagexx::detail {
namespace {
void check_count(std::size_t count) {
    if(count>std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/2)throw std::overflow_error("half dimensions overflow");
}
struct alignas(64) Partial {std::uint16_t low=0,high=0;ResizeRangeInfo info;};
void add(Partial& state,std::uint16_t value) {
    if(half_nan(value)){state.info.has_nan=true;return;}
    if(half_zero(value)) {
        if(value&0x8000)state.info.negative_zero=true;else state.info.positive_zero=true;
    }
    if(!state.info.has_value){state.low=state.high=value;state.info.has_value=true;}
    else {if(half_less(value,state.low))state.low=value;if(half_less(state.high,value))state.high=value;}
}
}
void half_copy_into(const void* input,std::uint16_t* output,std::size_t count,bool swap) {
    check_count(count);if(!count)return;
    if(!input || !output || input==output)throw std::invalid_argument("disjoint nonnull half storage required");
    const auto* bytes=static_cast<const unsigned char*>(input);
    partition(count,active_workers(count),[&](std::size_t begin,std::size_t end,auto) {
        if(!swap){std::memcpy(output+begin,bytes+2*begin,2*(end-begin));return;}
#ifdef SKIMAGEXX_AVX2
        if(has_avx2())begin+=simd::half_copy(bytes+2*begin,output+begin,end-begin,true);
#endif
        for(auto i=begin;i<end;++i){std::uint16_t word;std::memcpy(&word,bytes+2*i,2);output[i]=std::uint16_t((word>>8)|(word<<8));}
    });
}
ResizeRangeInfo half_range_into(const std::uint16_t* input,std::size_t count,std::uint16_t* lower,std::uint16_t* upper) {
    check_count(count);if(!count)return {};
    if(!input || !lower || !upper)throw std::invalid_argument("nonnull half range storage required");
    const auto workers=active_workers(count);std::vector<Partial> partials{std::size_t(workers)};
    partition(count,workers,[&](std::size_t begin,std::size_t end,std::size_t id) {
        auto& state=partials[id];
#ifdef SKIMAGEXX_AVX2
        if(has_avx2())begin+=simd::half_range(input+begin,end-begin,&state.low,&state.high,state.info);
#endif
        for(auto i=begin;i<end;++i)add(state,input[i]);
    });
    Partial result;
    for(const auto& state:partials) {
        result.info.has_nan|=state.info.has_nan;result.info.positive_zero|=state.info.positive_zero;result.info.negative_zero|=state.info.negative_zero;
        if(state.info.has_value){add(result,state.low);add(result,state.high);}
    }
    *lower=result.info.has_value ? result.low : 0x7e00;*upper=result.info.has_value ? result.high : 0x7e00;return result.info;
}
void half_clip_into(std::uint16_t* data,std::size_t count,std::uint16_t lower,std::uint16_t upper,int policy) {
    if(policy<0 || policy>1)throw std::invalid_argument("invalid half clip policy");
    check_count(count);if(!count)return;
    if(!data)throw std::invalid_argument("nonnull half clip storage required");
    const bool low_nan=half_nan(lower),high_nan=half_nan(upper),fill=policy==1 && (low_nan||high_nan);
    const auto nan=low_nan ? lower : upper;
    partition(count,active_workers(count),[&](std::size_t begin,std::size_t end,auto) {
#ifdef SKIMAGEXX_AVX2
        if(has_avx2())begin+=simd::half_clip(data+begin,end-begin,lower,upper,policy);
#endif
        for(auto i=begin;i<end;++i) {
            auto value=data[i];if(half_nan(value))continue;
            if(fill){data[i]=nan;continue;}
            if(!low_nan && half_less(value,lower))value=lower;
            if(!high_nan && half_less(upper,value))value=upper;
            data[i]=value;
        }
    });
}
}
