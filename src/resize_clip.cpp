#include "resize_clip.hpp"
#include "skimagexx/image.hpp"
#include "execution.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace skimagexx::detail {
namespace {
template<class T>
void clip(T* data,std::size_t count,PixelType type,const void* lower,const void* upper,ResizeClipPolicy policy) {
    const bool legacy=policy!=ResizeClipPolicy::modern;
    const T low=*static_cast<const T*>(lower),high=*static_cast<const T*>(upper);
    bool fill_nan=false;T nan{};
    if constexpr(std::is_floating_point_v<T>) {
        fill_nan=policy!=ResizeClipPolicy::legacy && (std::isnan(low) || std::isnan(high));
        nan=std::isnan(low) ? low : high;
    }
    partition(count,active_workers(count),[&](std::size_t begin,std::size_t end,auto) {
        if constexpr(std::is_floating_point_v<T>) {
            if(fill_nan) {
                if(policy==ResizeClipPolicy::modern)std::fill(data+begin,data+end,nan);
                else {
#ifdef SKIMAGEXX_AVX2
                    if(has_avx2())begin+=simd::resize_clip_nan(data+begin,end-begin,type,&nan);
#endif
                    for(auto i=begin;i<end;++i)if(!std::isnan(data[i]))data[i]=nan;
                }
                return;
            }
        }
#ifdef SKIMAGEXX_AVX2
        if(has_avx2())begin+=simd::resize_clip(data+begin,end-begin,type,&low,&high,legacy);
#else
        (void)type;
#endif
        for(auto i=begin;i<end;++i) {
            T value=data[i];
            if constexpr(std::is_floating_point_v<T>) {
                // Keep selection in the integer representation: MSVC can
                // retain +0 on a <= -0 assignment even under /fp:strict.
                // NumPy's legacy policy requires the exact bound bits.
                using Bits=std::conditional_t<sizeof(T)==4,std::uint32_t,std::uint64_t>;
                Bits bits,lo,hi;std::memcpy(&bits,&value,sizeof(T));
                std::memcpy(&lo,&low,sizeof(T));std::memcpy(&hi,&high,sizeof(T));
                const Bits below=Bits(0)-Bits(legacy ? value<=low : value<low);
                bits=(bits&~below)|(lo&below);std::memcpy(&value,&bits,sizeof(T));
                const Bits above=Bits(0)-Bits(legacy ? value>=high : value>high);
                bits=(bits&~above)|(hi&above);std::memcpy(data+i,&bits,sizeof(T));
                continue;
            } else {if(value<low)value=low;if(value>high)value=high;}
            data[i]=value;
        }
    });
}
}
void resize_clip_into(void* image,std::size_t count,PixelType type,const void* lower,const void* upper,ResizeClipPolicy policy) {
    if(int(policy)<0 || int(policy)>2)throw std::invalid_argument("invalid clip policy");
    const auto bytes=pixel_size(type);
    if(count>std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/bytes)throw std::overflow_error("clip dimensions overflow");
    if(!count)return;
    if(!image || !lower || !upper)throw std::invalid_argument("nonnull clip storage and bounds required");
    switch(type) {
        case PixelType::boolean:clip<std::uint8_t>(static_cast<std::uint8_t*>(image),count,type,lower,upper,policy);break;
        #define CLIP(tag,T) case PixelType::tag:clip(static_cast<T*>(image),count,type,lower,upper,policy);break;
        CLIP(uint8,std::uint8_t) CLIP(int8,std::int8_t) CLIP(uint16,std::uint16_t) CLIP(int16,std::int16_t)
        CLIP(uint32,std::uint32_t) CLIP(int32,std::int32_t) CLIP(uint64,std::uint64_t) CLIP(int64,std::int64_t)
        CLIP(float32,float) CLIP(float64,double)
        #undef CLIP
        default:throw std::invalid_argument("unsupported clip dtype");
    }
}
}
