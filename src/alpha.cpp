#include "skimagexx/alpha.hpp"
#include "skimagexx/image.hpp"
#include "alpha_detail.hpp"
#include "execution.hpp"
#include "simd.hpp"
#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>
namespace skimagexx {
namespace {
std::size_t checked_pixels(std::size_t rows,std::size_t columns,std::size_t item_bytes,int channels) {
    const auto limit=std::size_t(std::numeric_limits<std::ptrdiff_t>::max());
    if(columns && rows>limit/columns)throw std::overflow_error("image dimensions overflow");
    const auto count=rows*columns;
    if(item_bytes && (item_bytes>limit/std::size_t(channels) || count>limit/(item_bytes*std::size_t(channels))))
        throw std::overflow_error("expanded image size overflow");
    return count;
}
template<class Input,bool Boolean=false> void composite(const void* image,void* output,std::size_t rows,std::size_t columns,
    const double* background,const std::ptrdiff_t* offsets,std::ptrdiff_t pixel_step,std::ptrdiff_t channel_step) {
    using F=detail::ColorOutput<Input>;
    const auto count=checked_pixels(rows,columns,sizeof(F),3);
    if(background)for(int j=0;j<3;++j)if(background[j]<0 || background[j]>1)throw std::invalid_argument("background values must be between zero and one");
    if(!count)return;
    if(!image || !output || !background || !offsets)throw std::invalid_argument("nonnull RGBA storage required");
    const auto* source=static_cast<const unsigned char*>(image);auto* target=static_cast<F*>(output);
    const F bg[]={F(background[0]),F(background[1]),F(background[2])};
    detail::partition(count,detail::active_workers(count),[&](auto begin,auto end,auto) {
        while(begin<end) {
            const auto row=begin/columns,column=begin%columns,span=std::min(end-begin,columns-column);
            const auto* start=source+offsets[row]+std::ptrdiff_t(column)*pixel_step;
            std::size_t i=0;
#ifdef SKIMAGEXX_AVX2
            if(has_avx2())i=simd::rgba_composite(start,target+3*begin,span,
                std::is_same_v<Input,float> ? PixelType::float32 :
                std::is_same_v<Input,double> ? PixelType::float64 :
                Boolean ? PixelType::boolean :
                std::is_same_v<Input,std::uint8_t> ? PixelType::uint8 :
                std::is_same_v<Input,std::int8_t> ? PixelType::int8 :
                std::is_same_v<Input,std::uint16_t> ? PixelType::uint16 :
                std::is_same_v<Input,std::int16_t> ? PixelType::int16 :
                std::is_same_v<Input,std::uint32_t> ? PixelType::uint32 :
                std::is_same_v<Input,std::int32_t> ? PixelType::int32 :
                std::is_same_v<Input,std::uint64_t> ? PixelType::uint64 : PixelType::int64,
                background,pixel_step,channel_step);
#endif
            for(;i<span;++i) {
                const auto* pixel=start+std::ptrdiff_t(i)*pixel_step;
                const F alpha=detail::alpha_normalized<Input,Boolean>(pixel+3*channel_step);
                for(int channel=0;channel<3;++channel) {
                    const F color=detail::alpha_normalized<Input,Boolean>(pixel+std::ptrdiff_t(channel)*channel_step);
                    const F value=F(F(F(F(1)-alpha)*bg[channel])+F(alpha*color));
                    target[3*(begin+i)+std::size_t(channel)]=detail::alpha_clip(value);
                }
            }
            begin+=span;
        }
    });
}
}
void gray_expand_rows_into(const void* image,const void* alpha,void* output,std::size_t rows,std::size_t columns,
    std::size_t item_bytes,int channels,const std::ptrdiff_t* image_offsets,const std::ptrdiff_t* alpha_offsets,
    std::ptrdiff_t image_step,std::ptrdiff_t alpha_step,std::size_t swap_component,const std::size_t* byte_order) {
    if(channels!=3 && channels!=4)throw std::invalid_argument("three or four output channels required");
    if(swap_component && (swap_component!=2 && swap_component!=4 && swap_component!=8 || item_bytes%swap_component))throw std::invalid_argument("invalid byte-swap component");
    if(byte_order)for(std::size_t j=0;j<item_bytes;++j)if(byte_order[j]>=item_bytes)throw std::invalid_argument("byte permutation exceeds item storage");
    const auto count=checked_pixels(rows,columns,item_bytes,channels);
    if(!count || !item_bytes)return;
    if(!image || !output || !image_offsets || (channels==4 && (!alpha || !alpha_offsets)))
        throw std::invalid_argument("nonnull expansion storage required");
    const auto* source=static_cast<const unsigned char*>(image);
    const auto* opacity=static_cast<const unsigned char*>(alpha);auto* target=static_cast<unsigned char*>(output);
    detail::partition(count,detail::active_workers(count),[&](auto begin,auto end,auto) {
        std::vector<unsigned char> swapped(swap_component || byte_order ? item_bytes : 0);
        while(begin<end) {
            const auto row=begin/columns,column=begin%columns,span=std::min(end-begin,columns-column);
            const auto* start=source+image_offsets[row]+std::ptrdiff_t(column)*image_step;
            const auto* a=channels==4 ? opacity+alpha_offsets[row]+std::ptrdiff_t(column)*alpha_step : nullptr;
            auto* out=target+begin*std::size_t(channels)*item_bytes;
            std::size_t i=0;
#ifdef SKIMAGEXX_AVX2
            if(has_avx2())i=byte_order ? simd::gray_expand_mapped(start,a,out,span,item_bytes,channels,image_step,alpha_step,byte_order) : simd::gray_expand(start,a,out,span,item_bytes,channels,image_step,alpha_step,swap_component);
#endif
            for(;i<span;++i) {
                const auto* value=start+std::ptrdiff_t(i)*image_step;
                if(byte_order) {
                    for(std::size_t byte=0;byte<item_bytes;++byte)swapped[byte]=value[byte_order[byte]];
                    value=swapped.data();
                } else if(swap_component) {
                    for(std::size_t byte=0;byte<item_bytes;++byte)swapped[byte]=value[(byte/swap_component)*swap_component+swap_component-1-byte%swap_component];
                    value=swapped.data();
                }
                for(int channel=0;channel<3;++channel)std::memcpy(out+(i*std::size_t(channels)+std::size_t(channel))*item_bytes,value,item_bytes);
                if(channels==4) {
                    auto* destination=out+(i*4+3)*item_bytes;const auto* source_alpha=a+std::ptrdiff_t(i)*alpha_step;
                    if(byte_order)for(std::size_t byte=0;byte<item_bytes;++byte)destination[byte]=source_alpha[byte_order[byte]];
                    else if(swap_component)for(std::size_t byte=0;byte<item_bytes;++byte)destination[byte]=source_alpha[(byte/swap_component)*swap_component+swap_component-1-byte%swap_component];
                    else std::memcpy(destination,source_alpha,item_bytes);
                }
            }
            begin+=span;
        }
    });
}
void rgba2rgb_rows_into(const void* image,void* output,std::size_t rows,std::size_t columns,PixelType type,
    const double* background,const std::ptrdiff_t* offsets,std::ptrdiff_t pixel_step,std::ptrdiff_t channel_step) {
    if(type==PixelType::boolean) {composite<std::uint8_t,true>(image,output,rows,columns,background,offsets,pixel_step,channel_step);return;}
#define ALPHA_CASE(tag,ctype) case PixelType::tag:composite<ctype>(image,output,rows,columns,background,offsets,pixel_step,channel_step);return;
    switch(type) {
    ALPHA_CASE(uint8,std::uint8_t) ALPHA_CASE(int8,std::int8_t) ALPHA_CASE(uint16,std::uint16_t) ALPHA_CASE(int16,std::int16_t)
    ALPHA_CASE(uint32,std::uint32_t) ALPHA_CASE(int32,std::int32_t) ALPHA_CASE(uint64,std::uint64_t) ALPHA_CASE(int64,std::int64_t)
    ALPHA_CASE(float32,float) ALPHA_CASE(float64,double)
    default:throw std::invalid_argument("unsupported RGBA input type");
    }
#undef ALPHA_CASE
}
}
