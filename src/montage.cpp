#include "montage.hpp"
#include "execution.hpp"
#include <algorithm>
#include <cstring>
#include <type_traits>

namespace skimagexx::detail {
void montage_into(const MontageInput& input,MontageGrid grid,
                  const void* typed_channel_fill,MontageOutput output,int workers) {
    const auto channels=input.shape[3],pixel_bytes=channels*input.itemsize;
    const auto row_bytes=output.columns*pixel_bytes;
    const auto output_bytes=output.rows*row_bytes;
    if(!output_bytes)return;
    const auto* fill=static_cast<const unsigned char*>(typed_channel_fill);
    auto* destination=static_cast<unsigned char*>(output.data);
    const bool zero=std::all_of(fill,fill+pixel_bytes,[](unsigned char value){return value==0;});
    partition(output.rows,output_bytes>=parallel_min?workers:1,
              [&](auto begin,auto end,auto) {
        if(begin==end)return;
        auto* start=destination+begin*row_bytes;
        const auto bytes=(end-begin)*row_bytes;
        if(zero){std::memset(start,0,bytes);return;}
        std::memcpy(start,fill,pixel_bytes);
        // Repeat an initialized whole-pixel pattern; each memcpy has disjoint
        // source/destination spans and stays within this worker's fresh output.
        for(std::size_t initialized=pixel_bytes;initialized<bytes;) {
            const auto n=std::min(initialized,bytes-initialized);
            std::memcpy(start+initialized,start,n);initialized+=n;
        }
    });
    const auto images=input.shape[0],height=input.shape[1],width=input.shape[2];
    if(!images||!height||!width)return;
    const auto* source=static_cast<const unsigned char*>(input.data);
    const auto s_image=input.strides[0],s_row=input.strides[1];
    const auto s_column=input.strides[2],s_channel=input.strides[3];
    const bool channel_run=channels==1||s_channel==std::ptrdiff_t(input.itemsize);
    const bool row_run=channel_run&&(width==1||s_column==std::ptrdiff_t(pixel_bytes));
    const auto copy_bytes=images*height*width*pixel_bytes;
    partition(images*height,copy_bytes>=parallel_min?workers:1,
              [&](auto begin,auto end,auto) {
        for(auto index=begin;index<end;++index) {
            const auto image=index/height,row=index%height;
            const auto out_row=grid.padding+(image/grid.columns)*(height+grid.padding)+row;
            const auto out_column=grid.padding+(image%grid.columns)*(width+grid.padding);
            auto* target=destination+(out_row*output.columns+out_column)*pixel_bytes;
            const auto* from=source+std::ptrdiff_t(image)*s_image+std::ptrdiff_t(row)*s_row;
            if(row_run){std::memcpy(target,from,width*pixel_bytes);continue;}
            const auto copy_row=[&](auto size) {
                constexpr auto fixed=decltype(size)::value;
                for(std::size_t column=0;column<width;++column) {
                    const auto* pixel=from+std::ptrdiff_t(column)*s_column;
                    auto* to=target+column*pixel_bytes;
                    if(channel_run) {
                        std::memcpy(to,pixel,fixed?fixed:pixel_bytes);
                        continue;
                    }
                    for(std::size_t channel=0;channel<channels;++channel)
                        std::memcpy(to+channel*input.itemsize,
                                    pixel+std::ptrdiff_t(channel)*s_channel,
                                    fixed?fixed:input.itemsize);
                }
            };
            // Constant memcpy sizes permit unaligned inline word copies without
            // imposing typed alignment or aliasing requirements on POD storage.
            switch(channel_run?pixel_bytes:input.itemsize) {
                case 1:copy_row(std::integral_constant<std::size_t,1>{});break;
                case 2:copy_row(std::integral_constant<std::size_t,2>{});break;
                case 4:copy_row(std::integral_constant<std::size_t,4>{});break;
                case 8:copy_row(std::integral_constant<std::size_t,8>{});break;
                case 16:copy_row(std::integral_constant<std::size_t,16>{});break;
                default:copy_row(std::integral_constant<std::size_t,0>{});break;
            }
        }
    });
}
}
