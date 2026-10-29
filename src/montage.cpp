#include "montage_simd.hpp"
#include "execution.hpp"
#include <algorithm>
#include <cstring>

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
    // Reused for whole padding rows and the disjoint holes between tile spans.
    const auto fill_span=[&](unsigned char* start,std::size_t bytes) {
        if(!bytes)return;
        if(bytes==1){*start=*fill;return;}
        if(zero||pixel_bytes==1){std::memset(start,*fill,bytes);return;}
        std::memcpy(start,fill,pixel_bytes);
        for(std::size_t initialized=pixel_bytes;initialized<bytes;) {
            const auto n=std::min(initialized,bytes-initialized);
            std::memcpy(start+initialized,start,n);initialized+=n;
        }
    };
    const auto images=input.shape[0],height=input.shape[1],width=input.shape[2];
    const auto* source=static_cast<const unsigned char*>(input.data);
    const auto s_image=input.strides[0],s_row=input.strides[1];
    const auto s_column=input.strides[2],s_channel=input.strides[3];
    const bool channel_run=channels==1||s_channel==std::ptrdiff_t(input.itemsize);
    const bool row_run=channel_run&&(width==1||s_column==std::ptrdiff_t(pixel_bytes));
    const bool words32=channels==1&&input.itemsize==4;
    const bool words64=channels==1&&input.itemsize==8;
#ifdef SKIMAGEXX_AVX2
    const bool stride2_simd=words32&&s_column==8&&has_avx2();
#endif
    const auto tile_bytes=width*pixel_bytes,padding_bytes=grid.padding*pixel_bytes;
    const auto row_step=height+grid.padding;
    partition(output.rows,output_bytes>=parallel_min?workers:1,
              [&](auto begin,auto end,auto) {
        if(!images||!height||!width) {
            fill_span(destination+begin*row_bytes,(end-begin)*row_bytes);return;
        }
        for(auto y=begin;y<end;++y) {
            auto* start=destination+y*row_bytes;
            if(y<grid.padding){fill_span(start,row_bytes);continue;}
            const auto local=y-grid.padding,tile_row=local/row_step,row=local%row_step;
            const auto first=tile_row*grid.columns;
            if(tile_row>=grid.rows||row>=height||first>=images) {
                fill_span(start,row_bytes);continue;
            }
            const auto occupied=std::min(grid.columns,images-first);
            auto* target=start;
            fill_span(target,padding_bytes);target+=padding_bytes;
            for(std::size_t tile=0;tile<occupied;++tile) {
                const auto* from=source+std::ptrdiff_t(first+tile)*s_image+std::ptrdiff_t(row)*s_row;
                if(row_run)std::memcpy(target,from,tile_bytes);
#ifdef SKIMAGEXX_AVX2
                else if(stride2_simd)simd::montage_stride2_words32(from,target,width);
#endif
                else if(words32) {
                    for(std::size_t column=0;column<width;++column)
                        std::memcpy(target+column*4,from+std::ptrdiff_t(column)*s_column,4);
                } else if(words64) {
                    for(std::size_t column=0;column<width;++column)
                        std::memcpy(target+column*8,from+std::ptrdiff_t(column)*s_column,8);
                } else {
                    for(std::size_t column=0;column<width;++column) {
                        const auto* pixel=from+std::ptrdiff_t(column)*s_column;
                        auto* to=target+column*pixel_bytes;
                        if(channel_run){std::memcpy(to,pixel,pixel_bytes);continue;}
                        for(std::size_t channel=0;channel<channels;++channel)
                            std::memcpy(to+channel*input.itemsize,pixel+std::ptrdiff_t(channel)*s_channel,input.itemsize);
                    }
                }
                target+=tile_bytes;
                if(tile+1<occupied){fill_span(target,padding_bytes);target+=padding_bytes;}
            }
            // Includes the final padding and every unused tile in this row.
            fill_span(target,std::size_t(start+row_bytes-target));
        }
    });
}
}
