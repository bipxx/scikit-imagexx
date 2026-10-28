#include "montage_simd.hpp"
#include "execution.hpp"

namespace skimagexx::detail {
void montage_into(const MontageInput& input,MontageGrid grid,
                  const void* typed_channel_fill,MontageOutput output,int workers) {
#ifdef SKIMAGEXX_AVX2
    if(input.shape[3]==1&&input.itemsize==4&&input.strides[2]==8&&has_avx2()) {
        // Reuse the unchanged generic fill exactly once, without copying tiles.
        auto fill_only=input;fill_only.shape[0]=0;
        montage_generic_into(fill_only,grid,typed_channel_fill,output,workers);
        const auto images=input.shape[0],height=input.shape[1],width=input.shape[2];
        if(!output.rows||!output.columns||!images||!height||!width)return;
        auto* destination=static_cast<unsigned char*>(output.data);
        const auto* source=static_cast<const unsigned char*>(input.data);
        const auto row_bytes=output.columns*4,copy_bytes=images*height*width*4;
        partition(images*height,copy_bytes>=parallel_min?workers:1,
                  [&](auto begin,auto end,auto) {
            for(auto index=begin;index<end;++index) {
                const auto image=index/height,row=index%height;
                const auto out_row=grid.padding+(image/grid.columns)*(height+grid.padding)+row;
                const auto out_column=grid.padding+(image%grid.columns)*(width+grid.padding);
                const auto* from=source+std::ptrdiff_t(image)*input.strides[0]
                                       +std::ptrdiff_t(row)*input.strides[1];
                simd::montage_stride2_words32(from,destination+out_row*row_bytes+out_column*4,width);
            }
        });
        return;
    }
#endif
    montage_generic_into(input,grid,typed_channel_fill,output,workers);
}
}
