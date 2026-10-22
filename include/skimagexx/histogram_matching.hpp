#pragma once
#include "convert.hpp"
namespace skimagexx {
// Contiguous channel-last real buffers, with independent spatial sample counts.
// Source/template types may differ. Output has source_count*channels elements;
// buffers must not overlap. Float16 inputs may be losslessly prepared as float32.
// half_round reproduces channel-wise binary16 storage before float32 promotion.
// Returned bits:1 invalid integer cast,2 floating overflow,4 underflow.
unsigned match_histograms_into(const void* source,const void* reference,void* output,
    std::size_t source_count,std::size_t reference_count,std::size_t channels,
    PixelType source_type,PixelType reference_type,PixelType output_type,
    bool unsigned_source=false,bool half_round=false,bool swap_output=false,unsigned* channel_flags=nullptr);
}
