#pragma once
#include "convert.hpp"
#include <cstddef>
#include <cstdint>

namespace skimagexx {
struct LabelSummary {
    std::size_t regions=1;
    std::uint64_t background=0;
    bool has_background=false;
};
// Contiguous raw labels (possibly unaligned or byte swapped), C-order ranks.
// Overlay uses upstream's minimal unsigned cast and inverse-rank wrapping;
// average mode preserves the original numeric label groups. Returns false for
// arithmetic domains which require explicit compatibility handling.
// Caller provides count uint64 output elements; input/output must not overlap.
bool label_ranks(const void* input,std::size_t count,PixelType type,bool swap,
    std::uint64_t background,bool overlay,std::uint64_t* ranks,LabelSummary& summary);
// Validate underlay before arithmetic. Overlay floats are bounded to [0,1];
// average floats to finite magnitude <=1e6. Only u8/f32/f64 underlays currently.
bool label_image_valid(const void*,std::size_t elements,PixelType,bool overlay);
void label_saturation(void*,std::size_t pixels,PixelType,double saturation);
// Palette is pre-scaled constant metadata, background first then cycling colors.
// Outputs are aligned, contiguous count*3 elements and do not overlap inputs.
// Inputs must pass label_image_valid; average background is pre-cast to image dtype.
void label_overlay(const std::uint64_t*,std::size_t,const void* image,PixelType,bool rgb,
    const double* palette,std::size_t colors,double image_alpha,double image_background,
    double label_background,bool remove_background,std::uint64_t background,
    void* output,bool output_single);
void label_average(const std::uint64_t*,std::size_t,std::size_t regions,
    const void* image,PixelType,bool rgb,const void* background_rgb,bool has_background,void* output,
    std::size_t reduction_block=0);
}
