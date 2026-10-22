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
}
