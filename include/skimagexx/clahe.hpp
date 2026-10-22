#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace skimagexx {
// CLAHE for contiguous, aligned uint16 images, with normalized float output.
// clip_count is the absolute histogram limit (kernel element count disables
// clipping). Buffers must not overlap. The implementation supports 1..8 axes.
void clahe_into(const std::uint16_t* input,void* output,bool single,
    const std::vector<std::size_t>& shape,const std::vector<std::size_t>& kernel,
    std::size_t clip_count,std::size_t bins=256);
}
