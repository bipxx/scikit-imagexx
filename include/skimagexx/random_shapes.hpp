#pragma once

#include <cstddef>
#include <cstdint>

namespace skimagexx {

struct RandomShapeIndexView {
    const std::intptr_t* data;
    std::size_t stride;
};

// Paint an ordered, prepared coordinate list into C-order HWC uint8 storage.
// Image extents are 1..256 and channels are 1..8. Occupancy is a separate
// C-order byte plane with one byte per pixel. Row/column strides, measured in
// index elements, may be one or two (including interleaved coordinates).
// A false return means a disallowed overlap and leaves both outputs unchanged.
// Invalid buffers or coordinates throw before writes. Read-only spans can
// overlap each other. Neither writable span can alias another argument.
bool random_shape_paint_into(std::uint8_t* image, std::size_t image_count,
    std::uint8_t* occupied, std::size_t occupied_count, int height, int width, int channels,
    RandomShapeIndexView rows, RandomShapeIndexView columns, std::size_t count,
    const std::uint8_t* color, std::size_t color_count, bool allow_overlap);

}  // namespace skimagexx
