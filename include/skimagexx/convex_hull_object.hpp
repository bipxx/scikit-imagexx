#pragma once

#include <cstddef>
#include <cstdint>

namespace skimagexx {

// Both buffers are disjoint and address height * width bytes in C order.
// Canonical binary images and connectivity 1 or 2 are supported within the
// planar hull limits. On false, the caller must discard the output.
bool convex_hull_object_into(const std::uint8_t* image, std::size_t height,
                             std::size_t width, int connectivity, std::uint8_t* output);

} // namespace skimagexx
