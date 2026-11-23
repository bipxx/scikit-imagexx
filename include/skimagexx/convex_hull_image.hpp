#pragma once

#include <cstddef>
#include <cstdint>

namespace skimagexx {

inline constexpr std::size_t convex_hull_image_max_extent = 512;

// Both buffers have height * width accessible bytes in C order.
// Input bytes must be 0 or 1. Empty and degenerate hulls return false
// without writing output; callers provide their own compatibility behavior.
bool convex_hull_image_into(const std::uint8_t* image, std::size_t height, std::size_t width,
                            bool offset_coordinates, bool include_borders, std::uint8_t* output);

} // namespace skimagexx
