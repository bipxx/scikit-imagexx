#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace skimagexx {

enum class SkeletonizeInput { boolean, uint8 };

// Zhang thinning with a zero border and simultaneous deletions per phase.
// Both extents are within [1, 512]. Input is contiguous row-major storage.
// Boolean bytes must be 0 or 1. UInt8 treats every nonzero byte as foreground.
// Success replaces output with one row-major byte per pixel. False preserves it.
bool skeletonize_zhang(
    const std::uint8_t* image, std::size_t rows, std::size_t columns,
    SkeletonizeInput kind, std::vector<std::uint8_t>& output);

}  // namespace skimagexx
