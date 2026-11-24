#pragma once

#include "convert.hpp"

#include <cstddef>

namespace skimagexx {

// Resize a complete C-order uint8 or uint16 plane using the grid-intersection
// model and a separate dense C-order double output. Both input extents are
// 16..2048. For preserve_range, both complete block factors are 2, 4 or 8.
// For normalized values, both factors must be 2. Use round-to-nearest in both
// modes. Element counts are checked separately from shape.
// Returns false before output writes for unsupported input type, geometry or
// floating-point mode; invalid or overlapping storage throws.
bool resize_local_mean_into(const void* source, PixelType type, std::size_t source_count,
    int height, int width, double* output, std::size_t output_count, int rows, int columns, bool preserve_range);

}  // namespace skimagexx
