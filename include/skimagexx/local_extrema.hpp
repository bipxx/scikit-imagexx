#pragma once

#include <cstddef>
#include <cstdint>

namespace skimagexx {

enum class LocalExtremaKind { maxima, minima };
enum class LocalExtremaNeighbors { four, eight };

// Ordered serial plateau traversal for a two-dimensional integer image.
// Input is contiguous. The caller supplies disjoint, valid input and output
// storage. Output has one byte (0 or 1) per pixel and may have row padding.
// Supported T are signed and unsigned 8-, 16-, 32- and 64-bit integers.
template<class T>
void local_extrema_2d_into(const T* image, std::size_t image_count,
                           std::size_t rows, std::size_t columns,
                           LocalExtremaKind kind, LocalExtremaNeighbors neighbors,
                           bool allow_borders, std::uint8_t* mask,
                           std::size_t mask_count, std::size_t mask_stride);

}  // namespace skimagexx
