#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace skimagexx {

struct ContourPoint {
    double row;
    double column;
};

using Contour = std::vector<ContourPoint>;
using Contours = std::vector<Contour>;

struct ContourOptions {
    double level;
    bool default_level;
    bool connect_high;
};

// Contiguous native-endian input and optional byte mask with equal extents.
// Both axes are at least 2 and the product is at most 131072 elements.
// Floating samples and explicit levels are finite in [-2^40,2^40] under
// round-to-nearest. The default range midpoint is supported for double only.
// The result preserves row-major cell order and first-created contour order,
// with the lower-valued side on the left. False leaves output unchanged.
bool find_contours_shape_supported(std::size_t rows, std::size_t columns);
bool find_contours(const double* image, const std::uint8_t* mask, std::size_t rows,
                   std::size_t columns, ContourOptions, Contours& output);
bool find_contours(const float* image, const std::uint8_t* mask, std::size_t rows,
                   std::size_t columns, ContourOptions, Contours& output);
bool find_contours(const std::uint8_t* image, const std::uint8_t* mask, std::size_t rows,
                   std::size_t columns, ContourOptions, Contours& output);

}  // namespace skimagexx
