#pragma once
#include "convert.hpp"
#include <vector>

namespace skimagexx {
enum class MorphologyOp { erosion, dilation, opening, closing, white_tophat, black_tophat,
                          binary_erosion, binary_dilation, binary_opening, binary_closing };
enum class MorphologyMode { reflect, constant, nearest, mirror, wrap, maximum, minimum, ignore };
struct MorphologyFootprint {
    // Unique row-major coordinates, relative to the already-padded footprint center.
    std::vector<std::ptrdiff_t> offsets;
    std::int64_t iterations = 1;
    bool scipy_separable = false; // Padded footprint is entirely nonzero.
};
struct MorphologyInfo { bool unsupported_nan = false, invalid_subtract = false, overflow_subtract = false; };
// Contiguous arrays, no normalization of grayscale pixels. Binary operations
// interpret nonzero as true and may write any supported numeric output dtype.
// Grayscale output dtype must equal input dtype. Overlapping buffers are safe.
// NaN grayscale input/constant fill is reported unsupported without modifying output.
MorphologyInfo morphology_into(const void* input, void* output, PixelType input_type, PixelType output_type,
    const std::vector<std::size_t>& shape, const std::vector<MorphologyFootprint>& footprints,
    MorphologyOp operation, MorphologyMode mode, double cval = 0);
} // namespace skimagexx
