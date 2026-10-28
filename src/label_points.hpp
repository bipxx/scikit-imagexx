#pragma once
#include <cstddef>
#include <cstdint>

namespace skimagexx::detail {
struct LabelPointsInput {
    const std::int64_t* const* axes;
    const std::ptrdiff_t* axis_strides;
    const std::size_t* dimensions;
    std::size_t ndim;
    std::size_t point_count;
    std::size_t output_count;
};
// Boundary invariants: ndim >= 1; stable native int64 axis views, each with
// point_count addressable elements at its signed BYTE stride; all indices
// already satisfy 0 <= index < dimensions[axis]; output_count is the exact
// addressable product. Output is disjoint, contiguous uint64 storage; labels
// point+1 fit uint64, workers >= 1. Empty output implies no points.
// Zero-fill may use independent workers; scatter stays in point order so the
// last occurrence of a repeated coordinate wins. No floating arithmetic.
void label_points_into(const LabelPointsInput& input, std::uint64_t* output,
                       int workers);
}

namespace skimagexx::simd {
void label_points_zero(std::uint64_t* output, std::size_t count);
}
