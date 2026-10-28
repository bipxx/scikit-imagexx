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
// point_count addressable elements at its signed BYTE stride; output_count is
// the exact
// addressable product. Output is disjoint, contiguous uint64 storage; labels
// point+1 fit uint64, workers >= 1. Output must remain private until acceptance.
// Zero-fill may use independent workers; scatter stays in point order so the
// last occurrence of a repeated coordinate wins. Bounds are checked once while
// computing offsets. false means discard the ENTIRE private output and use the
// literal provider; no partial output may escape. No floating arithmetic.
bool label_points_into(const LabelPointsInput& input, std::uint64_t* output,
                       int workers);
// Ordered assignment without clearing: other output elements remain unchanged.
// The same prepared-input and private-refusal invariants apply. For constructor
// semantics the caller supplies a zero-initialized output. No floating math.
bool label_points_assign(const LabelPointsInput& input, std::uint64_t* output,
                         int workers);
}

