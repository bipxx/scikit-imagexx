#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace skimagexx {

inline constexpr std::size_t medial_axis_max_extent = 192;

// Ordered medial-axis thinning of an owner-validated row-major Boolean mask.
// Distances are supplied externally for the whole shape. Unique tie priorities
// are a permutation indexed by foreground pixels in row-major order.
// Both extents are in [1, medial_axis_max_extent]. False preserves output.
bool medial_axis_ordered(
    const std::uint8_t* foreground, const double* distances, std::size_t distance_count,
    const std::int64_t* priorities, std::size_t priority_count,
    std::size_t rows, std::size_t columns, std::vector<std::uint8_t>& output);

}  // namespace skimagexx
