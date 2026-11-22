#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace skimagexx {

struct HoughLinePeak {
    std::uint64_t votes;
    std::int32_t distance;
    std::int32_t angle;
};

// Serial peak suppression on an aligned, complete, C-order uint64 span.
// Height <= 1024, width <= 512, and 1 <= height * width < 262144.
// The distance radius and the angle radius after clamping to width are <= 32.
// Threshold is finite and nonnegative; nullopt uses one half of the maximum.
// Counts above 1048576 throw std::overflow_error. Invalid shape, span or controls
// throw std::invalid_argument. The caller keeps the input stable during the call.
// Returns all surviving peaks in suppression order. The caller owns any final
// provider-specific ordering or peak cap and indexing into coordinate vectors.
std::vector<HoughLinePeak> hough_line_peak_indices(
    const std::uint64_t* accumulator, std::size_t count, int height, int width,
    int min_distance, int min_angle, std::optional<double> threshold = std::nullopt);

}  // namespace skimagexx
