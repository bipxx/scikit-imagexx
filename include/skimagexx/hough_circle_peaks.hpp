#pragma once

#include <cstddef>
#include <optional>
#include <vector>

namespace skimagexx {

struct HoughCirclePeak {
    double vote;
    int row;
    int column;
};

// Return local peaks before NumPy's optional per-plane ranking and before
// cross-radius pruning. The input is a complete C-order float64 plane with
// extents 1..256, nonnegative coordinates for spacing, and x distance no
// greater than its width. Distances are at most 32. The threshold is finite
// and nonnegative or absent, which uses half the plane maximum.
// Invalid storage or controls throw std::invalid_argument. A disengaged
// optional reports unsupported vote values or floating-point rounding mode.
std::optional<std::vector<HoughCirclePeak>> hough_circle_peak_indices(const double* plane,
    std::size_t count, int height, int width, int min_xdistance, int min_ydistance,
    std::optional<double> threshold = std::nullopt);

}  // namespace skimagexx
