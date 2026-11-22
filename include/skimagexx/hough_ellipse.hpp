#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace skimagexx {

struct HoughEllipseDetection {
    std::intptr_t accumulator;
    double yc;
    double xc;
    double a;
    double b;
    double orientation;
};

struct HoughEllipseResult {
    bool has_foreground = false;
    std::vector<HoughEllipseDetection> detections;
};

// Ordered ellipse search on a complete C-order byte mask. Nonzero bytes
// mark edges. Extents are 0 to 64, with at most 128 foreground points.
// Use threshold in [0, 128], min_size in [1, 64], accuracy 0.75, 1, 1.5
// or 2, and an optional max_size in [1, 32]. Call under round-to-nearest.
// Invalid storage throws std::invalid_argument. An unsupported request
// returns a disengaged optional. An empty image has has_foreground == false.
// The returned vector owns its records and preserves endpoint-pair order.
std::optional<HoughEllipseResult> hough_ellipse_bounded(const std::uint8_t* image, std::size_t image_count,
    int height, int width, int threshold = 4, double accuracy = 1.0, int min_size = 4,
    std::optional<int> max_size = std::nullopt);

}  // namespace skimagexx
