#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace skimagexx {

struct ProbabilisticHoughPoint {
    int x;
    int y;
};

struct ProbabilisticHoughSegment {
    ProbabilisticHoughPoint first;
    ProbabilisticHoughPoint second;
};

// Ordered progressive line transform on a complete row-major byte mask.
// Each extent is 1..256 and at most 4096 bytes may be nonzero. The
// permutation enumerates every foreground index exactly once; the search
// visits it in reverse. Supply up to 256 sine/cosine pairs with a valid
// direction and bounded projection. Threshold is 1..4096 and length and
// gap are 0..256. Build without FP contraction and call under round-to-nearest.
// Invalid spans or parameters throw std::invalid_argument. Inputs remain
// unchanged, and the returned vector owns its endpoints.
std::vector<ProbabilisticHoughSegment> probabilistic_hough_line_ordered(
    const std::uint8_t* image, std::size_t image_count, int height, int width,
    const double* cosines, const double* sines, std::size_t angle_count,
    const std::intptr_t* permutation, std::size_t count, int threshold, int line_length, int line_gap);

}  // namespace skimagexx
