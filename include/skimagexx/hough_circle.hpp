#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace skimagexx {

struct HoughCirclePlane {
    int height;
    int width;
    int padding;
};

// Bounded circle transform on positive image extents <= 512, with 1 to 16
// radii in [0, 64], at most 4194304 output cells and 67108864 worst-case visits.
// Plane order is radius order, including duplicates. Full output pads by the
// maximum radius. A disengaged result denotes an unsupported request.
std::optional<HoughCirclePlane> hough_circle_plane(int height, int width,
    const std::vector<int>& radii, bool full_output);

// Input is a complete C-order byte mask; each nonzero byte is an edge.
// Output is the complete, aligned, disjoint double span for radii * plane.
// Validation precedes writes. Normalization uses one repeated double addition
// per Bresenham visit, including duplicate visits. Call under round-to-nearest
// and keep buffers and radii stable until return.
void hough_circle_into(const std::uint8_t* image, std::size_t image_count, int height, int width,
    const std::vector<int>& radii, bool normalize, bool full_output, double* output, std::size_t output_count);

}  // namespace skimagexx
