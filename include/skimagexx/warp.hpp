#pragma once

#include "skimagexx/gaussian_nd.hpp"
#include <array>
#include <cstddef>

namespace skimagexx {
struct WarpShape {
    std::size_t rows, cols, channels = 1;
};
enum class WarpMapping { matrix, swirl, polar_linear, polar_log };
struct WarpMap {
    WarpMapping kind;
    // Matrix in row-major order. Swirl: x, y, rotation, strength, decay.
    // Polar: x, y, angular scale, radial scale.
    std::array<double, 9> params;
    // Optional homogeneous results for nearest matrix sampling. Triples
    // (x, y, w) are in column-major pixel order, before perspective division.
    const double* homogeneous = nullptr;
};
struct WarpOptions {
    int order = 1;
    GaussianBorder border = GaussianBorder::constant;
    double cval = 0;
    bool clip = true;
    bool preserve_range = false;
    bool legacy_reflect = false;
    bool round_clip_cval = true;
};

// Interleaved contiguous images and disjoint output. Orders 1/3 with a
// matrix use the homography interpolation path. Other maps use ndimage-style
// sampling. Generic spline sampling requires a single image plane.
// Output retains the input type, except uint8 with order > 0 uses double.
// Returns false for mappings outside the supported numerical domain.
bool warp_into(const void* input, void* output, PixelType type, WarpShape source,
               WarpShape target, const WarpMap& map, const WarpOptions& options = {});

// Planar output in (row, column[, band]) order. Color coordinates repeat
// across bands, and the third plane contains the band indices.
template<class T>
bool warp_coords_into(T* output, WarpShape shape, bool color, const WarpMap& map);
extern template bool warp_coords_into(float*, WarpShape, bool, const WarpMap&);
extern template bool warp_coords_into(double*, WarpShape, bool, const WarpMap&);
}
