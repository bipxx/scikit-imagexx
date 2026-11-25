#pragma once

#include "gaussian_nd.hpp"

namespace skimagexx {

struct CannyOptions {
    // Empty skips smoothing. Otherwise supply the normalized symmetric
    // Gaussian weights in increasing offset order, including the center.
    std::vector<double> weights;
    GaussianBorder border = GaussianBorder::constant;
    double cval = 0;
    double low = 0.1;
    double high = 0.2;
    bool quantiles = false;
    // NumPy 2 compares a floating quantile at float64 precision.
    bool wide_quantile = true;
    // Gaussian grid-constant is an alias, but doesn't enable unmasked bleed.
    bool constant_bleed = true;
};

inline constexpr std::size_t canny_max_pixels = 16 * 1024 * 1024;
inline constexpr std::size_t canny_max_radius = 64;

// Positively sized contiguous, aligned row-major buffers. Image supports
// Boolean, integers through 32 bits and float32/float64. An optional mask is
// read as Boolean bytes. Output bytes have values 0 or 1 and must be disjoint.
// Absolute thresholds are expressed in converted image units. On rejection
// (invalid or unsupported configuration/data), output stays untouched.
bool canny_into(const void* image, PixelType type, const std::uint8_t* mask,
                std::size_t rows, std::size_t cols, const CannyOptions& options,
                std::uint8_t* output);

}  // namespace skimagexx
