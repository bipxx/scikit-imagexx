#pragma once

#include "convert.hpp"

#include <cstdint>
#include <vector>

namespace skimagexx {

struct SiftOptions {
    int upsampling = 2, n_octaves = 8, n_scales = 3;
    double sigma_min = 1.6, sigma_in = 0.5;
    double c_dog = 0.04 / 3, c_edge = 10;
    int n_bins = 36;
    double lambda_ori = 1.5, c_max = 0.8, lambda_descr = 6;
    int n_hist = 4, n_ori = 8;
};

struct SiftKeypoint {
    double row, col, sigma, orientation;
    int scale, octave;
};

struct SiftResult {
    std::vector<SiftKeypoint> keypoints;
    // Row-major array with n_hist * n_hist * n_ori columns if requested.
    std::vector<std::uint8_t> descriptors;
    bool promoted_positions = false;
    bool multiple_orientations = false;
    // Degenerate descriptor rows are zero-filled.
    bool descriptor_has_zero_norm = false;
};

// Native-endian aligned contiguous real image. The SDK caps the requested
// octave count to the image size, with a 12-pixel minimum for the last octave.
// Unsupported options or image dimensions report std::invalid_argument.
SiftResult sift_detect(const void* image, int height, int width, PixelType type,
                       const SiftOptions& options = {}, bool describe = false);
std::vector<std::uint8_t> sift_extract(const void* image, int height, int width,
                                      PixelType type, const SiftOptions& options,
                                      const std::vector<SiftKeypoint>& keypoints,
                                      bool* descriptor_has_zero_norm = nullptr);

} // namespace skimagexx
