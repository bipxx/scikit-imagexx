#pragma once
#include "skimagexx/convert.hpp"
#include <cstddef>

namespace skimagexx {

// Uniform-window, grayscale SSIM score on aligned C-order images. Supports
// float32 and float64, odd windows from 3 to 255, and data_range in [0.5, 4].
// Pixels must be finite and within [-data_range, data_range]. Nonzero
// magnitudes below 1e-8 and unsupported floating environments are declined.
// Inputs may alias. False leaves score untouched.
bool structural_similarity_score(const void* left, const void* right, PixelType type,
                                  std::size_t rows, std::size_t cols,
                                  std::size_t window, double data_range,
                                  bool use_sample_covariance, double& score);

} // namespace skimagexx
