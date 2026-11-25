#pragma once

#include "convert.hpp"

#include <array>
#include <cstddef>
#include <optional>
#include <vector>

namespace skimagexx {

enum class BlobMethod { dog, log, doh };

struct BlobDetectionOptions {
    BlobMethod method;
    // One row/column sigma pair per Gaussian, LoG response, or Hessian response.
    // DoG uses one more Gaussian than its number of response scales.
    std::vector<std::array<double, 2>> scales;
    double dog_normalization = 1;
    std::optional<double> threshold;
    std::optional<double> threshold_relative;
    double overlap = 0.5;
    std::array<std::size_t, 2> exclude_border{};
};

struct BlobDetection {
    std::size_t row;
    std::size_t column;
    std::array<double, 2> sigma;
};

// Dense aligned row-major uint8, float32, or float64 image. The result owns its
// detections in descending response order, with ties resolved in raster/scale
// order. Returns nullopt for nonfinite or out-of-domain arithmetic, or when
// overlap pruning depends on the upstream spatial provider's pair ordering.
std::optional<std::vector<BlobDetection>> blob_detect(
    const void* image, std::size_t count, PixelType type, std::size_t height,
    std::size_t width, const BlobDetectionOptions& options);

}  // namespace skimagexx
