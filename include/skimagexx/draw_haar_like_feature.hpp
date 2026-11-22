#pragma once

#include "convert.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace skimagexx {

struct HaarRectangle {
    std::int64_t row_start, column_start, row_end, column_end;
};
using HaarFeature = std::vector<HaarRectangle>;
using HaarFeatures = std::vector<HaarFeature>;

// Prepared C-order input and fresh three-channel output. Every endpoint
// uses the index rules of the corresponding array axis, including negatives.
// The input and output spans are disjoint. A float32 image returns float32;
// bool, uint8 and float64 return float64. Draw order is always serial.
bool draw_haar_like_feature_into(const void* input, std::size_t input_elements, PixelType type,
                                std::size_t rows, std::size_t columns, std::size_t channels,
                                void* output, std::size_t output_elements,
                                const HaarFeatures& features, const std::vector<std::size_t>* selection,
                                const std::array<double,3>& positive, const std::array<double,3>& negative,
                                double alpha);

namespace detail {
bool haar_coordinates_in_bounds(std::size_t rows, std::size_t columns, const HaarFeatures&);
}
}
