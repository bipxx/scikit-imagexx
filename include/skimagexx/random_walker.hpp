#pragma once

#include "convert.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace skimagexx {

struct RandomWalkerOptions {
    double beta = 130;
    double tolerance = 1e-3;
    double probability_tolerance = 1e-3;
    std::array<double, 3> spacing{1, 1, 1};
    bool return_probabilities = false;
};

struct RandomWalkerResult {
    std::size_t classes = 0;
    std::vector<std::int64_t> labels;
    std::vector<double> probabilities;
    bool convergence_warning = false;
    bool probability_warning = false;
};

// Aligned C-order 2D/3D grayscale image (float64, float32, or uint8), and
// fixed-width integer markers. Markers must include positive seeds and zeros.
// Positive seed values are mapped to class identifiers 1, ..., classes.
// Full probabilities are class-major, with one plane of shape per class.
// Nullopt denotes unsupported or numerically degenerate inputs. The ordered
// Jacobi-preconditioned CG solver is serial and independent of worker settings.
std::optional<RandomWalkerResult> random_walker(
    const void* data, PixelType data_type, const void* markers, PixelType marker_type,
    const std::vector<std::size_t>& shape, const RandomWalkerOptions& options = {});

}  // namespace skimagexx
