#pragma once

#include <cstddef>
#include <optional>
#include <vector>

namespace skimagexx {

enum class CensureMode { dob, octagon, star };

struct CensureOptions {
    int min_scale = 1;
    int max_scale = 7;
    CensureMode mode = CensureMode::dob;
    double non_max_threshold = 0.15;
    double line_threshold = 10;
};

struct CensureKeypoint {
    std::size_t row;
    std::size_t column;
    int scale;
};

// Input is a contiguous, normalized, already converted grayscale image.
// Results are ordered by row, column and scale. Unsupported magnitudes
// return nullopt. Invalid dimensions and options throw invalid_argument.
std::optional<std::vector<CensureKeypoint>> censure_detect(
    const float* image, std::size_t rows, std::size_t columns, const CensureOptions& options = {});
std::optional<std::vector<CensureKeypoint>> censure_detect(
    const double* image, std::size_t rows, std::size_t columns, const CensureOptions& options = {});

}  // namespace skimagexx
