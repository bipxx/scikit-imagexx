#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace skimagexx {

enum class ActiveContourBoundary { periodic, fixed, free, fixed_free, free_fixed };

struct ActiveContourOptions {
    double alpha = 0.01;
    double beta = 0.1;
    double w_line = 0.0;
    double w_edge = 1.0;
    double gamma = 0.01;
    double max_px_move = 1.0;
    int max_num_iter = 2500;
    double convergence = 0.1;
    ActiveContourBoundary boundary = ActiveContourBoundary::periodic;
};

bool active_contour_shape_supported(std::size_t rows, std::size_t columns,
                                    std::size_t channels, std::size_t points);

// Contiguous grayscale or interleaved RGB pixels, and interleaved (row, column)
// snake coordinates. uint8 pixels are scaled by 1/255. The result has two
// coordinates per input point and owns its storage. Unsupported dimensions,
// parameters, samples or singular systems return false and preserve output.
bool active_contour(const double* image, std::size_t rows, std::size_t columns,
                    std::size_t channels, const double* snake, std::size_t points,
                    ActiveContourOptions options, std::vector<double>& output);
bool active_contour(const float* image, std::size_t rows, std::size_t columns,
                    std::size_t channels, const float* snake, std::size_t points,
                    ActiveContourOptions options, std::vector<float>& output);
bool active_contour(const std::uint8_t* image, std::size_t rows, std::size_t columns,
                    std::size_t channels, const double* snake, std::size_t points,
                    ActiveContourOptions options, std::vector<double>& output);

}  // namespace skimagexx
