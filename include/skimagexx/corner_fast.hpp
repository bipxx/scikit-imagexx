#pragma once

#include <cstddef>
#include <cstdint>

namespace skimagexx {

inline constexpr int corner_fast_min_n = 1;
inline constexpr int corner_fast_max_n = 16;
inline constexpr double corner_fast_max_threshold = 256.0;
inline constexpr std::size_t corner_fast_max_elements = 4194304;

constexpr bool corner_fast_shape_supported(
    std::size_t rows, std::size_t columns) noexcept {
    return rows >= 2 && columns >= 2
        && rows <= corner_fast_max_elements / columns;
}

// Native C-order FAST corner response kernels.
// Input and output storage must be disjoint and hold rows * columns elements.
// Unsupported shapes, invalid parameters, out-of-range floating values or
// non-nearest rounding leave response untouched and return false.
bool corner_fast_u8_into(const std::uint8_t* image, std::size_t rows,
    std::size_t columns, int n, double threshold, double* response);

bool corner_fast_f32_into(const float* image, std::size_t rows,
    std::size_t columns, int n, float threshold, float* response);

bool corner_fast_f64_into(const double* image, std::size_t rows,
    std::size_t columns, int n, double threshold, double* response);

}  // namespace skimagexx
