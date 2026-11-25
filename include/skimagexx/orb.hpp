#pragma once

#include "convert.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace skimagexx {

inline constexpr std::size_t orb_descriptor_size = 256;

struct OrbPoint {
    std::ptrdiff_t row;
    std::ptrdiff_t col;
    double orientation;
    double response;
};

struct OrbOctave {
    bool had_fast_peaks;
    std::vector<OrbPoint> points;
};

// Buffers are aligned, contiguous, and disjoint. Input dimensions must be
// positive. Preparation returns false for nonfinite values or |value| > 4096.
// Float32 input stays float32; other input formats produce float64.
bool orb_prepare(const void* input, void* output, PixelType source,
                 std::size_t rows, std::size_t cols);

// Subsequent operations consume prepared float32 (single=true) or float64
// levels. The downsample target must be positive and no larger than the input.
// Gaussian coefficients are symmetric, normalized and in increasing offset
// order. The array of nine coefficients uses sigma=1 for the Harris response.
void orb_reduce(const void* input, void* output, bool single,
                std::size_t rows, std::size_t cols,
                std::size_t output_rows, std::size_t output_cols,
                const std::vector<double>& weights, bool legacy_zoom=false);
OrbOctave orb_detect_octave(const void* input, bool single,
                            std::size_t rows, std::size_t cols, int fast_n,
                            double threshold, double harris_k,
                            const std::array<double, 9>& harris_weights);

// Coordinates retain their fractional part for border testing, then truncate
// toward zero. Mask is in input order and descriptors are packed by accepted
// point, with one byte per decision. All orientations must be finite.
struct OrbDescriptors {
    std::vector<std::uint8_t> data;
    std::vector<std::uint8_t> mask;
};
OrbDescriptors orb_extract_octave(const void* input, bool single,
                                  std::size_t rows, std::size_t cols,
                                  const double* points, const void* orientations,
                                  std::size_t count);

} // namespace skimagexx
