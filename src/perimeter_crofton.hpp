#pragma once
#include "skimagexx/convert.hpp"
#include <cstddef>
#include <cstdint>

namespace skimagexx::detail {
// Read-only, aligned native C storage. The binder converts half to float32;
// all other supported input widths are preserved. Empty images allow null data.
struct CroftonInput {
    const void* data;
    PixelType type;
    std::size_t height, width;
};
// Caller owns 16 intp counters, disjoint from input. Counts include the entire
// one-pixel zero-padded convolution output, including its zero-pattern bin.
void crofton_counts(const CroftonInput&, std::ptrdiff_t histogram[16], int workers);
// Counts must finish before Python evaluates directions. The second stage
// consumes the exact source coefficient list and preserves caller FEnv.
double crofton_perimeter(const std::ptrdiff_t histogram[16], const double coefficients[16], int workers);
}

namespace skimagexx::simd {
std::size_t crofton_threshold(const void*, PixelType, std::uint8_t*, std::size_t count);
std::size_t crofton_row(const std::uint8_t* current, const std::uint8_t* previous,
                       std::size_t width, std::ptrdiff_t histogram[16]);
}
