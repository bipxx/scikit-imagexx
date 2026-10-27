#pragma once
#include <cstddef>
#include <cstdint>

namespace skimagexx::detail {
// Native C uint8 storage after the public source-ordered astype provider.
// The caller supplies 256 disjoint bins; return the source bincount length.
std::size_t perimeter_counts(const std::uint8_t* image, std::size_t height,
    std::size_t width, bool cross, std::ptrdiff_t histogram[256], int workers);
// Called only for a source histogram of length 50. Restores caller FEnv.
double perimeter_weighted(const std::ptrdiff_t histogram[50],
    const double weights[50], int workers);
}

namespace skimagexx::simd {
std::size_t perimeter_border(const std::uint8_t* image, const std::uint8_t* eroded,
    std::uint8_t* border, std::size_t count);
std::size_t perimeter_row(const std::uint8_t* previous, const std::uint8_t* current,
    const std::uint8_t* next, std::size_t width, std::ptrdiff_t histogram[256]);
}
