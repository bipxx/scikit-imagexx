#pragma once
#include <cstddef>
#include <cstdint>

namespace skimagexx::detail {
// Native C binary storage, already padded with one zero pixel on every side.
// Binary conversion and padding precede Python connectivity callbacks.
struct EulerInput {
    const std::uint8_t* data;
    std::size_t depth, height, width; // depth is 1 for a 2D image.
};
// Caller owns 16 (2D) or 256 (3D) counters; all padded pixels are counted.
void euler_counts(const EulerInput&, bool volume, std::ptrdiff_t* histogram, int workers);
// Exact source LUT values. False leaves result untouched when Snum > 2**53.
// For 3D, integer division by eight matches the source's exact eighth-grid dot.
bool euler_weighted(const std::ptrdiff_t* histogram, const std::int8_t* coefficients,
                    bool volume, std::int64_t& result, int workers);
}

namespace skimagexx::simd {
std::size_t euler_row(const std::uint8_t* current, const std::uint8_t* above,
                      const std::uint8_t* previous, const std::uint8_t* previous_above,
                      std::size_t width, std::ptrdiff_t histogram[256]);
}
