#pragma once

#include <cstddef>
#include <cstdint>

namespace skimagexx {

inline constexpr std::size_t thin_max_extent = 512;
inline constexpr std::size_t thin_max_iterations = 512;

// Both buffers have height * width accessible bytes in row-major order.
// Byte inputs use zero/nonzero; canonical_boolean rejects bytes above one.
// A zero iteration limit means convergence. Unsupported calls leave output intact.
bool thin_into(const std::uint8_t* input, std::size_t height, std::size_t width,
               bool canonical_boolean, std::size_t max_iterations, std::uint8_t* output);

} // namespace skimagexx
