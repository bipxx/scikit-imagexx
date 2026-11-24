#pragma once

#include <cstddef>
#include <cstdint>

namespace skimagexx {

inline constexpr std::size_t data_checkerboard_rows = 200;
inline constexpr std::size_t data_checkerboard_columns = 200;
inline constexpr std::size_t data_checkerboard_elements =
    data_checkerboard_rows * data_checkerboard_columns;

// Decode the exact 200x200 uint8 checkerboard sample into caller storage.
// Returns false and leaves output untouched if output is null or count < 40000.
bool data_checkerboard_u8_into(std::uint8_t* output, std::size_t count) noexcept;

}  // namespace skimagexx
