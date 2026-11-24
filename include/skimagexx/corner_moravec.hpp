#pragma once

#include <cstddef>
#include <cstdint>

namespace skimagexx {

inline constexpr int corner_moravec_max_window = 3;
inline constexpr std::size_t corner_moravec_max_extent = 320;

constexpr bool corner_moravec_shape_supported(
    std::size_t rows, std::size_t columns, int window) noexcept {
    if (window < 1 || window > corner_moravec_max_window) {
        return false;
    }
    const auto minimum = static_cast<std::size_t>(4 * window + 1);
    return rows >= minimum && columns >= minimum
        && rows <= corner_moravec_max_extent && columns <= corner_moravec_max_extent;
}

// Native C-order byte image. Response is separate C-order binary64 storage.
// Caller supplies rows*columns readable bytes and writable doubles.
// Refused shapes, null storage or non-nearest rounding leave response intact.
// The bounded implementation is serial; a two-window-wide border is zero.
bool corner_moravec_u8_into(const std::uint8_t* image, std::size_t rows,
    std::size_t columns, int window, double* response);

}  // namespace skimagexx
