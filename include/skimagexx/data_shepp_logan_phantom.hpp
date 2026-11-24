#pragma once

#include <cstddef>

namespace skimagexx {

inline constexpr std::size_t data_shepp_logan_phantom_rows = 400;
inline constexpr std::size_t data_shepp_logan_phantom_columns = 400;
inline constexpr std::size_t data_shepp_logan_phantom_elements =
    data_shepp_logan_phantom_rows * data_shepp_logan_phantom_columns;

// Decode the exact 400x400 float64 Shepp-Logan phantom into caller storage.
// Returns false and leaves output untouched if output is null or count < 160000.
bool data_shepp_logan_phantom_f64_into(double* output, std::size_t count) noexcept;

}  // namespace skimagexx
