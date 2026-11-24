#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace skimagexx {

inline constexpr std::size_t regionprops_table_max_extent = 1024;
inline constexpr std::int32_t regionprops_table_max_label = 4096;

struct RegionPropsTableRow {
    std::intptr_t label;
    std::array<std::intptr_t, 4> bbox;
};

// The caller supplies a readable, contiguous row-major buffer of native int32
// words. Extents are positive and bounded. Label 0 is background.
// Results contain ascending positive labels and half-open [r0, c0, r1, c1]
// bounds. An unsupported extent/label returns false and leaves rows untouched.
// An all-background buffer succeeds with no rows. The scan is serial.
bool regionprops_table_bounds(const void* labels, std::size_t height, std::size_t width,
                              std::vector<RegionPropsTableRow>& rows);

}  // namespace skimagexx
