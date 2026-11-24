#pragma once

#include <cstddef>
#include <cstdint>

namespace skimagexx {

inline constexpr std::size_t corner_orientations_max_image_elements = 4194304;
inline constexpr std::size_t corner_orientations_max_corners = 65536;
inline constexpr std::size_t corner_orientations_max_mask_extent = 63;

constexpr bool corner_orientations_shape_supported(std::size_t rows,
    std::size_t columns, std::size_t num_corners,
    std::size_t mask_extent) noexcept {
    return rows >= 2 && columns >= 2
        && rows <= corner_orientations_max_image_elements / columns
        && num_corners <= corner_orientations_max_corners
        && mask_extent >= 1
        && mask_extent <= corner_orientations_max_mask_extent
        && (mask_extent % 2) == 1;
}

// Native C-order intensity-centroid corner orientation kernels.
// All corner coordinates must lie inside [radius, extent - radius) where
// radius = mask_extent / 2. Unsupported shapes, out-of-bounds corners,
// null required buffers or non-nearest rounding leave orientations untouched
// and return false.
bool corner_orientations_u8_into(const std::uint8_t* image, std::size_t rows,
    std::size_t columns, const std::intptr_t* corners, std::size_t num_corners,
    const std::uint8_t* mask, std::size_t mask_extent, double* orientations);

bool corner_orientations_f32_into(const float* image, std::size_t rows,
    std::size_t columns, const std::intptr_t* corners, std::size_t num_corners,
    const std::uint8_t* mask, std::size_t mask_extent, float* orientations);

bool corner_orientations_f64_into(const double* image, std::size_t rows,
    std::size_t columns, const std::intptr_t* corners, std::size_t num_corners,
    const std::uint8_t* mask, std::size_t mask_extent, double* orientations);

}  // namespace skimagexx
