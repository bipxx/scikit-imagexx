#include <skimagexx/corner_orientations.hpp>

#include <cfenv>
#include <cmath>

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC push_options
#pragma GCC optimize ("fp-contract=off")
#endif
#if defined(_MSC_VER)
#pragma float_control(precise, on, push)
#pragma fp_contract(off)
#endif

namespace skimagexx {
namespace {

inline double read_pixel(std::uint8_t value) noexcept {
    constexpr double reciprocal = 1.0 / 255.0;
    return static_cast<double>(value) * reciprocal;
}

inline float read_pixel(float value) noexcept {
    return value;
}

inline double read_pixel(double value) noexcept {
    return value;
}

inline float finish_atan2(float m01, float m10) noexcept {
    return static_cast<float>(std::atan2(static_cast<double>(m01), static_cast<double>(m10)));
}

inline double finish_atan2(double m01, double m10) noexcept {
    return std::atan2(m01, m10);
}

template <typename PixelT, typename OutT>
bool corner_orientations_impl(const PixelT* image, std::size_t rows,
    std::size_t columns, const std::intptr_t* corners, std::size_t num_corners,
    const std::uint8_t* mask, std::size_t mask_extent, OutT* orientations) {
#ifdef __clang__
#pragma clang fp contract(off)
#endif
    if (!image || !mask
        || !corner_orientations_shape_supported(rows, columns, num_corners, mask_extent)
        || std::fegetround() != FE_TONEAREST) {
        return false;
    }
    if (num_corners == 0) {
        return true;
    }
    if (!corners || !orientations) {
        return false;
    }

    const auto radius = static_cast<std::intptr_t>(mask_extent / 2);
    const auto height = static_cast<std::intptr_t>(rows);
    const auto width = static_cast<std::intptr_t>(columns);
    if (height - radius <= radius || width - radius <= radius) {
        return false;
    }
    for (std::size_t i = 0; i < num_corners; ++i) {
        const std::intptr_t r = corners[2 * i];
        const std::intptr_t c = corners[2 * i + 1];
        if (r < radius || r >= height - radius || c < radius || c >= width - radius) {
            return false;
        }
    }

    const int extent = static_cast<int>(mask_extent);
    const int rad = static_cast<int>(radius);
    for (std::size_t i = 0; i < num_corners; ++i) {
        const std::intptr_t r = corners[2 * i];
        const std::intptr_t c = corners[2 * i + 1];
        OutT m01 = OutT(0);
        OutT m10 = OutT(0);
        for (int y = 0; y < extent; ++y) {
            const int dy = y - rad;
            const PixelT* row_ptr = image + (r + dy) * width + (c - rad);
            const std::uint8_t* mask_row = mask + y * extent;
            OutT row_sum = OutT(0);
            for (int x = 0; x < extent; ++x) {
                if (mask_row[x] != 0) {
                    const OutT pixel = read_pixel(row_ptr[x]);
                    row_sum += pixel;
                    m10 += static_cast<OutT>(x - rad) * pixel;
                }
            }
            m01 += static_cast<OutT>(dy) * row_sum;
        }
        orientations[i] = finish_atan2(m01, m10);
    }
    return true;
}

}  // namespace

bool corner_orientations_u8_into(const std::uint8_t* image, std::size_t rows,
    std::size_t columns, const std::intptr_t* corners, std::size_t num_corners,
    const std::uint8_t* mask, std::size_t mask_extent, double* orientations) {
    return corner_orientations_impl(image, rows, columns, corners, num_corners,
        mask, mask_extent, orientations);
}

bool corner_orientations_f32_into(const float* image, std::size_t rows,
    std::size_t columns, const std::intptr_t* corners, std::size_t num_corners,
    const std::uint8_t* mask, std::size_t mask_extent, float* orientations) {
    return corner_orientations_impl(image, rows, columns, corners, num_corners,
        mask, mask_extent, orientations);
}

bool corner_orientations_f64_into(const double* image, std::size_t rows,
    std::size_t columns, const std::intptr_t* corners, std::size_t num_corners,
    const std::uint8_t* mask, std::size_t mask_extent, double* orientations) {
    return corner_orientations_impl(image, rows, columns, corners, num_corners,
        mask, mask_extent, orientations);
}

}  // namespace skimagexx

#if defined(_MSC_VER)
#pragma float_control(pop)
#endif
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC pop_options
#endif
