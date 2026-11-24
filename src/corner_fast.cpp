#include <skimagexx/corner_fast.hpp>

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <vector>

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

constexpr int kRingDy[16] = {
    0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1};
constexpr int kRingDx[16] = {
    3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1, 0, 1, 2, 3};

template <typename T>
bool corner_fast_impl(const T* pixels, std::size_t rows, std::size_t columns,
    int n, T threshold, T* response) {
#ifdef __clang__
#pragma clang fp contract(off)
#endif
    std::fill_n(response, rows * columns, T(0));
    if (rows < 7 || columns < 7) {
        return true;
    }

    const int height = static_cast<int>(rows);
    const int width = static_cast<int>(columns);
    int offsets[16];
    for (int k = 0; k < 16; ++k) {
        offsets[k] = kRingDy[k] * width + kRingDx[k];
    }

    for (int r = 3; r < height - 3; ++r) {
        for (int c = 3; c < width - 3; ++c) {
            const T* center = pixels + r * width + c;
            const T cp = *center;
            const T hi = cp + threshold;
            const T lo = cp - threshold;

            if (n >= 12) {
                const T v0 = center[offsets[0]];
                const T v4 = center[offsets[4]];
                const T v8 = center[offsets[8]];
                const T v12 = center[offsets[12]];
                const int bright = (v0 > hi) + (v4 > hi) + (v8 > hi) + (v12 > hi);
                const int dark = (v0 < lo) + (v4 < lo) + (v8 < lo) + (v12 < lo);
                if (bright < 3 && dark < 3) {
                    continue;
                }
            }

            T ring[16];
            for (int k = 0; k < 16; ++k) {
                ring[k] = center[offsets[k]];
            }

            int bright_run = 0;
            int dark_run = 0;
            bool accepted = false;
            for (int k = 0; k < 32; ++k) {
                const T v = ring[k & 15];
                bright_run = (v > hi) ? (bright_run + 1) : 0;
                dark_run = (v < lo) ? (dark_run + 1) : 0;
                if (bright_run >= n || dark_run >= n) {
                    accepted = true;
                    break;
                }
            }
            if (accepted) {
                T score = T(0);
                for (int k = 0; k < 16; ++k) {
                    score += std::abs(ring[k] - cp);
                }
                response[r * width + c] = score;
            }
        }
    }
    return true;
}

template <typename T>
bool valid_floating_inputs(const T* pixels, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
        const T v = pixels[i];
        if (!std::isfinite(v) || v < T(-256) || v > T(256)) {
            return false;
        }
    }
    return true;
}

bool valid_common(const void* image, const void* response, std::size_t rows,
    std::size_t columns, int n, double threshold) {
    return image && response && corner_fast_shape_supported(rows, columns)
        && n >= corner_fast_min_n && n <= corner_fast_max_n
        && std::isfinite(threshold) && threshold >= 0.0
        && threshold <= corner_fast_max_threshold
        && std::fegetround() == FE_TONEAREST;
}

}  // namespace

bool corner_fast_u8_into(const std::uint8_t* image, std::size_t rows,
    std::size_t columns, int n, double threshold, double* response) {
    if (!valid_common(image, response, rows, columns, n, threshold)) {
        return false;
    }
    constexpr double reciprocal = 1.0 / 255.0;
    const std::size_t count = rows * columns;
    std::vector<double> pixels(count);
    std::transform(image, image + count, pixels.begin(),
        [reciprocal](std::uint8_t pixel) { return double(pixel) * reciprocal; });
    return corner_fast_impl(pixels.data(), rows, columns, n, threshold, response);
}

bool corner_fast_f32_into(const float* image, std::size_t rows,
    std::size_t columns, int n, float threshold, float* response) {
    if (!valid_common(image, response, rows, columns, n, static_cast<double>(threshold))
        || !valid_floating_inputs(image, rows * columns)) {
        return false;
    }
    return corner_fast_impl(image, rows, columns, n, threshold, response);
}

bool corner_fast_f64_into(const double* image, std::size_t rows,
    std::size_t columns, int n, double threshold, double* response) {
    if (!valid_common(image, response, rows, columns, n, threshold)
        || !valid_floating_inputs(image, rows * columns)) {
        return false;
    }
    return corner_fast_impl(image, rows, columns, n, threshold, response);
}

}  // namespace skimagexx

#if defined(_MSC_VER)
#pragma float_control(pop)
#endif
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC pop_options
#endif
