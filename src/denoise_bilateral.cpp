#include "skimagexx/denoise_bilateral.hpp"
#include "skimagexx/image.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

#if defined(SKIMAGEXX_AVX2) && (defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#define SKIMAGEXX_BILATERAL_AVX2
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable: 4752)
#define BILATERAL_AVX_ENTRY __declspec(noinline)
#else
#define BILATERAL_AVX_ENTRY __attribute__((target("avx2"), noinline))
#endif
#endif

namespace skimagexx {
namespace {

template<class T, std::size_t Channels, bool Interior>
void evaluate(const T* image, std::size_t height, std::size_t width,
              std::size_t row, std::size_t col, std::size_t window, T cval,
              const T* color, const T* spatial, T scale, T last_bin, T* output) {
    const auto reach = static_cast<std::ptrdiff_t>(window / 2);
    const auto* center = image + (row * width + col) * Channels;
    std::array<T, Channels> sums{};
    T total = 0;
    const auto color_weight = [&](const T* neighbor) {
        T squared = 0;
        for (std::size_t channel = 0; channel < Channels; ++channel) {
            const T difference = center[channel] - (neighbor ? neighbor[channel] : cval);
            squared += difference * difference;
        }
        const T distance = static_cast<T>(std::sqrt(static_cast<double>(squared)));
        // Clamp before truncation, including for a NaN outside the prepared contract.
        const auto bin = static_cast<std::size_t>(std::min(last_bin, distance * scale));
        return color[bin];
    };
    T border_color = 0;
    if constexpr (!Interior) border_color = color_weight(nullptr);
    for (std::size_t kr = 0; kr < window; ++kr) {
        const auto r = static_cast<std::ptrdiff_t>(row + kr) - reach;
        const T* line = nullptr;
        if (Interior || (r >= 0 && static_cast<std::size_t>(r) < height))
            line = image + static_cast<std::size_t>(r) * width * Channels;
        for (std::size_t kc = 0; kc < window; ++kc) {
            const auto c = static_cast<std::ptrdiff_t>(col + kc) - reach;
            const bool inside = Interior || (line && c >= 0 && static_cast<std::size_t>(c) < width);
            const auto* neighbor = inside ? line + static_cast<std::size_t>(c) * Channels : nullptr;
            const T weight = *spatial++ * (inside ? color_weight(neighbor) : border_color);
            for (std::size_t channel = 0; channel < Channels; ++channel)
                sums[channel] += (inside ? neighbor[channel] : cval) * weight;
            total += weight;
        }
    }
    for (std::size_t channel = 0; channel < Channels; ++channel)
        output[channel] = sums[channel] / total;
}

template<class T>
void evaluate_gray_block(const T* center, std::size_t width, std::size_t window,
                         const T* color, const T* spatial, T scale, T last_bin,
                         T* output) {
    constexpr std::size_t block = 4;
    std::array<T, block> sums{}, totals{};
    const auto radius = window / 2;
    const auto* first = center - radius * width - radius;
    // Each tap serves four adjacent windows, in the original tap order
    // for every output pixel.
    for (std::size_t kr = 0; kr < window; ++kr) {
        const auto* line = first + kr * width;
        for (std::size_t kc = 0; kc < window; ++kc) {
            const T spatial_weight = *spatial++;
            for (std::size_t lane = 0; lane < block; ++lane) {
                const T neighbor = line[kc + lane];
                const T difference = center[lane] - neighbor;
                const T squared = T(0) + difference * difference;
                const T distance = static_cast<T>(std::sqrt(static_cast<double>(squared)));
                const auto bin = static_cast<std::size_t>(std::min(last_bin, distance * scale));
                const T weight = spatial_weight * color[bin];
                sums[lane] += neighbor * weight;
                totals[lane] += weight;
            }
        }
    }
    for (std::size_t lane = 0; lane < block; ++lane)
        output[lane] = sums[lane] / totals[lane];
}

#ifdef SKIMAGEXX_BILATERAL_AVX2
BILATERAL_AVX_ENTRY void evaluate_gray_avx2(
    const float* center, std::size_t width, std::size_t window, const float* color,
    const float* spatial, float scale, float last_bin, float* output) {
    const auto radius = window / 2;
    const auto* first = center - radius * width - radius;
    const auto centers = _mm_loadu_ps(center);
    const auto scales = _mm_set1_ps(scale);
    const auto last = _mm_set1_ps(last_bin);
    auto sum = _mm_setzero_ps();
    auto total = _mm_setzero_ps();
    for (std::size_t kr = 0; kr < window; ++kr) {
        const auto* line = first + kr * width;
        for (std::size_t kc = 0; kc < window; ++kc) {
            const auto neighbor = _mm_loadu_ps(line + kc);
            const auto delta = _mm_sub_ps(centers, neighbor);
            const auto squared = _mm_add_ps(_mm_setzero_ps(), _mm_mul_ps(delta, delta));
            // As in the scalar path, round the squared difference to float,
            // take the root in double, then narrow before scaling the bin.
            const auto distance = _mm256_cvtpd_ps(_mm256_sqrt_pd(_mm256_cvtps_pd(squared)));
            const auto bins = _mm_cvttps_epi32(_mm_min_ps(_mm_mul_ps(distance, scales), last));
            const auto weight = _mm_mul_ps(_mm_set1_ps(*spatial++), _mm_i32gather_ps(color, bins, 4));
            sum = _mm_add_ps(sum, _mm_mul_ps(neighbor, weight));
            total = _mm_add_ps(total, weight);
        }
    }
    _mm_storeu_ps(output, _mm_div_ps(sum, total));
}

BILATERAL_AVX_ENTRY void evaluate_gray_avx2(
    const double* center, std::size_t width, std::size_t window, const double* color,
    const double* spatial, double scale, double last_bin, double* output) {
    const auto radius = window / 2;
    const auto* first = center - radius * width - radius;
    const auto centers = _mm256_loadu_pd(center);
    const auto scales = _mm256_set1_pd(scale);
    const auto last = _mm256_set1_pd(last_bin);
    auto sum = _mm256_setzero_pd();
    auto total = _mm256_setzero_pd();
    for (std::size_t kr = 0; kr < window; ++kr) {
        const auto* line = first + kr * width;
        for (std::size_t kc = 0; kc < window; ++kc) {
            const auto neighbor = _mm256_loadu_pd(line + kc);
            const auto delta = _mm256_sub_pd(centers, neighbor);
            const auto squared = _mm256_add_pd(_mm256_setzero_pd(), _mm256_mul_pd(delta, delta));
            const auto distance = _mm256_sqrt_pd(squared);
            const auto bins = _mm256_cvttpd_epi32(_mm256_min_pd(_mm256_mul_pd(distance, scales), last));
            const auto weight = _mm256_mul_pd(_mm256_set1_pd(*spatial++), _mm256_i32gather_pd(color, bins, 8));
            sum = _mm256_add_pd(sum, _mm256_mul_pd(neighbor, weight));
            total = _mm256_add_pd(total, weight);
        }
    }
    _mm256_storeu_pd(output, _mm256_div_pd(sum, total));
}
#undef BILATERAL_AVX_ENTRY
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
#endif

template<class T, std::size_t Channels>
void stencil(const T* image, std::size_t height, std::size_t width,
             std::size_t window, T cval, const T* color, std::size_t bins,
             double maximum, const T* spatial, T* result) {
    const auto radius = window / 2;
    const T scale = static_cast<T>((bins / Channels) / maximum);
    const T last_bin = static_cast<T>(bins - 1);
#ifdef SKIMAGEXX_BILATERAL_AVX2
    const bool vectorized = Channels == 1 && has_avx2();
#endif
    for (std::size_t row = 0; row < height; ++row) {
        const auto first = row >= radius && row + radius < height && width > 2 * radius
            ? radius : width;
        const auto last = first == width ? width : width - radius;
        const auto border = [&](std::size_t col) {
            evaluate<T, Channels, false>(image, height, width, row, col, window, cval,
                color, spatial, scale, last_bin, result + (row * width + col) * Channels);
        };
        std::size_t col = 0;
        for (; col < first; ++col) border(col);
        if constexpr (Channels == 1) {
            for (; col + 4 <= last; col += 4) {
                const auto index = row * width + col;
#ifdef SKIMAGEXX_BILATERAL_AVX2
                if (vectorized) {
                    evaluate_gray_avx2(image + index, width, window, color, spatial, scale,
                                       last_bin, result + index);
                    continue;
                }
#endif
                evaluate_gray_block(image + index, width, window, color, spatial, scale,
                                    last_bin, result + index);
            }
        }
        for (; col < last; ++col)
            evaluate<T, Channels, true>(image, height, width, row, col, window, cval,
                color, spatial, scale, last_bin, result + (row * width + col) * Channels);
        for (; col < width; ++col) border(col);
    }
}

}  // namespace

template<class T>
void denoise_bilateral_constant_into(
    const T* image, std::size_t image_count, std::size_t height, std::size_t width,
    std::size_t channels, double maximum, std::size_t window, double cval,
    const T* color_lut, std::size_t bins, const T* spatial_lut, std::size_t spatial_count,
    T* result, std::size_t result_count) {
    if (!image || !result || !color_lut || !spatial_lut || height < 2 || height > 512 ||
        width < 2 || width > 512 || (channels != 1 && channels != 3) ||
        !(maximum >= 1.0 / 256 && maximum <= 1) || !(cval >= 0 && cval <= 1) ||
        window < 5 || window > 25 || window % 2 == 0 || bins < 4 || bins > 65536)
        throw std::invalid_argument("invalid prepared bilateral parameters");
    const auto count = height * width * channels;
    if (image_count < count || result_count < count || spatial_count < (window + 1) * (window + 1))
        throw std::invalid_argument("incomplete prepared bilateral storage");
    if (channels == 1)
        stencil<T, 1>(image, height, width, window, static_cast<T>(cval),
                       color_lut, bins, maximum, spatial_lut, result);
    else
        stencil<T, 3>(image, height, width, window, static_cast<T>(cval),
                       color_lut, bins, maximum, spatial_lut, result);
}

template void denoise_bilateral_constant_into<float>(
    const float*, std::size_t, std::size_t, std::size_t, std::size_t, double, std::size_t, double,
    const float*, std::size_t, const float*, std::size_t, float*, std::size_t);
template void denoise_bilateral_constant_into<double>(
    const double*, std::size_t, std::size_t, std::size_t, std::size_t, double, std::size_t, double,
    const double*, std::size_t, const double*, std::size_t, double*, std::size_t);

}  // namespace skimagexx
