#include "skimagexx/denoise_bilateral.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

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
    for (std::size_t kr = 0; kr < window; ++kr) {
        const auto r = static_cast<std::ptrdiff_t>(row + kr) - reach;
        for (std::size_t kc = 0; kc < window; ++kc) {
            const auto c = static_cast<std::ptrdiff_t>(col + kc) - reach;
            const bool inside = Interior || (r >= 0 && c >= 0 &&
                static_cast<std::size_t>(r) < height && static_cast<std::size_t>(c) < width);
            const auto* neighbor = inside ? image +
                (static_cast<std::size_t>(r) * width + static_cast<std::size_t>(c)) * Channels : nullptr;
            std::array<T, Channels> values;
            T squared = 0;
            for (std::size_t channel = 0; channel < Channels; ++channel) {
                const auto value = inside ? neighbor[channel] : cval;
                values[channel] = value;
                const T difference = center[channel] - value;
                squared += difference * difference;
            }
            const T distance = static_cast<T>(std::sqrt(static_cast<double>(squared)));
            // Finite nonnegative distances give the same truncation after an
            // upper clamp. Clamping before conversion also bounds a NaN index
            // if a caller violates the prepared finite-input contract.
            const auto bin = static_cast<std::size_t>(std::min(last_bin, distance * scale));
            const T weight = spatial[kr * window + kc] * color[bin];
            for (std::size_t channel = 0; channel < Channels; ++channel)
                sums[channel] += values[channel] * weight;
            total += weight;
        }
    }
    for (std::size_t channel = 0; channel < Channels; ++channel)
        output[channel] = sums[channel] / total;
}

template<class T, std::size_t Channels>
void stencil(const T* image, std::size_t height, std::size_t width,
             std::size_t window, T cval, const T* color, std::size_t bins,
             double maximum, const T* spatial, T* result) {
    const auto radius = window / 2;
    const T scale = static_cast<T>((bins / Channels) / maximum);
    const T last_bin = static_cast<T>(bins - 1);
    for (std::size_t row = 0; row < height; ++row) {
        for (std::size_t col = 0; col < width; ++col) {
            auto* output = result + (row * width + col) * Channels;
            if (row >= radius && col >= radius && row + radius < height && col + radius < width)
                evaluate<T, Channels, true>(image, height, width, row, col, window, cval,
                                             color, spatial, scale, last_bin, output);
            else
                evaluate<T, Channels, false>(image, height, width, row, col, window, cval,
                                              color, spatial, scale, last_bin, output);
        }
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
