#include "skimagexx/quickshift.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace skimagexx {
namespace {

using Index = std::ptrdiff_t;

template<class T, Index fixed_channels>
T distance(const T* a, const T* b, T row_squared, T col_squared, Index channels) {
    T result = 0;
    const auto count = fixed_channels ? fixed_channels : channels;
    for (Index channel = 0; channel < count; ++channel) {
        const T difference = a[channel] - b[channel];
        result += difference * difference;
    }
    result += row_squared;
    result += col_squared;
    return result;
}

template<class T, Index fixed_channels>
void segment(const T* image, const T* noise, Index* labels,
             Index height, Index width, Index channels, T kernel_size, T max_distance) {
    const auto count = height * width;
    const auto radius = static_cast<Index>(std::min(
        std::ceil(static_cast<double>(T{3} * kernel_size)),
        static_cast<double>(std::max(height, width))));
    const T inverse = static_cast<T>(-0.5 / (kernel_size * kernel_size));
    std::vector<T> densities(noise, noise + count);
    std::vector<T> squared_offsets(2 * radius + 1);
    T* const squared = squared_offsets.data() + radius;
    for (Index offset = -radius; offset <= radius; ++offset) {
        const T difference = static_cast<T>(offset);
        squared[offset] = difference * difference;
    }

    // Each pair contributes to both densities. Traversing the upper triangle
    // in row-major order preserves the summation order at every pixel.
    for (Index row = 0; row < height; ++row) {
        const auto bottom = std::min(height, row + radius + 1);
        for (Index col = 0; col < width; ++col) {
            const auto pixel = row * width + col;
            auto& density = densities[pixel];
            density += T{1};
            const auto* values = image + pixel * channels;
            const auto left = std::max(Index{0}, col - radius);
            const auto right = std::min(width, col + radius + 1);
            for (auto other_row = row; other_row < bottom; ++other_row) {
                const auto start = other_row == row ? col + 1 : left;
                const auto row_squared = squared[other_row - row];
                auto other = other_row * width + start;
                for (auto other_col = start; other_col < right; ++other_col, ++other) {
                    const auto squared_distance = distance<T, fixed_channels>(
                        values, image + other * channels, row_squared,
                        squared[other_col - col], channels);
                    const auto weight = std::exp(static_cast<double>(squared_distance * inverse));
                    density = static_cast<T>(density + weight);
                    densities[other] = static_cast<T>(densities[other] + weight);
                }
            }
        }
    }

    // Points beyond this spatial radius cannot survive the distance cut, even
    // after the output distance is rounded to T.
    const auto search_radius = static_cast<Index>(std::min(
        static_cast<double>(radius), std::floor(static_cast<double>(max_distance)) + 1));
    for (Index row = 0; row < height; ++row) {
        const auto top = std::max(Index{0}, row - search_radius);
        const auto bottom = std::min(height, row + search_radius + 1);
        for (Index col = 0; col < width; ++col) {
            const auto pixel = row * width + col;
            const auto left = std::max(Index{0}, col - search_radius);
            const auto right = std::min(width, col + search_radius + 1);
            const auto* values = image + pixel * channels;
            const auto density = densities[pixel];
            T closest = std::numeric_limits<T>::infinity();
            auto parent = pixel;
            for (auto other_row = top; other_row < bottom; ++other_row) {
                const auto row_squared = squared[row - other_row];
                auto other = other_row * width + left;
                for (auto other_col = left; other_col < right; ++other_col, ++other) {
                    if (densities[other] <= density) continue;
                    const auto squared_distance = distance<T, fixed_channels>(
                        values, image + other * channels, row_squared,
                        squared[col - other_col], channels);
                    if (squared_distance < closest) {
                        closest = squared_distance;
                        parent = other;
                    }
                }
            }
            labels[pixel] = static_cast<T>(std::sqrt(static_cast<double>(closest))) <= max_distance
                            ? parent : pixel;
        }
    }

    for (Index pixel = 0; pixel < count; ++pixel) {
        auto root = pixel;
        while (labels[root] != root) root = labels[root];
        auto node = pixel;
        while (labels[node] != root) {
            const auto next_node = labels[node];
            labels[node] = root;
            node = next_node;
        }
    }
    // Encode labels as negative values until all compressed root indices have
    // been resolved. Numbering the roots in pixel order preserves tie order.
    Index next = 0;
    for (Index pixel = 0; pixel < count; ++pixel) {
        if (labels[pixel] == pixel)
            labels[pixel] = -1 - next++;
    }
    for (Index pixel = 0; pixel < count; ++pixel) {
        if (labels[pixel] >= 0)
            labels[pixel] = labels[labels[pixel]];
    }
    for (Index pixel = 0; pixel < count; ++pixel)
        labels[pixel] = -1 - labels[pixel];
}

} // namespace

template<class T>
void quickshift_into(const T* image, const T* noise, Index* labels,
                     Index height, Index width, Index channels, T kernel_size, T max_distance) {
    switch (channels) {
    case 1:
        segment<T, 1>(image, noise, labels, height, width, channels, kernel_size, max_distance);
        break;
    case 3:
        segment<T, 3>(image, noise, labels, height, width, channels, kernel_size, max_distance);
        break;
    default:
        segment<T, 0>(image, noise, labels, height, width, channels, kernel_size, max_distance);
    }
}

template void quickshift_into(const float*, const float*, Index*, Index, Index, Index, float, float);
template void quickshift_into(const double*, const double*, Index*, Index, Index, Index, double, double);

} // namespace skimagexx
