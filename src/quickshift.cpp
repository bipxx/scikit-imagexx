#include "skimagexx/quickshift.hpp"
#include "skimagexx/image.hpp"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <limits>
#include <numeric>
#include <type_traits>
#include <vector>

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize ("fp-contract=off")
#elif defined(_MSC_VER)
#pragma float_control(precise, on, push)
#pragma fp_contract(off)
#endif

namespace skimagexx {
#ifdef SKIMAGEXX_AVX2
namespace simd {
template<class T, std::ptrdiff_t fixed_channels>
extern std::ptrdiff_t quickshift_density_row(
    const T*, const T*, T&, T*, T, const T*, std::ptrdiff_t, std::ptrdiff_t, std::ptrdiff_t, T,
    const double*);

template<class T, std::ptrdiff_t fixed_channels>
extern std::ptrdiff_t quickshift_parent_row(
    const T*, const T*, const T*, T, T, const T*, std::ptrdiff_t, std::ptrdiff_t,
    std::ptrdiff_t, std::ptrdiff_t, T&, std::ptrdiff_t&);
}
#endif

namespace {

using Index = std::ptrdiff_t;

template<class T, Index fixed_channels>
T distance(const T* a, const T* b, T row_squared, T col_squared, Index pixels, Index channels) {
    T result = 0;
    const auto count = fixed_channels ? fixed_channels : channels;
    for (Index channel = 0; channel < count; ++channel) {
        const T difference = a[channel * pixels] - b[channel * pixels];
        result += difference * difference;
    }
    result += row_squared;
    result += col_squared;
    return result;
}

template<class T>
void add_weight(T& density, T& other_density, double weight) {
    density = static_cast<T>(density + weight);
    other_density = static_cast<T>(other_density + weight);
}

#ifdef SKIMAGEXX_AVX2
template<Index fixed_channels>
void resolve_density_ties(double* densities, const double* noise, const double* planes,
                          Index* order, const double* squared, Index height, Index width,
                          Index channels, Index radius, double inverse) {
    const Index count = height * width;
    const double side = 2 * static_cast<double>(radius) + 1;
    const double terms = std::min(static_cast<double>(height), side) *
                         std::min(static_cast<double>(width), side);
    // With |noise| <= 1 and weights <= 1, a partial sum has magnitude at
    // most terms + 1 up to rounding. Allow for both paths' summation
    // error, including the vector reduction, and the vector exponential's
    // absolute weight error of 2e-14.
    const double gap = terms * (4e-14 + 8 * std::numeric_limits<double>::epsilon() * (terms + 1));
    std::iota(order, order + count, Index{0});
    std::sort(order, order + count, [densities](Index a, Index b) {
        return densities[a] < densities[b];
    });
    for (Index first = 0; first < count;) {
        Index end = first + 1;
        while (end < count && densities[order[end]] - densities[order[end - 1]] <= gap) ++end;
        // Distinct groups are farther apart than the combined error bound.
        // Recompute every member of a group that could contain a tie.
        if (end - first > 1) {
            for (Index i = first; i < end; ++i) {
                const auto pixel = order[i];
                const auto row = pixel / width;
                const auto col = pixel % width;
                const auto top = std::max(Index{0}, row - radius);
                const auto bottom = std::min(height, row + radius + 1);
                const auto left = std::max(Index{0}, col - radius);
                const auto right = std::min(width, col + radius + 1);
                double density = noise[pixel];
                for (auto y = top; y < bottom; ++y) {
                    for (auto x = left; x < right; ++x) {
                        const auto other = y * width + x;
                        const auto d = distance<double, fixed_channels>(
                            planes + pixel, planes + other, squared[y - row], squared[x - col],
                            count, channels);
                        density += std::exp(d * inverse);
                    }
                }
                densities[pixel] = density;
            }
        }
        first = end;
    }
}
#endif

template<class T, Index fixed_channels, bool vectorized>
void segment(const T* image, const T* noise, Index* labels,
             Index height, Index width, Index channels, T kernel_size, T max_distance) {
    const auto count = height * width;
    const auto radius = static_cast<Index>(std::min(
        std::ceil(static_cast<double>(T{3} * kernel_size)),
        static_cast<double>(std::max(height, width))));
    const T inverse = static_cast<T>(-0.5 / (kernel_size * kernel_size));
    std::vector<T> channel_planes(count * channels);
    for (Index channel = 0; channel < channels; ++channel) {
        for (Index pixel = 0; pixel < count; ++pixel)
            channel_planes[channel * count + pixel] = image[pixel * channels + channel];
    }
    const auto* planes = channel_planes.data();
    std::vector<T> densities(noise, noise + count);
    std::vector<T> squared_offsets(2 * radius + 1);
    T* const squared = squared_offsets.data() + radius;
    for (Index offset = -radius; offset <= radius; ++offset) {
        const T difference = static_cast<T>(offset);
        squared[offset] = difference * difference;
    }

#ifdef SKIMAGEXX_AVX2
    std::vector<double> spatial_weights;
    const Index spatial_stride = 2 * radius + 1;
    if constexpr (vectorized && fixed_channels == 1 && std::is_same_v<T, double>) {
        // On normalized grayscale input the color exponent is in [-1/16, 0].
        // Keep the spatial table small and use the general kernel otherwise.
        if (radius <= 32 && inverse >= -0.0625 &&
            std::all_of(image, image + count, [](double value) { return value >= 0 && value <= 1; })) {
            spatial_weights.resize((radius + 1) * spatial_stride);
            for (Index dy = 0; dy <= radius; ++dy) {
                for (Index dx = -radius; dx <= radius; ++dx) {
                    spatial_weights[dy * spatial_stride + radius + dx] =
                        std::exp((squared[dy] + squared[dx]) * inverse);
                }
            }
        }
    }
#endif

    // Each pair contributes to both densities. Traversing the upper triangle
    // in row-major order preserves the order of contributions. Double AVX2
    // groups a row's terms and resolves uncertain density comparisons below.
    for (Index row = 0; row < height; ++row) {
        const auto bottom = std::min(height, row + radius + 1);
        for (Index col = 0; col < width; ++col) {
            const auto pixel = row * width + col;
            auto& density = densities[pixel];
            density += T{1};
            const auto* values = planes + pixel;
            const auto left = std::max(Index{0}, col - radius);
            const auto right = std::min(width, col + radius + 1);
            for (auto other_row = row; other_row < bottom; ++other_row) {
                const auto start = other_row == row ? col + 1 : left;
                const auto row_squared = squared[other_row - row];
                const auto other = other_row * width + start;
                Index i = 0;
#ifdef SKIMAGEXX_AVX2
                if constexpr (vectorized) {
                    const auto* weights = spatial_weights.empty() ? nullptr
                        : spatial_weights.data() + (other_row - row) * spatial_stride + radius + (start - col);
                    i = simd::quickshift_density_row<T, fixed_channels>(values, planes + other, density,
                        densities.data() + other, row_squared, squared + (start - col),
                        right - start, count, channels, inverse, weights);
                }
#endif
                for (; i < right - start; ++i) {
                    const auto squared_distance = distance<T, fixed_channels>(
                        values, planes + other + i, row_squared,
                        squared[start + i - col], count, channels);
                    const auto weight = std::exp(static_cast<double>(squared_distance * inverse));
                    add_weight(density, densities[other + i], weight);
                }
            }
        }
    }

#ifdef SKIMAGEXX_AVX2
    if constexpr (vectorized && std::is_same_v<T, double>) {
        resolve_density_ties<fixed_channels>(
            densities.data(), noise, planes, labels, squared, height, width, channels, radius, inverse);
    }
#endif

    const auto cutoff = static_cast<double>(max_distance);
    const auto square_radius = static_cast<Index>(
        std::min(static_cast<double>(radius), std::floor(cutoff) + 1));
    // Widen the disk to account for rounding in the two spatial squares,
    // their sum, and the final square root cast to T. Color terms can only
    // increase the squared distance.
    const double widened_cutoff = std::nextafter(
        cutoff * (1 + 4 * static_cast<double>(std::numeric_limits<T>::epsilon())),
        std::numeric_limits<double>::infinity());
    const double cutoff_radius = std::min(2 * static_cast<double>(square_radius), widened_cutoff);
    const auto search_radius = static_cast<Index>(
        std::min(static_cast<double>(square_radius), std::floor(cutoff_radius)));
    std::vector<Index> col_radii(search_radius + 1);
    for (Index dy = 0; dy <= search_radius; ++dy) {
        const auto y = static_cast<double>(dy);
        col_radii[dy] = std::min(search_radius, static_cast<Index>(
            std::sqrt(cutoff_radius * cutoff_radius - y * y)));
    }
    for (Index row = 0; row < height; ++row) {
        const auto top = std::max(Index{0}, row - search_radius);
        const auto bottom = std::min(height, row + search_radius + 1);
        for (Index col = 0; col < width; ++col) {
            const auto pixel = row * width + col;
            const auto* values = planes + pixel;
            const auto density = densities[pixel];
            T closest = std::numeric_limits<T>::infinity();
            auto parent = pixel;
            if constexpr (fixed_channels == 1) {
                // All candidates outside the four axial neighbors have
                // spatial distance squared >= 2. Visit in reference tie order.
                const auto try_axial = [&](Index other) {
                    if (densities[other] <= density) return;
                    const T delta = *values - planes[other];
                    const T d = delta * delta + T{1};
                    if (d < closest) {
                        closest = d;
                        parent = other;
                    }
                };
                if (row > 0) try_axial(pixel - width);
                if (col > 0) try_axial(pixel - 1);
                if (col + 1 < width) try_axial(pixel + 1);
                if (row + 1 < height) try_axial(pixel + width);
                if (closest < T{2}) {
                    labels[pixel] = static_cast<T>(std::sqrt(static_cast<double>(closest))) <= max_distance
                                    ? parent : pixel;
                    continue;
                }
                closest = std::numeric_limits<T>::infinity();
                parent = pixel;
            }
            for (auto other_row = top; other_row < bottom; ++other_row) {
                const auto dy = std::abs(other_row - row);
                const auto row_squared = squared[dy];
                if (row_squared >= closest) continue;
                const auto col_radius = col_radii[dy];
                const auto left = std::max(Index{0}, col - col_radius);
                const auto right = std::min(width, col + col_radius + 1);
                const auto other = other_row * width + left;
                Index i = 0;
#ifdef SKIMAGEXX_AVX2
                if constexpr (vectorized) {
                    i = simd::quickshift_parent_row<T, fixed_channels>(values, planes + other,
                        densities.data() + other, density, row_squared, squared + (left - col),
                        right - left, count, channels, other, closest, parent);
                }
#endif
                for (; i < right - left; ++i) {
                    const auto col_squared = squared[left + i - col];
                    if (row_squared + col_squared >= closest || densities[other + i] <= density) continue;
                    const auto squared_distance = distance<T, fixed_channels>(
                        values, planes + other + i, row_squared,
                        col_squared, count, channels);
                    if (squared_distance < closest) {
                        closest = squared_distance;
                        parent = other + i;
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

template<bool vectorized, class T>
void dispatch_channels(const T* image, const T* noise, Index* labels,
                       Index height, Index width, Index channels, T kernel_size, T max_distance) {
    switch (channels) {
    case 1:
        segment<T, 1, vectorized>(image, noise, labels, height, width, channels, kernel_size, max_distance);
        break;
    case 3:
        segment<T, 3, vectorized>(image, noise, labels, height, width, channels, kernel_size, max_distance);
        break;
    default:
        segment<T, 0, vectorized>(image, noise, labels, height, width, channels, kernel_size, max_distance);
    }
}

} // namespace

template<class T>
void quickshift_into(const T* image, const T* noise, Index* labels,
                     Index height, Index width, Index channels, T kernel_size, T max_distance) {
#ifdef SKIMAGEXX_AVX2
    if (has_avx2() && std::fegetround() == FE_TONEAREST) {
        const bool bounded_density = !std::is_same_v<T, double> ||
            (kernel_size <= std::sqrt(std::numeric_limits<T>::max()) &&
             std::all_of(noise, noise + height * width, [](T jitter) {
                 return std::abs(jitter) <= T{1};
             }));
        if (bounded_density) {
            dispatch_channels<true>(image, noise, labels, height, width, channels, kernel_size, max_distance);
            return;
        }
    }
#endif
    dispatch_channels<false>(image, noise, labels, height, width, channels, kernel_size, max_distance);
}

template void quickshift_into(const float*, const float*, Index*, Index, Index, Index, float, float);
template void quickshift_into(const double*, const double*, Index*, Index, Index, Index, double, double);

} // namespace skimagexx

#if defined(_MSC_VER) && !defined(__clang__)
#pragma float_control(pop)
#endif
