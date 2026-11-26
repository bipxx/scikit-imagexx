#include "skimagexx/canny.hpp"

#include "execution.hpp"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <type_traits>

namespace skimagexx {
namespace {

template<class T> bool finite_image(const void* image, std::size_t size) {
    // Leave NumPy's floating overflow and nonfinite behavior to the provider.
    const auto* typed = static_cast<const T*>(image);
    const auto limit = std::sqrt(std::numeric_limits<T>::max()) / T(64);
    return std::all_of(typed, typed + size, [limit](T x) { return std::abs(x) <= limit; });
}

std::size_t compact_coordinate(std::size_t x, std::size_t extent,
                               std::size_t compact_extent, std::size_t radius) {
    if (extent == compact_extent || x < radius) return x;
    return x < extent - radius ? radius : compact_extent - (extent - x);
}

template<class T> void normalize(T* smooth, const std::uint8_t* mask, PixelType target,
                                 std::size_t rows, std::size_t cols, const CannyOptions& options,
                                 const std::vector<GaussianPass>& passes, int workers) {
    constexpr T epsilon = std::numeric_limits<T>::epsilon();
    const auto radius = options.weights.size() / 2;
    if (!mask && radius == 0) {
        detail::partition(rows * cols, workers, [&](auto begin, auto end, auto) {
            for (auto i = begin; i < end; ++i) smooth[i] /= T(1) + epsilon;
        });
        return;
    }

    // For an unmasked constant border, all samples farther than the kernel
    // radius from a side have the same Gaussian weight as the center.
    const auto bleed_rows = mask ? rows : std::min(rows, 2 * radius + 1);
    const auto bleed_cols = mask ? cols : std::min(cols, 2 * radius + 1);
    const auto bleed_size = bleed_rows * bleed_cols;
    std::unique_ptr<T[]> bleed(new T[bleed_size]);
    if (mask) {
        gaussian_nd_into(mask, bleed.get(), PixelType::boolean, target,
                         {bleed_rows, bleed_cols}, passes);
    } else {
        const std::vector<T> ones(bleed_size, T(1));
        gaussian_nd_into(ones.data(), bleed.get(), target, target,
                         {bleed_rows, bleed_cols}, passes);
    }
    for (std::size_t i = 0; i < bleed_size; ++i) bleed[i] += epsilon;
    if (mask) {
        detail::partition(rows * cols, workers, [&](auto begin, auto end, auto) {
            for (auto i = begin; i < end; ++i) smooth[i] /= bleed[i];
        });
    } else {
        detail::partition(rows, workers, [&](auto begin, auto end, auto) {
            const auto first = std::min(radius, cols);
            const auto last = cols - first;
            for (auto y = begin; y < end; ++y) {
                const auto row = compact_coordinate(y, rows, bleed_rows, radius);
                const auto* weights = bleed.get() + row * bleed_cols;
                auto* line = smooth + y * cols;
                std::size_t x = 0;
                for (; x < first; ++x) line[x] /= weights[x];
                for (; x < last; ++x) line[x] /= weights[radius];
                for (; x < cols; ++x) line[x] /= weights[compact_coordinate(x, cols, bleed_cols, radius)];
            }
        });
    }
}

template<class T> bool smooth_image(const void* image, PixelType type, T* smooth,
                                    const std::uint8_t* mask, std::size_t rows, std::size_t cols,
                                    const CannyOptions& options, int workers) {
    const auto count = rows * cols;
    constexpr auto target = std::is_same_v<T, float> ? PixelType::float32 : PixelType::float64;
    if (type == target && !finite_image<T>(image, count)) return false;
    const std::vector<std::size_t> shape{rows, cols};
    std::vector<GaussianPass> passes;
    if (!options.weights.empty()) {
        passes = {{0, options.weights, options.border, options.cval},
                  {1, options.weights, options.border, options.cval}};
    }
    if (mask) {
        std::unique_ptr<T[]> selected(new T[count]);
        gaussian_nd_into(image, selected.get(), type, target, shape, {});
        detail::partition(count, workers, [&](auto begin, auto end, auto) {
            for (auto i = begin; i < end; ++i) if (!mask[i]) selected[i] = T(0);
        });
        gaussian_nd_into(selected.get(), smooth, target, target, shape, passes);
    } else {
        gaussian_nd_into(image, smooth, type, target, shape, passes);
    }
    if (mask || options.constant_bleed) normalize(smooth, mask, target, rows, cols, options, passes, workers);
    return true;
}

template<class T> void sobel_sample(const T* above, const T* middle, const T* below,
                                    std::size_t x, std::size_t left, std::size_t right,
                                    T* vertical, T* horizontal, T* magnitude) {
    // Each derivative is rounded to the array dtype before the second
    // separable pass, whose three terms are accumulated in double precision.
    const auto vertical_difference = [&](std::size_t col) { return T(double(below[col]) - double(above[col])); };
    const auto horizontal_difference = [&](const T* row) { return T(double(row[right]) - double(row[left])); };
    const T v = T(2.0 * double(vertical_difference(x)) +
                  (double(vertical_difference(left)) + double(vertical_difference(right))));
    const T h = T(2.0 * double(horizontal_difference(middle)) +
                  (double(horizontal_difference(above)) + double(horizontal_difference(below))));
    vertical[x] = v;
    horizontal[x] = h;
    magnitude[x] = std::sqrt(T(v * v) + T(h * h));
}

template<class T> void sobel(const T* smooth, T* vertical, T* horizontal, T* magnitude,
                             std::size_t rows, std::size_t cols, int workers) {
    detail::partition(rows, workers, [&](auto begin, auto end, auto) {
        for (auto y = begin; y < end; ++y) {
            const auto* middle = smooth + y * cols;
            const auto* above = y ? middle - cols : middle;
            const auto* below = y + 1 < rows ? middle + cols : middle;
            auto* v = vertical + y * cols;
            auto* h = horizontal + y * cols;
            auto* m = magnitude + y * cols;
            sobel_sample(above, middle, below, 0, 0, 1, v, h, m);
            for (std::size_t x = 1; x + 1 < cols; ++x)
                sobel_sample(above, middle, below, x, x - 1, x + 1, v, h, m);
            sobel_sample(above, middle, below, cols - 1, cols - 2, cols - 1, v, h, m);
        }
    });
}

template<class T> double magnitude_quantile(T* ordered, std::size_t size, double fraction) {
    // NumPy computes q from percent, then uses separate interpolation forms
    // on each side of halfway. The difference keeps the input precision.
    const double position = double(size - 1) * ((100.0 * fraction) / 100.0);
    const auto rank = static_cast<std::size_t>(position);
    auto* const end = ordered + size;
    if (rank == size - 1) return double(*std::max_element(ordered, end));
    std::nth_element(ordered, ordered + rank, ordered + size);
    const auto lower = ordered[rank];
    const auto upper = *std::min_element(ordered + rank + 1, end);
    const double weight = position - double(rank);
    const double delta = double(T(upper - lower));
    return weight < 0.5 ? double(lower) + delta * weight : double(upper) - delta * (1.0 - weight);
}

bool eroded(const std::uint8_t* mask, std::size_t width) {
    return mask[-1] && mask[0] && mask[1] &&
           (mask - width)[-1] && (mask - width)[0] && (mask - width)[1] &&
           (mask + width)[-1] && (mask + width)[0] && (mask + width)[1];
}

template<class T> bool keep_peak(const T* magnitude, T dy, T dx, std::ptrdiff_t cols) {
    const auto abs_y = std::abs(dy), abs_x = std::abs(dx);
    const bool same = (dy >= 0 && dx >= 0) || (dy <= 0 && dx <= 0);
    // Upstream chooses the vertical case for a tie only in the opposite-sign
    // quadrant. In either case the diagonal weight of a tie is one.
    const bool vertical = same ? abs_y > abs_x : abs_y >= abs_x;
    const auto along = vertical ? (same ? cols : -cols) : 1;
    const auto diagonal = same ? cols + 1 : 1 - cols;
    const auto w = vertical ? abs_x / abs_y : abs_y / abs_x;
    const auto center = *magnitude;
    const double rest = 1.0 - double(w);
    return double(T(magnitude[diagonal] * w)) + double(magnitude[along]) * rest <= center &&
           double(T(magnitude[-diagonal] * w)) + double(magnitude[-along]) * rest <= center;
}

void connect_edges(std::uint8_t* states, std::size_t size, std::size_t cols) {
    if (std::find(states, states + size, std::uint8_t(2)) == states + size) return;
    std::vector<std::size_t> frontier;
    const auto promote = [&](std::size_t index) {
        if (states[index] == 2) {
            states[index] = 3;
            frontier.push_back(index);
        }
    };
    const auto visit = [&](std::size_t id) {
        promote(id - cols - 1);
        promote(id - cols);
        promote(id - cols + 1);
        promote(id - 1);
        promote(id + 1);
        promote(id + cols - 1);
        promote(id + cols);
        promote(id + cols + 1);
    };
    // Border pixels never enter the frontier. Every weak edge transitions at
    // most once. Promoted edges are distinct from the original strong seeds.
    for (std::size_t i = cols + 1; i < size - cols - 1; ++i) {
        if (states[i] != 1) continue;
        visit(i);
        while (!frontier.empty()) {
            const auto next = frontier.back();
            frontier.pop_back();
            visit(next);
        }
    }
    for (std::size_t i = 0; i < size; ++i)
        states[i] &= 1;
}

template<class T> bool detect(const void* image, PixelType type, const std::uint8_t* mask,
                              std::size_t rows, std::size_t cols, const CannyOptions& options,
                              std::uint8_t* output) {
    const auto size = rows * cols;
    if (rows < 3 || cols < 3) {
        if ((type == PixelType::float32 || type == PixelType::float64) && !finite_image<T>(image, size)) return false;
        std::fill_n(output, size, std::uint8_t(0));
        return true;
    }
    const int workers = detail::active_workers(size);
    std::unique_ptr<T[]> gradients(new T[3 * size]);
    auto* vertical = gradients.get();
    auto* horizontal = vertical + size;
    auto* magnitude = horizontal + size;
    {
        std::unique_ptr<T[]> smooth(new T[size]);
        if (!smooth_image(image, type, smooth.get(), mask, rows, cols, options, workers)) return false;
        sobel(smooth.get(), vertical, horizontal, magnitude, rows, cols, workers);
    }

    double low = options.low, high = options.high;
    if (options.quantiles) {
        std::unique_ptr<T[]> ordered(new T[size]);
        std::copy_n(magnitude, size, ordered.get());
        low = magnitude_quantile(ordered.get(), size, low);
        high = magnitude_quantile(ordered.get(), size, high);
    }
    const float low_cutoff = float(low) == 0 ? 1e-14F : float(low);
    if (!options.quantiles || !options.wide_quantile) high = double(T(high));
    std::fill_n(output, size, std::uint8_t(0));
    detail::partition(rows - 2, workers, [&](auto begin, auto end, auto) {
        for (auto y = begin + 1; y <= end; ++y) {
            for (auto i = y * cols + 1, stop = (y + 1) * cols - 1; i < stop; ++i) {
                const T m = magnitude[i];
                if (!(m > 0 && m >= T(low_cutoff)) || (mask && !eroded(mask + i, cols))) continue;
                if (keep_peak(magnitude + i, vertical[i], horizontal[i], static_cast<std::ptrdiff_t>(cols)))
                    output[i] = double(m) >= high ? std::uint8_t(1) : std::uint8_t(2);
            }
        }
    });
    connect_edges(output, size, cols);
    return true;
}

}  // namespace

bool canny_into(const void* image, PixelType type, const std::uint8_t* mask,
                std::size_t rows, std::size_t cols, const CannyOptions& options,
                std::uint8_t* output) {
    if (!image || !output || rows == 0 || cols == 0 || rows > canny_max_pixels / cols ||
        type < PixelType::boolean || type > PixelType::float64 || type == PixelType::uint64 || type == PixelType::int64 ||
        options.border < GaussianBorder::reflect || options.border > GaussianBorder::wrap ||
        (options.constant_bleed && options.border != GaussianBorder::constant) ||
        !std::isfinite(options.cval) || (options.border == GaussianBorder::constant &&
            (options.cval < 0 || options.cval > 1)) ||
        !std::isfinite(options.low) || !std::isfinite(options.high) || options.low > options.high ||
        std::max(std::abs(options.low), std::abs(options.high)) > std::numeric_limits<float>::max() ||
        (options.quantiles && (options.low < 0 || options.high > 1))) return false;
    const auto& weights = options.weights;
    if (!weights.empty()) {
        if (weights.size() > 2 * canny_max_radius + 1 || weights.size() % 2 == 0 ||
            !std::equal(weights.begin(), weights.end(), weights.rbegin()) ||
            !std::all_of(weights.begin(), weights.end(), [](double w) { return w >= 0 && w <= 1; }) ||
            std::abs(std::accumulate(weights.begin(), weights.end(), 0.0) - 1.0) > 1e-12) return false;
    }
    if (std::fegetround() != FE_TONEAREST) return false;
    if (type == PixelType::float32) return detect<float>(image, type, mask, rows, cols, options, output);
    return detect<double>(image, type, mask, rows, cols, options, output);
}

}  // namespace skimagexx
