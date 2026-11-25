#include "skimagexx/blob_detectors.hpp"

#include "skimagexx/gaussian_nd.hpp"
#include "skimagexx/hessian_det.hpp"
#include "execution.hpp"
#include "simd.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <type_traits>

namespace skimagexx {
namespace {

constexpr std::size_t maximum_responses = 16 * 1024 * 1024;
constexpr std::size_t maximum_peaks = 4096;
constexpr double root_two = 1.4142135623730950488;
constexpr double pi = 3.1415926535897932385;

GaussianPass make_pass(std::size_t axis, double sigma, bool derivative) {
    const int radius = int(4 * sigma + 0.5);
    const double variance = sigma * sigma;
    std::vector<double> weights(std::size_t(2 * radius + 1));
    for (int offset = -radius; offset <= radius; ++offset)
        weights[std::size_t(offset + radius)] = std::exp((-0.5 / variance) * (offset * offset));
    const double sum = std::accumulate(weights.begin(), weights.end(), 0.0);
    for (int offset = -radius; offset <= radius; ++offset) {
        double& weight = weights[std::size_t(offset + radius)];
        weight /= sum;
        if (derivative) {
            const double inverse = 1 / variance;
            weight *= -inverse + (offset * offset) * (inverse * inverse);
        }
    }
    return {axis, std::move(weights), GaussianBorder::reflect};
}

template<class T>
bool valid_image(const T* data, std::size_t count) {
    return std::all_of(data, data + count, [](T value) { return std::isfinite(value) && std::abs(value) <= T(1e12); });
}

std::size_t reflected(std::ptrdiff_t coordinate, std::ptrdiff_t width) {
    const auto period = 2 * width;
    auto index = (coordinate % period + period) % period;
    if (index >= width)
        index = period - 1 - index;
    return std::size_t(index);
}

template<class T>
void horizontal(const T* input, T* output, std::size_t height, std::size_t width, const GaussianPass& pass) {
    const auto radius = pass.weights.size() / 2;
    // The reflected edges have the same coordinate maps on every row. Padding
    // gives both scalar and SIMD kernels the same symmetric contiguous taps.
    std::vector<std::size_t> left_edge(radius), right_edge(radius);
    for (std::size_t j = 0; j < radius; ++j) {
        left_edge[j] = reflected(std::ptrdiff_t(j) - std::ptrdiff_t(radius), std::ptrdiff_t(width));
        right_edge[j] = reflected(std::ptrdiff_t(width + j), std::ptrdiff_t(width));
    }
    const auto center_weight = pass.weights[radius];
    detail::partition(height, detail::active_workers(height * width), [&](auto begin, auto end, auto) {
        std::vector<T> padded(width + 2 * radius);
        auto* center = padded.data() + radius;
#ifdef SKIMAGEXX_AVX2
        const bool vectorized = has_avx2();
        constexpr bool single = std::is_same_v<T, float>;
        std::vector<const void*> left(radius), right(radius);
        for (std::size_t j = 0; j < radius; ++j) {
            left[j] = padded.data() + j;
            right[j] = padded.data() + 2 * radius - j;
        }
#endif
        for (auto y = begin; y < end; ++y) {
            const auto* row = input + y * width;
            auto* result = output + y * width;
            std::copy_n(row, width, center);
            for (std::size_t j = 0; j < radius; ++j) {
                padded[j] = row[left_edge[j]];
                center[width + j] = row[right_edge[j]];
            }
            std::size_t x = 0;
#ifdef SKIMAGEXX_AVX2
            if (vectorized)
                x = simd::gaussian_rows(left.data(), right.data(), center, result, width, radius,
                                       pass.weights.data(), center_weight, 0, single, single);
#endif
            for (; x < width; ++x) {
                double sum = double(center[x]) * center_weight;
                for (std::size_t j = 0; j < radius; ++j)
                    sum += (double(padded[x + j]) + double(padded[x + 2 * radius - j])) * pass.weights[j];
                result[x] = T(sum);
            }
        }
    });
}

template<class T>
bool make_responses(const void* image, PixelType source, const std::vector<std::size_t>& shape,
                    std::size_t count, const BlobDetectionOptions& options, T* output) {
    constexpr auto precision = std::is_same_v<T, float> ? PixelType::float32 : PixelType::float64;
    const auto& scales = options.scales;
    if (options.method == BlobMethod::doh) {
        std::unique_ptr<double[]> temporary;
        if constexpr (std::is_same_v<T, float>)
            temporary.reset(new double[count]);
        for (std::size_t i = 0; i < scales.size(); ++i) {
            double* result;
            if constexpr (std::is_same_v<T, float>)
                result = temporary.get();
            else
                result = output + i * count;
            if (!hessian_det_approx_into(image, source, result, shape[0], shape[1],
                                         std::int64_t(3 * scales[i][0])))
                return false;
            if constexpr (std::is_same_v<T, float>)
                std::transform(result, result + count, output + i * count,
                               [](double value) { return static_cast<float>(value); });
        }
        return true;
    }

    std::unique_ptr<double[]> normalized;
    if (source == PixelType::uint8) {
        normalized.reset(new double[count]);
        gaussian_nd_into(image, normalized.get(), source, precision, shape, {});
        image = normalized.get();
        source = precision;
    }
    std::unique_ptr<T[]> temporary(new T[count]);
    std::unique_ptr<T[]> vertical(new T[count]);
    const auto smooth = [&](const auto& scale, bool dy, bool dx, T* result) {
        gaussian_nd_into(image, vertical.get(), source, precision, shape, {make_pass(0, scale[0], dy)});
        horizontal(vertical.get(), result, shape[0], shape[1], make_pass(1, scale[1], dx));
    };
    if (options.method == BlobMethod::dog) {
        std::unique_ptr<T[]> previous(new T[count]);
        smooth(scales.front(), false, false, previous.get());
        for (std::size_t i = 1; i < scales.size(); ++i) {
            smooth(scales[i], false, false, temporary.get());
            auto* response = output + (i - 1) * count;
            detail::partition(count, detail::active_workers(count), [&](auto begin, auto end, auto) {
                for (auto j = begin; j < end; ++j)
                    response[j] = (previous[j] - temporary[j]) * T(options.dog_normalization);
            });
            std::swap(previous, temporary);
        }
        return true;
    }

    for (std::size_t i = 0; i < scales.size(); ++i) {
        const auto& scale = scales[i];
        auto* response = output + i * count;
        smooth(scale, true, false, response);
        smooth(scale, false, true, temporary.get());
        const T mean = (T(scale[0]) + T(scale[1])) / T(2);
        const T factor = mean * mean;
        detail::partition(count, detail::active_workers(count), [&](auto begin, auto end, auto) {
            for (auto j = begin; j < end; ++j)
                response[j] = -(response[j] + temporary[j]) * factor;
        });
    }
    return true;
}

template<class T>
double disk_overlap(const BlobDetection& first, const BlobDetection& second) {
    const auto& sigma = first.sigma[1] > second.sigma[1] ? first.sigma : second.sigma;
    const T largest = T(sigma[1]);
    const T r1 = T(first.sigma[1]) / largest;
    const T r2 = T(second.sigma[1]) / largest;
    const T row_scale = T(sigma[0]) * T(root_two);
    const T col_scale = largest * T(root_two);
    const T dy = T(second.row) / row_scale - T(first.row) / row_scale;
    const T dx = T(second.column) / col_scale - T(first.column) / col_scale;
    const T d = std::sqrt(dy * dy + dx * dx);
    if (d > r1 + r2)
        return 0;
    if (d <= std::abs(r1 - r2))
        return 1;
    const T d2 = d * d, r12 = r1 * r1, r22 = r2 * r2;
    const double a1 = std::acos(std::clamp((d2 + r12 - r22) / (T(2) * d * r1), T(-1), T(1)));
    const double a2 = std::acos(std::clamp((d2 + r22 - r12) / (T(2) * d * r2), T(-1), T(1)));
    const T product = (-d + r2 + r1) * (d - r2 + r1) * (d + r2 - r1) * (d + r2 + r1);
    const T min_radius = std::min(r1, r2);
    return (double(r12) * a1 + double(r22) * a2 - 0.5 * std::sqrt(double(std::abs(product)))) /
           (pi * double(min_radius * min_radius));
}

template<class T>
bool prune(std::vector<BlobDetection>& detections, double overlap) {
    if (detections.size() < 2 || overlap == 1)
        return true;
    double max_sigma = 0;
    for (const auto& blob : detections)
        max_sigma = std::max({max_sigma, blob.sigma[0], blob.sigma[1]});
    const double distance = double(T(2) * T(max_sigma) * T(root_two));
    const double distance_squared = distance * distance;
    std::vector<std::size_t> priority(detections.size());
    std::iota(priority.begin(), priority.end(), 0);
    std::sort(priority.begin(), priority.end(), [&](std::size_t a, std::size_t b) {
        const double x = detections[a].sigma[1], y = detections[b].sigma[1];
        return x == y ? a > b : x > y;
    });
    std::vector<bool> removed(detections.size());
    for (std::size_t rank = 1; rank < priority.size(); ++rank) {
        const auto current = priority[rank];
        bool suppressed_by_removed = false;
        for (std::size_t higher = 0; higher < rank; ++higher) {
            const auto competitor = priority[higher];
            const auto& first = detections[std::min(current, competitor)];
            const auto& second = detections[std::max(current, competitor)];
            const double dy = double(first.row) - double(second.row);
            const double dx = double(first.column) - double(second.column);
            if (dy * dy + dx * dx > distance_squared)
                continue;
            const double fraction = disk_overlap<T>(first, second);
            if (fraction > 0 && fraction < 1 && std::abs(fraction - overlap) < 16 * std::numeric_limits<T>::epsilon())
                return false;
            if (fraction <= overlap)
                continue;
            if (!removed[competitor]) {
                removed[current] = true;
                break;
            }
            suppressed_by_removed = true;
        }
        // A survivor always eliminates the current blob. An already defeated
        // blob can eliminate it only if its upstream pair is visited first.
        if (!removed[current] && suppressed_by_removed)
            return false;
    }
    std::size_t next = 0;
    for (std::size_t i = 0; i < detections.size(); ++i) {
        if (!removed[i])
            detections[next++] = detections[i];
    }
    detections.resize(next);
    return true;
}

template<class T>
std::optional<std::vector<BlobDetection>> detect(const void* image, PixelType source,
    std::size_t height, std::size_t width, std::size_t count, const BlobDetectionOptions& options,
    std::size_t number_of_scales) {
    if (source != PixelType::uint8 && !valid_image(static_cast<const T*>(image), count))
        return std::nullopt;
    const auto total = number_of_scales * count;
    std::unique_ptr<T[]> responses(new T[total]);
    if (!make_responses(image, source, {height, width}, count, options, responses.get()))
        return std::nullopt;
    const auto* data = responses.get();
    std::vector<BlobDetection> detections;
    double cutoff;
    if (!options.threshold && options.threshold_relative) {
        const auto [minimum, maximum] = std::minmax_element(data, data + total);
        cutoff = std::max(double(*minimum), double(T(*options.threshold_relative) * *maximum));
    } else {
        cutoff = options.threshold ? *options.threshold : double(*std::min_element(data, data + total));
        if (options.threshold_relative)
            cutoff = std::max(cutoff, double(T(*options.threshold_relative) * *std::max_element(data, data + total)));
    }
    const T threshold = T(cutoff);
    if (total > 1 && data[0] > threshold &&
        std::all_of(data + 1, data + total, [&](T value) { return value == data[0]; }))
        return detections;

    struct Peak { std::size_t position, scale; };
    std::vector<Peak> peaks;
    const auto by = options.exclude_border[0], bx = options.exclude_border[1];
    if (by >= height || bx >= width)
        return detections;
    for (std::size_t scale = 0; scale < number_of_scales; ++scale) {
        const auto* plane = data + scale * count;
        const auto first_scale = scale ? scale - 1 : scale;
        const auto last_scale = std::min(scale + 1, number_of_scales - 1);
        for (std::size_t y = by; y < height - by; ++y) {
            for (std::size_t x = bx; x < width - bx; ++x) {
                const auto pos = y * width + x;
                const T value = plane[pos];
                if (value <= threshold)
                    continue;
                bool peak = true;
                const auto x0 = x ? x - 1 : x, x1 = std::min(x + 1, width - 1);
                const auto y0 = y ? y - 1 : y, y1 = std::min(y + 1, height - 1);
                for (auto s = first_scale; peak && s <= last_scale; ++s) {
                    for (auto r = y0; peak && r <= y1; ++r) {
                        const auto* row = data + s * count + r * width;
                        for (auto c = x0; c <= x1; ++c)
                            peak = peak && value >= row[c];
                    }
                }
                if (peak) {
                    if (peaks.size() == maximum_peaks)
                        return std::nullopt;
                    peaks.push_back({pos, scale});
                }
            }
        }
    }
    std::sort(peaks.begin(), peaks.end(), [&](const Peak& a, const Peak& b) {
        const T x = data[a.scale * count + a.position], y = data[b.scale * count + b.position];
        if (x != y)
            return x > y;
        if (a.position != b.position)
            return a.position < b.position;
        return a.scale < b.scale;
    });
    detections.reserve(peaks.size());
    for (const auto& peak : peaks)
        detections.push_back({peak.position / width, peak.position % width, options.scales[peak.scale]});
    const bool valid = options.method == BlobMethod::doh || !std::is_same_v<T, float>
        ? prune<double>(detections, options.overlap) : prune<float>(detections, options.overlap);
    if (!valid)
        return std::nullopt;
    return detections;
}

}  // namespace

std::optional<std::vector<BlobDetection>> blob_detect(const void* image, std::size_t count,
    PixelType type, std::size_t height, std::size_t width, const BlobDetectionOptions& options) {
    const bool dog = options.method == BlobMethod::dog;
    const bool doh = options.method == BlobMethod::doh;
    if ((!dog && !doh && options.method != BlobMethod::log) || !image ||
        (type != PixelType::uint8 && type != PixelType::float32 && type != PixelType::float64) ||
        reinterpret_cast<std::uintptr_t>(image) % pixel_size(type) ||
        height == 0 || width == 0 || height > 4096 || width > 4096 || count != height * width ||
        options.scales.size() <= std::size_t(dog) || options.scales.size() > 64 + std::size_t(dog) ||
        !std::isfinite(options.dog_normalization) || options.dog_normalization <= 0 || options.dog_normalization > 1024 ||
        !std::isfinite(options.overlap) || options.overlap < 0 || options.overlap > 1 ||
        (options.threshold && (!std::isfinite(*options.threshold) || std::abs(*options.threshold) > 1e12)) ||
        (options.threshold_relative && (!std::isfinite(*options.threshold_relative) ||
                                        *options.threshold_relative < 0 || *options.threshold_relative > 1)))
        throw std::invalid_argument("invalid blob detector input or options");
    const auto responses = options.scales.size() - std::size_t(dog);
    if (count > maximum_responses / responses)
        return std::nullopt;
    for (const auto& scale : options.scales) {
        if (!(scale[0] >= 0.5 && scale[0] <= 128 && scale[1] >= 0.5 && scale[1] <= 128) ||
            (doh && scale[0] != scale[1]))
            throw std::invalid_argument("invalid blob detector scale");
    }
    if (type == PixelType::float32)
        return detect<float>(image, type, height, width, count, options, responses);
    return detect<double>(image, type, height, width, count, options, responses);
}

}  // namespace skimagexx
