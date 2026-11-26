// SLIC iteration and connectivity adapted from scikit-image 0.26.0
// src/skimage/segmentation/_slic.pyx (BSD-3-Clause).
// Copyright 2009-2022 the scikit-image team.
// See https://github.com/scikit-image/scikit-image/blob/v0.26.0/LICENSE.txt
#include "skimagexx/slic.hpp"
#include "skimagexx/image.hpp"
#include "skimagexx/regular_grid.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

#if defined(SKIMAGEXX_AVX2) && (defined(__SSE2__) || defined(_M_X64))
#include <emmintrin.h>
#define SLIC_SSE2
#endif

namespace skimagexx {
namespace {

using Index = std::ptrdiff_t;
using Label = std::int32_t;
using Point = std::array<Index, 3>;

struct Volume {
    Point shape;
    Index plane;
    Index pixels;
    Index channels;
};

Volume image_layout(const void* image, std::size_t image_size,
                    const std::array<std::size_t, 4>& shape,
                    const Index* output, std::size_t output_size) {
    std::size_t count = 1;
    for (auto extent : shape) {
        if (!extent || extent > (1u << 20) || count > (1u << 26) / extent)
            throw std::invalid_argument("SLIC requires a bounded nonempty image");
        count *= extent;
    }
    if (!image || !output || image_size != count || output_size != count / shape[3])
        throw std::invalid_argument("SLIC image or label storage does not match the shape");
    return {{Index(shape[0]), Index(shape[1]), Index(shape[2])},
            Index(shape[1] * shape[2]), Index(output_size), Index(shape[3])};
}

std::array<RegularGridSlice, 3> grid(const Volume& volume, std::size_t points) {
    const std::array<std::int64_t, 3> shape{
        volume.shape[0], volume.shape[1], volume.shape[2]};
    std::array<RegularGridSlice, 3> result{};
    if (!detail::regular_grid_slices_i64(shape.data(), shape.size(), points, result.data()))
        throw std::invalid_argument("invalid SLIC grid");
    return result;
}

struct Grid {
    Point start;
    Point step;
    Index count;
};

Grid initial_grid(const Volume& volume, std::size_t points) {
    const auto slices = grid(volume, points);
    Grid result{{}, {}, 1};
    for (std::size_t axis = 0; axis < slices.size(); ++axis) {
        const Index start = slices[axis].start;
        const Index step = slices[axis].step;
        result.start[axis] = start;
        result.step[axis] = step;
        result.count *= (volume.shape[axis] - start + step - 1) / step;
    }
    return result;
}

struct QueuePoint {
    Point coordinate;
    Index offset;
};

void connect_2d(const Volume& volume, const Label* labels, Index* output,
                Index min_size, Index max_size, Index start) {
    using Offset = std::int32_t;
    struct Pixel {
        Offset offset;
        Offset column;
    };
    const auto width = static_cast<Offset>(volume.shape[2]);
    const auto pixels = static_cast<Offset>(volume.pixels);
    const Index missing = start - 1;
    std::fill_n(output, pixels, missing);
    const auto limit = static_cast<std::size_t>(max_size);
    std::unique_ptr<Pixel[]> queue(new Pixel[limit]);
    Index next_label = start;
    Offset origin = 0;
    while (origin < pixels) {
        for (Offset x = 0; x < width; ++x, ++origin) {
            const Label label = labels[origin];
            if (label == missing || output[origin] != missing)
                continue;
            Index adjacent = next_label;
            output[origin] = next_label;
            queue[0] = {origin, x};
            std::size_t queued = 1;
            const auto visit = [&](Offset offset, Offset column) {
                const Index previous = output[offset];
                if (labels[offset] == label && previous == missing) {
                    output[offset] = next_label;
                    queue[queued++] = {offset, column};
                    return queued == limit;
                }
                if (previous != missing && previous != next_label)
                    adjacent = previous;
                return false;
            };
            for (std::size_t visited = 0; visited < queued && queued < limit; ++visited) {
                const auto [offset, column] = queue[visited];
                if ((column + 1 < width && visit(offset + 1, column + 1)) ||
                    (column > 0 && visit(offset - 1, column - 1)) ||
                    (offset + width < pixels && visit(offset + width, column)))
                    break;
                if (offset >= width)
                    visit(offset - width, column);
            }
            if (queued < static_cast<std::size_t>(min_size)) {
                for (std::size_t i = 0; i < queued; ++i)
                    output[queue[i].offset] = adjacent;
            } else {
                ++next_label;
            }
        }
    }
}

void connect(const Volume& volume, const Label* labels, Index* output,
             Index min_size, Index max_size, Index start) {
    const Index missing = start - 1;
    std::fill_n(output, volume.pixels, missing);
    const auto [depth, height, width] = volume.shape;
    constexpr Point deltas[] = {{0, 0, 1}, {0, 0, -1}, {0, 1, 0},
                                {0, -1, 0}, {1, 0, 0}, {-1, 0, 0}};
    std::vector<QueuePoint> queue;
    queue.reserve(std::min(max_size, volume.pixels));
    Index next_label = start;
    Index origin = 0;
    for (Index z = 0; z < depth; ++z) {
        for (Index y = 0; y < height; ++y) {
            for (Index x = 0; x < width; ++x, ++origin) {
                const Index label = labels[origin];
                if (label == missing || output[origin] != missing)
                    continue;
                output[origin] = next_label;
                Index adjacent = next_label;
                queue.clear();
                queue.push_back({{z, y, x}, origin});
                for (std::size_t visited = 0;
                     visited < queue.size() && queue.size() < std::size_t(max_size);
                     ++visited) {
                    const QueuePoint point = queue[visited];
                    for (const auto& delta : deltas) {
                        const Index nz = point.coordinate[0] + delta[0];
                        const Index ny = point.coordinate[1] + delta[1];
                        const Index nx = point.coordinate[2] + delta[2];
                        if (nz < 0 || nz >= depth || ny < 0 || ny >= height || nx < 0 || nx >= width)
                            continue;
                        const Index offset = point.offset + delta[0] * volume.plane +
                                             delta[1] * width + delta[2];
                        const Index previous = output[offset];
                        if (labels[offset] == label && previous == missing) {
                            output[offset] = next_label;
                            queue.push_back({{nz, ny, nx}, offset});
                            if (queue.size() == std::size_t(max_size))
                                break;
                        } else if (previous != missing && previous != next_label) {
                            adjacent = previous;
                        }
                    }
                }
                if (queue.size() < std::size_t(min_size)) {
                    for (const auto& point : queue)
                        output[point.offset] = adjacent;
                } else {
                    ++next_label;
                }
            }
        }
    }
}

template<class T>
T square(T value) {
    return value * value;
}

template<class T>
T color_distance(const T* values, const T* center, Index channels, Index plane_size) {
    T result = 0;
    for (Index c = 0; c < channels; ++c)
        result += square(values[c * plane_size] - center[c]);
    return result;
}

#ifdef SLIC_SSE2
template<class T, Index FixedChannels, bool Scale>
Index assign_vector(const T* image, const T* center, const T* x_distances,
                    T* distances, Label* nearest, Index length,
                    Index channels, Index pixels, Label label, T color_scale,
                    T spatial_yz, T spatial_weight, bool& change) {
    const auto count = FixedChannels ? FixedChannels : channels;
    const auto labels = _mm_set1_epi32(label);
    int changed = 0;
    Index x = 0;
    if constexpr (std::is_same_v<T, double>) {
        const auto yz = _mm_set1_pd(spatial_yz);
        const auto weight = _mm_set1_pd(spatial_weight);
        const auto scale = _mm_set1_pd(color_scale);
        for (; x + 2 <= length; x += 2) {
            const auto spatial = _mm_mul_pd(_mm_add_pd(yz, _mm_loadu_pd(x_distances + x)), weight);
            const auto previous = _mm_loadu_pd(distances + x);
            if (!_mm_movemask_pd(_mm_cmplt_pd(spatial, previous)))
                continue;
            auto color = _mm_setzero_pd();
            for (Index c = 0; c < count; ++c) {
                const auto difference = _mm_sub_pd(_mm_loadu_pd(image + c * pixels + x),
                                                   _mm_set1_pd(center[c]));
                color = _mm_add_pd(color, _mm_mul_pd(difference, difference));
            }
            if constexpr (Scale)
                color = _mm_div_pd(color, scale);
            const auto distance = _mm_add_pd(spatial, color);
            const auto mask = _mm_cmplt_pd(distance, previous);
            const auto updated = _mm_movemask_pd(mask);
            if (updated) {
                _mm_storeu_pd(distances + x, _mm_or_pd(_mm_and_pd(mask, distance),
                                                       _mm_andnot_pd(mask, previous)));
                const auto label_mask = _mm_shuffle_epi32(_mm_castpd_si128(mask),
                                                           _MM_SHUFFLE(2, 0, 2, 0));
                auto* current = reinterpret_cast<__m128i*>(nearest + x);
                const auto old_labels = _mm_loadl_epi64(current);
                _mm_storel_epi64(current, _mm_or_si128(_mm_and_si128(label_mask, labels),
                                                      _mm_andnot_si128(label_mask, old_labels)));
                changed = 1;
            }
        }
    } else {
        const auto yz = _mm_set1_ps(spatial_yz);
        const auto weight = _mm_set1_ps(spatial_weight);
        const auto scale = _mm_set1_ps(color_scale);
        for (; x + 4 <= length; x += 4) {
            const auto spatial = _mm_mul_ps(_mm_add_ps(yz, _mm_loadu_ps(x_distances + x)), weight);
            const auto previous = _mm_loadu_ps(distances + x);
            if (!_mm_movemask_ps(_mm_cmplt_ps(spatial, previous)))
                continue;
            auto color = _mm_setzero_ps();
            for (Index c = 0; c < count; ++c) {
                const auto difference = _mm_sub_ps(_mm_loadu_ps(image + c * pixels + x),
                                                   _mm_set1_ps(center[c]));
                color = _mm_add_ps(color, _mm_mul_ps(difference, difference));
            }
            if constexpr (Scale)
                color = _mm_div_ps(color, scale);
            const auto distance = _mm_add_ps(spatial, color);
            const auto mask = _mm_cmplt_ps(distance, previous);
            if (_mm_movemask_ps(mask)) {
                _mm_storeu_ps(distances + x, _mm_or_ps(_mm_and_ps(mask, distance),
                                                       _mm_andnot_ps(mask, previous)));
                const auto label_mask = _mm_castps_si128(mask);
                auto* current = reinterpret_cast<__m128i*>(nearest + x);
                const auto old_labels = _mm_loadu_si128(current);
                _mm_storeu_si128(current, _mm_or_si128(_mm_and_si128(label_mask, labels),
                                                      _mm_andnot_si128(label_mask, old_labels)));
                changed = 1;
            }
        }
    }
    change |= changed != 0;
    return x;
}
#endif

template<class T, Index FixedChannels>
bool cluster(const T* image, const Volume& volume, const Grid& seeds,
             const SlicOptions& options, Label* nearest) {
    const auto [depth, height, width] = volume.shape;
    const Index channels = FixedChannels ? FixedChannels : volume.channels;
    const Index features = 3 + channels;
    const auto [z0, y0, x0] = seeds.start;
    const auto [seed_z, seed_y, seed_x] = seeds.step;
    const auto windows = grid(volume, seeds.count);
    const Index window_z = 2 * windows[0].step;
    const Index window_y = 2 * windows[1].step;
    const Index window_x = 2 * windows[2].step;
    const T sz = static_cast<T>(options.spacing[0]);
    const T sy = static_cast<T>(options.spacing[1]);
    const T sx = static_cast<T>(options.spacing[2]);
    const float step = static_cast<float>(std::max({seed_z, seed_y, seed_x}));
    const T spatial_weight = static_cast<T>(1.0 / (step * step));
    const T initial_distance = std::is_same_v<T, float>
        ? std::numeric_limits<T>::infinity() : std::numeric_limits<T>::max();
    const auto start = static_cast<Label>(options.start_label);
    const Label missing = start - 1;
#ifdef SLIC_SSE2
    const bool vectorized = has_avx2();
#endif

    std::vector<T> centers(seeds.count * features);
    std::vector<Index> members(seeds.count);
    std::vector<T> maximum_color(seeds.count, T(1));
    std::vector<T> distances(volume.pixels);
    std::vector<T> x_distances(width);
    Index seed = 0;
    for (Index z = z0; z < depth; z += seed_z)
        for (Index y = y0; y < height; y += seed_y)
            for (Index x = x0; x < width; x += seed_x, ++seed) {
                centers[seed * features] = static_cast<T>(z);
                centers[seed * features + 1] = static_cast<T>(y);
                centers[seed * features + 2] = static_cast<T>(x);
            }
    std::fill_n(nearest, volume.pixels, missing);
    for (std::size_t iteration = 0; iteration < options.max_num_iter; ++iteration) {
        std::fill(distances.begin(), distances.end(), initial_distance);
        bool change = false;
        for (Index k = 0; k < seeds.count; ++k) {
            if (iteration && !members[k])
                continue;
            const T* center = centers.data() + k * features;
            const T cz = center[0];
            const T cy = center[1];
            const T cx = center[2];
            const Index z_min = static_cast<Index>(std::max(cz - T(window_z), T(0)));
            const Index y_min = static_cast<Index>(std::max(cy - T(window_y), T(0)));
            const Index x_min = static_cast<Index>(std::max(cx - T(window_x), T(0)));
            const Index z_max = static_cast<Index>(std::min(cz + T(window_z) + T(1), T(depth)));
            const Index y_max = static_cast<Index>(std::min(cy + T(window_y) + T(1), T(height)));
            const Index x_max = static_cast<Index>(std::min(cx + T(window_x) + T(1), T(width)));
            const T color_scale = options.slic_zero ? maximum_color[k] : T(1);
            for (Index x = x_min; x < x_max; ++x)
                x_distances[x] = square(sx * (cx - T(x)));
            const Label label = static_cast<Label>(k) + start;
            const auto assign = [&](auto scaled) {
                constexpr bool Scale = decltype(scaled)::value;
                for (Index z = z_min; z < z_max; ++z) {
                    const T dz = square(sz * (cz - T(z)));
                    for (Index y = y_min; y < y_max; ++y) {
                        const T spatial_yz = dz + square(sy * (cy - T(y)));
                        Index offset = z * volume.plane + y * width + x_min;
                        Index x = x_min;
#ifdef SLIC_SSE2
                        if (vectorized && x_max - x_min >= Index(16 / sizeof(T))) {
                            const auto consumed = assign_vector<T, FixedChannels, Scale>(
                                image + offset, center + 3, x_distances.data() + x_min,
                                distances.data() + offset, nearest + offset, x_max - x_min,
                                channels, volume.pixels, label, color_scale, spatial_yz,
                                spatial_weight, change);
                            offset += consumed;
                            x += consumed;
                        }
#endif
                        for (; x < x_max; ++x, ++offset) {
                            const T spatial = (spatial_yz + x_distances[x]) * spatial_weight;
                            if (spatial >= distances[offset])
                                continue;
                            T color = color_distance(image + offset, center + 3, channels, volume.pixels);
                            if constexpr (Scale)
                                color /= color_scale;
                            const T distance = spatial + color;
                            if (distances[offset] > distance) {
                                nearest[offset] = label;
                                distances[offset] = distance;
                                change = true;
                            }
                        }
                    }
                }
            };
            if (color_scale == T(1))
                assign(std::false_type{});
            else
                assign(std::true_type{});
        }
        if (!change)
            break;

        std::fill(members.begin(), members.end(), 0);
        std::fill(centers.begin(), centers.end(), T(0));
        Index offset = 0;
        for (Index z = 0; z < depth; ++z)
            for (Index y = 0; y < height; ++y)
                for (Index x = 0; x < width; ++x, ++offset) {
                    if (nearest[offset] == missing)
                        return false;
                    const Index k = nearest[offset] - start;
                    ++members[k];
                    T* center = centers.data() + k * features;
                    center[0] += static_cast<T>(z);
                    center[1] += static_cast<T>(y);
                    center[2] += static_cast<T>(x);
                    for (Index c = 0; c < channels; ++c)
                        center[c + 3] += image[c * volume.pixels + offset];
                }
        for (Index k = 0; k < seeds.count; ++k) {
            if (!members[k])
                continue;
            for (Index c = 0; c < features; ++c)
                centers[k * features + c] /= static_cast<T>(members[k]);
        }
        if (options.slic_zero)
            for (Index p = 0; p < volume.pixels; ++p) {
                const Index k = nearest[p] - start;
                const T color = color_distance(image + p, centers.data() + k * features + 3,
                                               channels, volume.pixels);
                if (maximum_color[k] < color)
                    maximum_color[k] = color;
            }
    }
    return true;
}

template<class T>
bool run_slic(const T* image, const Volume& volume, const SlicOptions& options, Index* output) {
    if (!options.n_segments || options.n_segments > (1u << 26) ||
        options.start_label < 0 || options.start_label > 1)
        throw std::invalid_argument("invalid SLIC segment count or start label");
    for (double spacing : options.spacing)
        if (spacing < 1. / 1024 || spacing > 1024. || !std::isfinite(spacing))
            return false;
    const auto seeds = initial_grid(volume, options.n_segments);
    if (!seeds.count)
        return false;

    Index min_size = 0;
    Index max_size = 0;
    std::vector<Label> nearest(volume.pixels);
    if (options.enforce_connectivity) {
        const double nominal = double(volume.pixels) / double(seeds.count);
        const double min = options.min_size_factor * nominal;
        const double max = options.max_size_factor * nominal;
        if (!std::isfinite(min) || !std::isfinite(max) || min < 0 || max < 1)
            return false;
        min_size = static_cast<Index>(std::min(min, double(volume.pixels) + 1));
        max_size = static_cast<Index>(std::min(max, double(volume.pixels)));
    }
    const bool ok = volume.channels == 1 ? cluster<T, 1>(image, volume, seeds, options, nearest.data()) :
                    volume.channels == 3 ? cluster<T, 3>(image, volume, seeds, options, nearest.data()) :
                                           cluster<T, 0>(image, volume, seeds, options, nearest.data());
    if (!ok)
        return false;
    if (!options.enforce_connectivity)
        std::copy(nearest.begin(), nearest.end(), output);
    else if (volume.shape[0] == 1)
        connect_2d(volume, nearest.data(), output, min_size, max_size, options.start_label);
    else
        connect(volume, nearest.data(), output, min_size, max_size, options.start_label);
    return true;
}

template<class T>
using Working = std::conditional_t<std::is_same_v<T, float>, float, double>;

template<class T>
Working<T> pixel(T value) {
    if constexpr (std::is_same_v<T, std::uint8_t>)
        return double(value) * (1. / 255.);
    else
        return value;
}

}

template<class T>
bool slic_into(const T* image, std::size_t image_size,
               const std::array<std::size_t, 4>& shape, const SlicOptions& options,
               std::ptrdiff_t* output, std::size_t output_size) {
    const Volume volume = image_layout(image, image_size, shape, output, output_size);
    using F = Working<T>;
    if (!(options.compactness >= 1. / 1024 && options.compactness <= 1024.))
        return false;
    T minimum = image[0], maximum = image[0];
    for (std::size_t i = 0; i < image_size; ++i) {
        const T value = image[i];
        if constexpr (std::is_floating_point_v<T>)
            if (!std::isfinite(value))
                return false;
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
    }
    const F low = pixel(minimum);
    const F span = pixel(maximum) - low;
    if (!std::isfinite(span))
        return false;
    const F inverse_compactness = static_cast<F>(1. / options.compactness);
    std::vector<F> features(image_size);
    if (span != F(0))
        for (Index c = 0; c < volume.channels; ++c)
            for (Index p = 0; p < volume.pixels; ++p)
                features[c * volume.pixels + p] =
                    ((pixel(image[p * volume.channels + c]) - low) / span) * inverse_compactness;
    return run_slic(features.data(), volume, options, output);
}

template<class T>
bool slic_prepared_into(const T* features, std::size_t image_size,
                        const std::array<std::size_t, 4>& shape, const SlicOptions& options,
                        std::ptrdiff_t* output, std::size_t output_size) {
    const Volume volume = image_layout(features, image_size, shape, output, output_size);
    // Normalized Lab/smoothed features with the admitted compactness bound
    // fit comfortably in this range, including large float32 centroid sums.
    if (volume.channels == 1) {
        for (Index p = 0; p < volume.pixels; ++p)
            if (!(std::abs(features[p]) <= T(1 << 20)))
                return false;
        return run_slic(features, volume, options, output);
    }
    std::unique_ptr<T[]> planes(new T[image_size]);
    for (Index c = 0; c < volume.channels; ++c)
        for (Index p = 0; p < volume.pixels; ++p) {
            const T value = features[p * volume.channels + c];
            if (!(std::abs(value) <= T(1 << 20)))
                return false;
            planes[c * volume.pixels + p] = value;
        }
    return run_slic(planes.get(), volume, options, output);
}

template bool slic_into(const std::uint8_t*, std::size_t, const std::array<std::size_t, 4>&,
                        const SlicOptions&, std::ptrdiff_t*, std::size_t);
template bool slic_into(const float*, std::size_t, const std::array<std::size_t, 4>&,
                        const SlicOptions&, std::ptrdiff_t*, std::size_t);
template bool slic_into(const double*, std::size_t, const std::array<std::size_t, 4>&,
                        const SlicOptions&, std::ptrdiff_t*, std::size_t);
template bool slic_prepared_into(const float*, std::size_t, const std::array<std::size_t, 4>&,
                                 const SlicOptions&, std::ptrdiff_t*, std::size_t);
template bool slic_prepared_into(const double*, std::size_t, const std::array<std::size_t, 4>&,
                                 const SlicOptions&, std::ptrdiff_t*, std::size_t);

}
