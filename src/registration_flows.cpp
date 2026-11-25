#include "skimagexx/registration_flows.hpp"

#include "execution.hpp"
#include "resample_detail.hpp"
#include "skimagexx/adaptive.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace skimagexx {
namespace {

template<class T>
constexpr auto pixel_type = std::is_same_v<T, float> ? PixelType::float32 : PixelType::float64;

struct Shape {
    std::size_t height, width;

    std::size_t count() const { return height * width; }
    std::vector<std::size_t> axes() const { return {height, width}; }
};

template<class T>
void check_buffers(const T* reference, const T* moving, const T* flow, Shape shape) {
    if (!reference || !moving || !flow || shape.height < 2 || shape.width < 2)
        throw std::invalid_argument("optical flow requires nonnull images with both axes at least 2");
    constexpr auto limit = std::size_t(std::numeric_limits<std::ptrdiff_t>::max()) / (2 * sizeof(T));
    if (shape.height > limit / shape.width)
        throw std::overflow_error("optical flow dimensions overflow");
}

std::vector<GaussianPass> gaussian_passes(double sigma, GaussianBorder border) {
    const auto radius = static_cast<std::ptrdiff_t>(4 * sigma + 0.5);
    const auto coefficient = -0.5 / (sigma * sigma);
    std::vector<double> weights(2 * radius + 1);
    for (auto x = -radius; x <= radius; ++x) {
        const auto distance = double(x);
        weights[x + radius] = std::exp(coefficient * (distance * distance));
    }
    const auto sum = std::accumulate(weights.begin(), weights.end(), 0.0);
    for (auto& weight : weights) weight /= sum;
    return {{0, weights, border}, {1, std::move(weights), border}};
}

template<class T>
void prefilter_flow(const T* input, T* output, Shape shape) {
    const auto axes = shape.axes();
    const std::vector<std::size_t> window{3, 3};
    const std::vector<GaussianBorder> borders(2, GaussianBorder::reflect);
    for (std::size_t component = 0; component < 2; ++component) {
        const auto offset = component * shape.count();
        adaptive_filter_into(input + offset, output + offset, pixel_type<T>, axes,
                             window, borders, AdaptiveMethod::median);
    }
}

struct LinearPoint {
    std::size_t lo, hi;
    double low_weight, high_weight;
};

LinearPoint edge_point(double coordinate, std::size_t last) {
    // Extend the two taps independently to preserve interpolation arithmetic
    // for fractional coordinates outside of the image.
    double integral;
    const auto fraction = std::modf(coordinate, &integral);
    const double low_weight = 1 - (fraction < 0 ? fraction + 1 : fraction);
    const double high_weight = 1 - low_weight;
    if (coordinate < 0) return {0, 0, low_weight, high_weight};
    if (coordinate >= double(last)) return {last, last, low_weight, high_weight};
    const auto base = static_cast<std::size_t>(integral);
    return {base, base + 1, low_weight, high_weight};
}

template<class T>
void warp(const T* image, const T* flow, T* output, Shape shape,
          const std::pair<T, T>& range, int workers) {
    const auto count = shape.count();
    detail::partition(shape.height, workers, [&](auto begin, auto end, auto) {
        for (auto y = begin; y < end; ++y) {
            for (std::size_t x = 0; x < shape.width; ++x) {
                const auto index = y * shape.width + x;
                const auto dy = double(T(y) + flow[index]);
                const auto dx = double(T(x) + flow[count + index]);
                if (!std::isfinite(dy) || !std::isfinite(dx)) {
                    output[index] = std::numeric_limits<T>::quiet_NaN();
                    continue;
                }
                const auto v = edge_point(dy, shape.height - 1);
                const auto u = edge_point(dx, shape.width - 1);
                const auto* top = image + v.lo * shape.width;
                const auto* bottom = image + v.hi * shape.width;
                double value = 0;
                value += double(top[u.lo]) * v.low_weight * u.low_weight;
                value += double(top[u.hi]) * v.low_weight * u.high_weight;
                value += double(bottom[u.lo]) * v.high_weight * u.low_weight;
                value += double(bottom[u.hi]) * v.high_weight * u.high_weight;
                output[index] = std::clamp(T(value), range.first, range.second);
            }
        }
    });
}

template<class T, class Store>
void gradients(const T* image, Shape shape, int workers, const Store& store) {
    detail::partition(shape.height, workers, [&](auto begin, auto end, auto) {
        for (auto y = begin; y < end; ++y) {
            const auto* upper = image + (y ? y - 1 : y) * shape.width;
            const auto* lower = image + (y + 1 < shape.height ? y + 1 : y) * shape.width;
            const auto* row = image + y * shape.width;
            const auto factor = y && y + 1 < shape.height ? T(0.5) : T(1);
            const auto base = y * shape.width;
            const auto last = shape.width - 1;
            store(base, (lower[0] - upper[0]) * factor, row[1] - row[0]);
            for (std::size_t x = 1; x < last; ++x)
                store(base + x, (lower[x] - upper[x]) * factor, (row[x + 1] - row[x - 1]) * T(0.5));
            store(base + last, (lower[last] - upper[last]) * factor, row[last] - row[last - 1]);
        }
    });
}

template<class T>
bool solve_local_system(T a, T b, T d, T rhs_y, T rhs_x, T& y, T& x) {
    // NumPy uses double-precision LAPACK for both supported result dtypes.
    const bool swap = std::abs(b) > std::abs(a);
    const double pivot = swap ? b : a;
    if (pivot == 0) {
        y = x = 0;
        return true;
    }
    const double upper = swap ? d : b;
    const double lower = swap ? a : b;
    const double factor = lower / pivot;
    const double diagonal = (swap ? double(b) : double(d)) - factor * upper;
    const double determinant = std::abs(pivot * diagonal);
    if (T(determinant) < T(1e-14)) {
        y = x = 0;
        return true;
    }
    // An active system near the aperture problem can amplify last-bit
    // differences between LAPACK and this local solve at each warp.
    const double scale = std::max(std::abs(double(a)), std::abs(double(d)));
    if (!std::isfinite(determinant) || determinant < (1e-4 * scale) * scale)
        return false;
    const double first = swap ? rhs_x : rhs_y;
    const double second = swap ? rhs_y : rhs_x;
    const double solved_x = (second - factor * first) / diagonal;
    x = T(solved_x);
    y = T((first - upper * solved_x) / pivot);
    return std::isfinite(y) && std::isfinite(x);
}

template<class T>
bool solve_ilk(const T* reference, const T* moving, T* flow, Shape shape, int workers,
               int radius, int num_warp, bool gaussian, bool prefilter,
               const std::vector<GaussianPass>& passes) {
    if (!num_warp) return true;
    const auto count = shape.count();
    const auto axes = shape.axes();
    const std::vector<std::size_t> window(2, 2 * std::size_t(radius) + 1);
    const std::vector<GaussianBorder> borders(2, GaussianBorder::mirror);
    const auto extrema = std::minmax_element(moving, moving + count);
    const auto range = std::make_pair(*extrema.first, *extrema.second);
    std::vector<T> warped(count);
    std::vector<T> filtered(5 * count), unfiltered(5 * count);
    std::vector<T> median(prefilter ? 2 * count : 0);

    for (int step = 0; step < num_warp; ++step) {
        const auto* current = flow;
        if (prefilter) {
            prefilter_flow(flow, median.data(), shape);
            current = median.data();
        }
        warp(moving, current, warped.data(), shape, range, workers);
        gradients(warped.data(), shape, workers, [&](std::size_t i, T y, T x) {
            const T residual = ((y * current[i] + x * current[count + i]) + reference[i]) - warped[i];
            unfiltered[i] = y * y;
            unfiltered[count + i] = y * x;
            unfiltered[2 * count + i] = x * x;
            unfiltered[3 * count + i] = y * residual;
            unfiltered[4 * count + i] = x * residual;
        });
        for (std::size_t channel = 0; channel < 5; ++channel) {
            const auto* input = unfiltered.data() + channel * count;
            auto* output = filtered.data() + channel * count;
            if (gaussian)
                gaussian_nd_into(input, output, pixel_type<T>, pixel_type<T>, axes, passes);
            else
                adaptive_filter_into(input, output, pixel_type<T>, axes, window, borders, AdaptiveMethod::mean);
        }
        std::atomic<bool> stable{true};
        detail::partition(count, workers, [&](auto begin, auto end, auto) {
            for (auto i = begin; i < end; ++i) {
                if (!solve_local_system(filtered[i], filtered[count + i], filtered[2 * count + i],
                                        filtered[3 * count + i], filtered[4 * count + i], flow[i], flow[count + i])) {
                    stable.store(false, std::memory_order_relaxed);
                    return;
                }
            }
        });
        if (!stable.load(std::memory_order_relaxed)) return false;
    }
    return true;
}

template<class T>
void regularize(const T* auxiliary, T* current, T* vertical, T* horizontal,
                Shape shape, T factor, int workers) {
    constexpr T time_step = T(0.25);
    for (int iteration = 0; iteration < 2; ++iteration) {
        detail::partition(shape.height, workers, [&](auto begin, auto end, auto) {
            for (auto y = begin; y < end; ++y) {
                const auto base = y * shape.width;
                const bool bottom = y + 1 == shape.height;
                for (std::size_t x = 0; x < shape.width; ++x) {
                    const auto i = base + x;
                    const T dy = bottom ? T(0) : current[i + shape.width] - current[i];
                    const T dx = x + 1 == shape.width ? T(0) : current[i + 1] - current[i];
                    const T norm = std::sqrt(dy * dy + dx * dx) * factor + T(1);
                    vertical[i] = (vertical[i] - time_step * dy) / norm;
                    horizontal[i] = (horizontal[i] - time_step * dx) / norm;
                }
            }
        });
        detail::partition(shape.height, workers, [&](auto begin, auto end, auto) {
            for (auto y = begin; y < end; ++y) {
                const auto base = y * shape.width;
                for (std::size_t x = 0; x < shape.width; ++x) {
                    const auto i = base + x;
                    T divergence = -(vertical[i] + horizontal[i]);
                    if (y) divergence += vertical[i - shape.width];
                    if (x) divergence += horizontal[i - 1];
                    current[i] = auxiliary[i] + divergence;
                }
            }
        });
    }
}

template<class T>
void solve_tvl1(const T* reference, const T* moving, T* flow, Shape shape, int workers,
                T attachment, T regularization, int num_warp, int num_iter, double tolerance,
                bool prefilter) {
    if (!num_warp) return;
    const auto count = shape.count();
    const auto extrema = std::minmax_element(moving, moving + count);
    const auto range = std::make_pair(*extrema.first, *extrema.second);
    std::vector<T> warped(count), strength(count), rho0(count), gy(count), gx(count);
    std::vector<T> auxiliary(2 * count), previous(2 * count), projection(4 * count, T(0));

    for (int step = 0; step < num_warp; ++step) {
        if (prefilter) {
            std::copy_n(flow, 2 * count, previous.data());
            prefilter_flow(previous.data(), flow, shape);
        }
        warp(moving, flow, warped.data(), shape, range, workers);
        gradients(warped.data(), shape, workers, [&](std::size_t i, T y, T x) {
            gy[i] = y;
            gx[i] = x;
            const T norm = y * y + x * x;
            strength[i] = norm == 0 ? T(1) : norm;
            rho0[i] = (warped[i] - reference[i]) - (y * flow[i] + x * flow[count + i]);
        });
        for (int iter = 0; iter < num_iter; ++iter) {
            detail::partition(count, workers, [&](auto begin, auto end, auto) {
                for (auto i = begin; i < end; ++i) {
                    const T rho = rho0[i] + (gy[i] * flow[i] + gx[i] * flow[count + i]);
                    T delta_y, delta_x;
                    if (std::abs(rho) <= attachment * strength[i]) {
                        delta_y = (rho * gy[i]) / strength[i];
                        delta_x = (rho * gx[i]) / strength[i];
                    } else {
                        const T signed_attachment = std::isnan(rho) ? rho : std::copysign(attachment, rho);
                        delta_y = signed_attachment * gy[i];
                        delta_x = signed_attachment * gx[i];
                    }
                    auxiliary[i] = flow[i] - delta_y;
                    auxiliary[count + i] = flow[count + i] - delta_x;
                }
            });
            // Without the prefilter, stock's previous flow aliases the first
            // data update of this warp until regularization replaces it.
            if (!prefilter && iter == 0) std::copy(auxiliary.begin(), auxiliary.end(), previous.begin());
            std::copy(auxiliary.begin(), auxiliary.end(), flow);
            for (std::size_t component = 0; component < 2; ++component) {
                const auto offset = component * count;
                regularize(auxiliary.data() + offset, flow + offset,
                           projection.data() + 2 * offset, projection.data() + 2 * offset + count,
                           shape, regularization, workers);
            }
        }
        if (!num_iter && !prefilter) {
            // The in-place convergence subtraction uses the same buffer twice.
            for (std::size_t i = 0; i < 2 * count; ++i) flow[i] -= flow[i];
            continue;
        }
        double change = 0;
        for (std::size_t i = 0; i < 2 * count; ++i) {
            const T delta = previous[i] - flow[i];
            change += T(delta * delta);
        }
        if (change < tolerance) break;
    }
}

template<class T>
std::vector<T> reduce_image(const T* image, Shape input, Shape output,
                            const std::vector<GaussianPass>& passes) {
    std::vector<T> smoothed(input.count()), reduced(output.count());
    gaussian_nd_into(image, smoothed.data(), pixel_type<T>, pixel_type<T>, input.axes(), passes);
    detail::zoom_typed_into(smoothed.data(), reduced.data(), pixel_type<T>, input.axes(), output.axes(), {});
    const auto extrema = std::minmax_element(smoothed.begin(), smoothed.end());
    for (auto& value : reduced) value = std::clamp(value, *extrema.first, *extrema.second);
    return reduced;
}

template<class T>
void expand_flow(const T* source, T* target, Shape from, Shape to, int workers) {
    const auto y_ratio = double(from.height - 1) / double(to.height - 1);
    const auto x_ratio = double(from.width - 1) / double(to.width - 1);
    const auto y_scale = T(double(to.height) / double(from.height));
    const auto x_scale = T(double(to.width) / double(from.width));
    std::vector<std::size_t> columns(to.width);
    for (std::size_t x = 0; x < to.width; ++x)
        columns[x] = static_cast<std::size_t>(double(x) * x_ratio + 0.5);
    detail::partition(to.height, workers, [&](auto begin, auto end, auto) {
        for (auto y = begin; y < end; ++y) {
            const auto row = static_cast<std::size_t>(double(y) * y_ratio + 0.5);
            for (std::size_t x = 0; x < to.width; ++x) {
                const auto input = row * from.width + columns[x];
                const auto output = y * to.width + x;
                target[output] = source[input] * y_scale;
                target[to.count() + output] = source[from.count() + input] * x_scale;
            }
        }
    });
}

template<class T, class Solver>
bool coarse_to_fine(const T* reference, const T* moving, T* output, Shape shape, const Solver& solve) {
    struct Layer {
        Shape shape;
        std::vector<T> reference, moving;
    };
    std::vector<Layer> pyramid;
    const int workers = get_num_threads();
    auto current = shape;
    if (std::min(current.height, current.width) > 32) {
        const auto* previous_reference = reference;
        const auto* previous_moving = moving;
        const auto passes = gaussian_passes(2.0 / 3.0, GaussianBorder::reflect);
        for (int level = 1; level < 10 && std::min(current.height, current.width) > 32; ++level) {
            const Shape next{(current.height + 1) / 2, (current.width + 1) / 2};
            auto ref_reduced = reduce_image(previous_reference, current, next, passes);
            auto moving_reduced = reduce_image(previous_moving, current, next, passes);
            pyramid.push_back({next, std::move(ref_reduced), std::move(moving_reduced)});
            previous_reference = pyramid.back().reference.data();
            previous_moving = pyramid.back().moving.data();
            current = next;
        }
    }

    const auto layer_workers = [&](Shape layer) { return layer.count() >= detail::parallel_min ? workers : 1; };
    if (pyramid.empty()) {
        std::fill_n(output, 2 * shape.count(), T(0));
        return solve(reference, moving, output, shape, layer_workers(shape));
    }
    std::vector<T> flow(2 * current.count(), T(0)), next_flow;
    for (auto level = pyramid.rbegin(); level != pyramid.rend(); ++level) {
        const auto threads = layer_workers(level->shape);
        if (level->shape.height != current.height || level->shape.width != current.width) {
            next_flow.resize(2 * level->shape.count());
            expand_flow(flow.data(), next_flow.data(), current, level->shape, threads);
            flow.swap(next_flow);
        }
        if (!solve(level->reference.data(), level->moving.data(), flow.data(), level->shape, threads))
            return false;
        current = level->shape;
    }
    expand_flow(flow.data(), output, current, shape, layer_workers(shape));
    return solve(reference, moving, output, shape, layer_workers(shape));
}

} // namespace

template<class T>
bool optical_flow_ilk_into(const T* reference, const T* moving, T* flow, std::size_t height,
                           std::size_t width, int radius, int num_warp, bool gaussian, bool prefilter) {
    const Shape shape{height, width};
    check_buffers(reference, moving, flow, shape);
    if (radius < 0 || num_warp < 0 || radius > (std::numeric_limits<int>::max() - 1) / 2)
        throw std::invalid_argument("invalid Lucas-Kanade radius or warp count");
    const auto passes = gaussian && num_warp ? gaussian_passes((2.0 * radius + 1) / 4, GaussianBorder::mirror)
                                             : std::vector<GaussianPass>{};
    return coarse_to_fine(reference, moving, flow, shape,
                   [&](const T* ref, const T* mov, T* level, Shape dims, int workers) {
        return solve_ilk(ref, mov, level, dims, workers, radius, num_warp, gaussian, prefilter, passes);
    });
}

template<class T>
void optical_flow_tvl1_into(const T* reference, const T* moving, T* flow, std::size_t height,
                            std::size_t width, double attachment, double tightness, int num_warp,
                            int num_iter, double tol, bool prefilter) {
    const Shape shape{height, width};
    check_buffers(reference, moving, flow, shape);
    if (num_warp < 0 || num_iter < 0 || !std::isfinite(attachment) || attachment <= 0 ||
        !std::isfinite(tightness) || tightness <= 0 || !std::isfinite(tol) || tol < 0)
        throw std::invalid_argument("invalid TV-L1 solver parameters");
    coarse_to_fine(reference, moving, flow, shape,
                   [&](const T* ref, const T* mov, T* level, Shape dims, int workers) {
        solve_tvl1(ref, mov, level, dims, workers, T(attachment * tightness), T(0.25 / tightness),
                   num_warp, num_iter, tol * double(dims.count()), prefilter);
        return true;
    });
}

template bool optical_flow_ilk_into(const float*, const float*, float*, std::size_t, std::size_t,
                                   int, int, bool, bool);
template bool optical_flow_ilk_into(const double*, const double*, double*, std::size_t, std::size_t,
                                   int, int, bool, bool);
template void optical_flow_tvl1_into(const float*, const float*, float*, std::size_t, std::size_t,
                                    double, double, int, int, double, bool);
template void optical_flow_tvl1_into(const double*, const double*, double*, std::size_t, std::size_t,
                                    double, double, int, int, double, bool);

} // namespace skimagexx
