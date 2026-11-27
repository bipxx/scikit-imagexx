#include "skimagexx/registration_flows.hpp"

#include "execution.hpp"
#include "resample_detail.hpp"
#include "skimagexx/adaptive.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <limits>
#include <memory>
#include <new>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef SKIMAGEXX_AVX2
#include <immintrin.h>
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4752)
#define FLOW_AVX2 __declspec(noinline)
#else
#define FLOW_AVX2 __attribute__((target("avx2,no-fma"), noinline))
#endif
#endif

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
    if (coordinate >= 0 && coordinate < double(last)) {
        const auto base = static_cast<std::size_t>(coordinate);
        const double low = 1 - (coordinate - double(base));
        return {base, base + 1, low, 1 - low};
    }
    // Extend the two taps independently to preserve interpolation arithmetic
    // for fractional coordinates outside of the image.
    double integral;
    const auto fraction = std::modf(coordinate, &integral);
    const double low_weight = 1 - (fraction < 0 ? fraction + 1 : fraction);
    const double high_weight = 1 - low_weight;
    if (coordinate < 0) return {0, 0, low_weight, high_weight};
    return {last, last, low_weight, high_weight};
}

template<class T>
void warp(const T* image, const T* flow, T* output, Shape shape,
          const std::pair<T, T>& range, int workers) {
    const auto count = shape.count();
    detail::partition(shape.height, workers, [&](auto begin, auto end, auto) {
        for (auto y = begin; y < end; ++y) {
            const auto* vertical = flow + y * shape.width;
            const auto* horizontal = vertical + count;
            auto* row = output + y * shape.width;
            const auto row_coordinate = T(y);
            for (std::size_t x = 0; x < shape.width; ++x) {
                const auto dy = double(row_coordinate + vertical[x]);
                const auto dx = double(T(x) + horizontal[x]);
                if (!std::isfinite(dy) || !std::isfinite(dx)) {
                    row[x] = std::numeric_limits<T>::quiet_NaN();
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
                row[x] = std::clamp(T(value), range.first, range.second);
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
struct IlkScratch {
    std::vector<T> warped, median, unfiltered, filtered;
    std::vector<std::size_t> rows, columns;
};

void mirror_indices(std::vector<std::size_t>& indices, std::size_t length, int radius) {
    indices.resize(length + 2 * std::size_t(radius));
    const auto period = 2 * (length - 1);
    const auto shift = std::size_t(radius) % period;
    for (std::size_t i = 0; i < indices.size(); ++i) {
        const auto x = (i + period - shift) % period;
        indices[i] = x < length ? x : period - x;
    }
}

template<class T>
bool mean_tensor_solve(const T* input, T* vertical, T* flow, Shape shape, int radius,
                       const std::vector<std::size_t>& rows,
                       const std::vector<std::size_t>& columns, int workers) {
    const auto count = shape.count();
    const auto window = 2 * std::size_t(radius) + 1;
    const double divisor = double(window);
    // Match the separable uniform filter's rolling double sums and its cast
    // between axes. Keep all five tensor terms together for each local solve.
    constexpr std::size_t max_block_width = 32;
    using StripSums = std::array<double, 5 * max_block_width>;
    const auto block_width = std::min(max_block_width, 1 + (shape.width - 1) / std::size_t(workers));
    const auto blocks = 1 + (shape.width - 1) / block_width;
    detail::partition(blocks, workers, [&](auto begin, auto end, auto) {
        for (auto block = begin; block < end; ++block) {
            const auto x = block * block_width;
            const auto terms = 5 * std::min(block_width, shape.width - x);
            const auto row_at = [&](std::size_t k) {
                return input + 5 * (rows[k] * shape.width + x);
            };
            // Advance whole strips to load and store adjacent tensor terms.
            StripSums total{};
            for (std::size_t k = 0; k < window; ++k) {
                const auto* row = row_at(k);
                for (std::size_t c = 0; c < terms; ++c) total[c] += row[c];
            }
            auto* dst = vertical + 5 * x;
            for (std::size_t c = 0; c < terms; ++c) dst[c] = T(total[c] / divisor);
            for (std::size_t y = 1; y < shape.height; ++y) {
                const auto* head = row_at(y + window - 1);
                const auto* tail = row_at(y - 1);
                dst += 5 * shape.width;
                for (std::size_t c = 0; c < terms; ++c) {
                    total[c] += double(head[c]) - double(tail[c]);
                    dst[c] = T(total[c] / divisor);
                }
            }
        }
    });
    std::atomic<bool> stable{true};
    detail::partition(shape.height, workers, [&](auto begin, auto end, auto) {
        for (auto y = begin; y < end; ++y) {
            const auto base = y * shape.width;
            const auto* row = vertical + 5 * base;
            std::array<double, 5> total{};
            for (std::size_t k = 0; k < window; ++k)
                for (std::size_t c = 0; c < 5; ++c) total[c] += row[5 * columns[k] + c];
            for (std::size_t x = 0; x < shape.width; ++x) {
                if (x) {
                    const auto* head = row + 5 * columns[x + window - 1];
                    const auto* tail = row + 5 * columns[x - 1];
                    for (std::size_t c = 0; c < 5; ++c) total[c] += double(head[c]) - double(tail[c]);
                }
                if (!solve_local_system(T(total[0] / divisor), T(total[1] / divisor),
                                        T(total[2] / divisor), T(total[3] / divisor),
                                        T(total[4] / divisor), flow[base + x], flow[count + base + x])) {
                    stable.store(false, std::memory_order_relaxed);
                    return;
                }
            }
        }
    });
    return stable.load(std::memory_order_relaxed);
}

template<class T>
bool solve_ilk(const T* reference, const T* moving, T* flow, Shape shape, int workers,
               int radius, int num_warp, bool gaussian, bool prefilter,
               const std::vector<GaussianPass>& passes, IlkScratch<T>& scratch) {
    if (!num_warp) return true;
    const auto count = shape.count();
    const auto axes = shape.axes();
    const auto extrema = std::minmax_element(moving, moving + count);
    const auto range = std::make_pair(*extrema.first, *extrema.second);
    auto* warped = scratch.warped.data();
    auto* unfiltered = scratch.unfiltered.data();
    auto* filtered = scratch.filtered.data();
    if (!gaussian && radius) {
        mirror_indices(scratch.rows, shape.height, radius);
        mirror_indices(scratch.columns, shape.width, radius);
    }

    for (int step = 0; step < num_warp; ++step) {
        const auto* current = flow;
        if (prefilter) {
            prefilter_flow(flow, scratch.median.data(), shape);
            current = scratch.median.data();
        }
        warp(moving, current, warped, shape, range, workers);
        std::atomic<bool> stable{true};
        gradients(warped, shape, workers, [&](std::size_t i, T y, T x) {
            const T residual = ((y * current[i] + x * current[count + i]) + reference[i]) - warped[i];
            if (!gaussian && !radius) {
                if (!solve_local_system(y * y, y * x, x * x, y * residual, x * residual,
                                        flow[i], flow[count + i]))
                    stable.store(false, std::memory_order_relaxed);
                return;
            }
            auto* tensor = unfiltered + (gaussian ? i : 5 * i);
            const auto stride = gaussian ? count : 1;
            tensor[0] = y * y;
            tensor[stride] = y * x;
            tensor[2 * stride] = x * x;
            tensor[3 * stride] = y * residual;
            tensor[4 * stride] = x * residual;
        });
        if (!gaussian) {
            if (!stable.load(std::memory_order_relaxed) ||
                (radius && !mean_tensor_solve(unfiltered, filtered, flow, shape, radius,
                                              scratch.rows, scratch.columns, workers)))
                return false;
            continue;
        }
        for (std::size_t channel = 0; channel < 5; ++channel) {
            gaussian_nd_into(unfiltered + channel * count, filtered + channel * count,
                             pixel_type<T>, pixel_type<T>, axes, passes);
        }
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
void tvl1_data_pixel(T flow_y, T flow_x, T y, T x, T rho0, T strength, T cutoff,
                     T attachment, T& auxiliary_y, T& auxiliary_x) {
    const T rho = rho0 + (y * flow_y + x * flow_x);
    T delta_y, delta_x;
    if (std::abs(rho) <= cutoff) {
        delta_y = (rho * y) / strength;
        delta_x = (rho * x) / strength;
    } else {
        const T signed_attachment = std::isnan(rho) ? rho : std::copysign(attachment, rho);
        delta_y = signed_attachment * y;
        delta_x = signed_attachment * x;
    }
    auxiliary_y = flow_y - delta_y;
    auxiliary_x = flow_x - delta_x;
}

#ifdef SKIMAGEXX_AVX2
template<bool fused>
FLOW_AVX2 std::size_t regularize_float_row(
    const float* prior, const float* auxiliary, float* current, float* vertical, float* horizontal,
    Shape shape, std::size_t y, float factor) {
    const auto width = shape.width;
    const auto base = y * width;
    const bool bottom = y + 1 == shape.height;
    const auto zero = _mm256_setzero_ps();
    const auto sign = _mm256_set1_ps(-0.0f);
    const auto step = _mm256_set1_ps(0.25f);
    const auto scale = _mm256_set1_ps(factor);
    const auto one = _mm256_set1_ps(1);
    const auto shift = _mm256_setr_epi32(0, 0, 1, 2, 3, 4, 5, 6);
    const auto last_lane = _mm256_set1_epi32(7);
    auto left = fused ? _mm256_set1_ps(horizontal[base]) : zero;
    std::size_t x = 1;
    for (; x + 8 < width; x += 8) {
        const auto i = base + x;
        const auto center = _mm256_loadu_ps(prior + i);
        const auto dy = bottom ? zero : _mm256_sub_ps(_mm256_loadu_ps(prior + i + width), center);
        const auto dx = _mm256_sub_ps(_mm256_loadu_ps(prior + i + 1), center);
        const auto square = _mm256_add_ps(_mm256_mul_ps(dy, dy), _mm256_mul_ps(dx, dx));
        const auto norm = _mm256_add_ps(_mm256_mul_ps(_mm256_sqrt_ps(square), scale), one);
        const auto v = _mm256_div_ps(_mm256_sub_ps(_mm256_loadu_ps(vertical + i), _mm256_mul_ps(step, dy)), norm);
        const auto h = _mm256_div_ps(_mm256_sub_ps(_mm256_loadu_ps(horizontal + i), _mm256_mul_ps(step, dx)), norm);
        _mm256_storeu_ps(vertical + i, v);
        _mm256_storeu_ps(horizontal + i, h);
        if constexpr (fused) {
            auto divergence = _mm256_xor_ps(_mm256_add_ps(v, h), sign);
            if (y != 0) divergence = _mm256_add_ps(divergence, _mm256_loadu_ps(vertical + i - width));
            // Carry the preceding horizontal dual across vectors in a register.
            const auto previous = _mm256_blend_ps(_mm256_permutevar8x32_ps(h, shift), left, 1);
            divergence = _mm256_add_ps(divergence, previous);
            left = _mm256_permutevar8x32_ps(h, last_lane);
            _mm256_storeu_ps(current + i, _mm256_add_ps(_mm256_loadu_ps(auxiliary + i), divergence));
        }
    }
    return x;
}

FLOW_AVX2 std::size_t tvl1_data_float(const float* flow_y, const float* flow_x,
                                    const float* gy, const float* gx, const float* rho0,
                                    const float* strength, const float* cutoff, float attachment,
                                    float* auxiliary_y, float* auxiliary_x,
                                    std::size_t begin, std::size_t end) {
    const auto sign_bit = _mm256_set1_ps(-0.0f);
    const auto limit = _mm256_set1_ps(attachment);
    auto i = begin;
    for (; i + 8 <= end; i += 8) {
        const auto y = _mm256_loadu_ps(gy + i), x = _mm256_loadu_ps(gx + i);
        const auto vy = _mm256_loadu_ps(flow_y + i), vx = _mm256_loadu_ps(flow_x + i);
        const auto dot = _mm256_add_ps(_mm256_mul_ps(y, vy), _mm256_mul_ps(x, vx));
        const auto rho = _mm256_add_ps(_mm256_loadu_ps(rho0 + i), dot);
        const auto sign = _mm256_and_ps(rho, sign_bit);
        const auto magnitude = _mm256_xor_ps(rho, sign);
        const auto mask = _mm256_cmp_ps(magnitude, _mm256_loadu_ps(cutoff + i), _CMP_LE_OQ);
        const auto active = _mm256_movemask_ps(mask);
        __m256 dy, dx;
        if (active == 255) {
            const auto norm = _mm256_loadu_ps(strength + i);
            dy = _mm256_div_ps(_mm256_mul_ps(rho, y), norm);
            dx = _mm256_div_ps(_mm256_mul_ps(rho, x), norm);
        } else if (active == 0) {
            const auto signed_limit = _mm256_or_ps(limit, sign);
            const auto nan = _mm256_cmp_ps(rho, rho, _CMP_UNORD_Q);
            const auto clipped = _mm256_blendv_ps(signed_limit, rho, nan);
            dy = _mm256_mul_ps(clipped, y);
            dx = _mm256_mul_ps(clipped, x);
        } else {
            // Evaluate only the selected expression in each mixed lane.
            for (auto j = i; j < i + 8; ++j)
                tvl1_data_pixel(flow_y[j], flow_x[j], gy[j], gx[j], rho0[j], strength[j], cutoff[j],
                                attachment, auxiliary_y[j], auxiliary_x[j]);
            continue;
        }
        _mm256_storeu_ps(auxiliary_y + i, _mm256_sub_ps(vy, dy));
        _mm256_storeu_ps(auxiliary_x + i, _mm256_sub_ps(vx, dx));
    }
    return i;
}
#ifdef _MSC_VER
#pragma warning(pop)
#endif
#undef FLOW_AVX2
#endif

template<class T>
void regularize(const T* auxiliary, T* current, T* vertical, T* horizontal,
                Shape shape, T factor, int workers, [[maybe_unused]] bool vectorized) {
    constexpr T time_step = T(0.25);
    const auto width = shape.width;
    const auto last = width - 1;
    const auto primal = [&](std::size_t i, std::size_t y, std::size_t x) {
        T divergence = -(vertical[i] + horizontal[i]);
        if (y != 0) divergence += vertical[i - width];
        if (x != 0) divergence += horizontal[i - 1];
        current[i] = auxiliary[i] + divergence;
    };
    const auto dual_row = [&](const T* prior, std::size_t y, auto fuse) {
        const auto base = y * width;
        const bool bottom = y + 1 == shape.height;
        const auto update = [&](std::size_t x, T dx) {
            const auto i = base + x;
            const T dy = bottom ? T(0) : prior[i + width] - prior[i];
            const T norm = std::sqrt(dy * dy + dx * dx) * factor + T(1);
            vertical[i] = (vertical[i] - time_step * dy) / norm;
            horizontal[i] = (horizontal[i] - time_step * dx) / norm;
            if constexpr (decltype(fuse)::value) primal(i, y, x);
        };
        update(0, prior[base + 1] - prior[base]);
        std::size_t x = 1;
#ifdef SKIMAGEXX_AVX2
        if constexpr (std::is_same_v<T, float>) {
            if (vectorized && width > 9)
                x = regularize_float_row<decltype(fuse)::value>(
                    prior, auxiliary, current, vertical, horizontal, shape, y, factor);
        }
#endif
        for (; x < last; ++x)
            update(x, prior[base + x + 1] - prior[base + x]);
        update(last, T(0));
    };

    if (workers == 1) {
        // Once the first update has written row y, the second can consume
        // row y-1. The first update still needs the first-pass duals on row y.
        dual_row(auxiliary, 0, std::true_type{});
        for (std::size_t y = 1; y < shape.height; ++y) {
            dual_row(auxiliary, y, std::true_type{});
            dual_row(current, y - 1, std::true_type{});
        }
        dual_row(current, shape.height - 1, std::true_type{});
        return;
    }

    for (int iteration = 0; iteration < 2; ++iteration) {
        const auto* prior = iteration == 0 ? auxiliary : current;
        detail::partition(shape.height, workers, [&](auto begin, auto end, auto) {
            for (auto y = begin; y < end; ++y) dual_row(prior, y, std::false_type{});
        });
        detail::partition(shape.height, workers, [&](auto begin, auto end, auto) {
            for (auto y = begin; y < end; ++y)
                for (std::size_t x = 0; x < width; ++x) primal(y * width + x, y, x);
        });
    }
}

template<class T>
struct Tvl1Scratch {
    // Six image planes, two two-component flows, and four dual planes.
    static constexpr std::size_t planes = 14;
    std::unique_ptr<T[]> storage;

    explicit Tvl1Scratch(std::size_t count) {
        if (count > std::numeric_limits<std::size_t>::max() / (planes * sizeof(T)))
            throw std::bad_alloc();
        if (count) storage.reset(new T[planes * count]);
    }
};

template<class T>
void solve_tvl1(const T* reference, const T* moving, T* flow, Shape shape, int workers,
                T attachment, T regularization, int num_warp, int num_iter, double tolerance,
                bool prefilter, Tvl1Scratch<T>& scratch) {
    if (!num_warp) return;
    const auto count = shape.count();
    const auto extrema = std::minmax_element(moving, moving + count);
    const auto range = std::make_pair(*extrema.first, *extrema.second);
    auto* warped = scratch.storage.get();
    auto* strength = warped + count;
    auto* cutoff = strength + count;
    auto* rho0 = cutoff + count;
    auto* gy = rho0 + count;
    auto* gx = gy + count;
    auto* auxiliary = gx + count;
    auto* previous = auxiliary + 2 * count;
    auto* projection = previous + 2 * count;
    std::fill_n(projection, 4 * count, T(0));
    const bool vectorized = std::is_same_v<T, float> && std::isfinite(regularization) && has_avx2();

    for (int step = 0; step < num_warp; ++step) {
        if (prefilter) {
            std::copy_n(flow, 2 * count, previous);
            prefilter_flow(previous, flow, shape);
        }
        warp(moving, flow, warped, shape, range, workers);
        gradients(warped, shape, workers, [&](std::size_t i, T y, T x) {
            gy[i] = y;
            gx[i] = x;
            const T norm = y * y + x * x;
            strength[i] = norm == 0 ? T(1) : norm;
            if (num_iter != 0) cutoff[i] = attachment * strength[i];
            rho0[i] = (warped[i] - reference[i]) - (y * flow[i] + x * flow[count + i]);
        });
        for (int iter = 0; iter < num_iter; ++iter) {
            // Stock's previous flow aliases the first data update of this
            // warp when prefiltering is disabled.
            const bool save_previous = !prefilter && iter == 0;
            detail::partition(count, workers, [&](auto begin, auto end, auto) {
                auto i = begin;
#ifdef SKIMAGEXX_AVX2
                if constexpr (std::is_same_v<T, float>) {
                    if (vectorized && std::isfinite(attachment) && end - begin >= 8) {
                        i = tvl1_data_float(flow, flow + count, gy, gx, rho0, strength, cutoff, attachment,
                                            auxiliary, auxiliary + count, begin, end);
                    }
                }
#endif
                for (; i < end; ++i)
                    tvl1_data_pixel(flow[i], flow[count + i], gy[i], gx[i], rho0[i], strength[i], cutoff[i],
                                    attachment, auxiliary[i], auxiliary[count + i]);
                if (save_previous) {
                    std::copy(auxiliary + begin, auxiliary + end, previous + begin);
                    std::copy(auxiliary + count + begin, auxiliary + count + end, previous + count + begin);
                }
            });
            for (std::size_t component = 0; component < 2; ++component) {
                const auto offset = component * count;
                regularize(auxiliary + offset, flow + offset,
                           projection + 2 * offset, projection + 2 * offset + count,
                           shape, regularization, workers, vectorized);
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
                            const std::vector<GaussianPass>& passes, std::vector<T>& smoothed) {
    std::vector<T> reduced(output.count());
    const auto input_axes = input.axes();
    gaussian_nd_into(image, smoothed.data(), pixel_type<T>, pixel_type<T>, input_axes, passes);
    detail::zoom_typed_into(smoothed.data(), reduced.data(), pixel_type<T>, input_axes, output.axes(), {});
    const auto extrema = std::minmax_element(smoothed.data(), smoothed.data() + input.count());
    for (auto& value : reduced) value = std::clamp(value, *extrema.first, *extrema.second);
    return reduced;
}

template<class T>
void expand_flow(const T* source, T* target, Shape from, Shape to, int workers,
                 std::vector<std::size_t>& columns) {
    const auto y_ratio = double(from.height - 1) / double(to.height - 1);
    const auto x_ratio = double(from.width - 1) / double(to.width - 1);
    const auto y_scale = T(double(to.height) / double(from.height));
    const auto x_scale = T(double(to.width) / double(from.width));
    columns.resize(to.width);
    for (std::size_t x = 0; x < to.width; ++x)
        columns[x] = static_cast<std::size_t>(double(x) * x_ratio + 0.5);
    const auto* source_horizontal = source + from.count();
    auto* target_horizontal = target + to.count();
    detail::partition(to.height, workers, [&](auto begin, auto end, auto) {
        for (auto y = begin; y < end; ++y) {
            const auto row = static_cast<std::size_t>(double(y) * y_ratio + 0.5);
            const auto* input_y = source + row * from.width;
            const auto* input_x = source_horizontal + row * from.width;
            auto* output_y = target + y * to.width;
            auto* output_x = target_horizontal + y * to.width;
            for (std::size_t x = 0; x < to.width; ++x) {
                const auto column = columns[x];
                output_y[x] = input_y[column] * y_scale;
                output_x[x] = input_x[column] * x_scale;
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
        std::vector<T> smoothed(current.count());
        for (int level = 1; level < 10 && std::min(current.height, current.width) > 32; ++level) {
            const Shape next{(current.height + 1) / 2, (current.width + 1) / 2};
            auto ref_reduced = reduce_image(previous_reference, current, next, passes, smoothed);
            auto moving_reduced = reduce_image(previous_moving, current, next, passes, smoothed);
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
    std::vector<std::size_t> columns;
    columns.reserve(shape.width);
    // Intermediate flow alternates between two buffers until the final
    // expansion can write directly to the caller's output.
    flow.reserve(2 * pyramid.front().shape.count());
    next_flow.reserve(2 * pyramid.front().shape.count());
    for (auto level = pyramid.rbegin(); level != pyramid.rend(); ++level) {
        const auto threads = layer_workers(level->shape);
        if (level->shape.height != current.height || level->shape.width != current.width) {
            next_flow.resize(2 * level->shape.count());
            expand_flow(flow.data(), next_flow.data(), current, level->shape, threads, columns);
            flow.swap(next_flow);
        }
        if (!solve(level->reference.data(), level->moving.data(), flow.data(), level->shape, threads))
            return false;
        current = level->shape;
    }
    expand_flow(flow.data(), output, current, shape, layer_workers(shape), columns);
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
    IlkScratch<T> scratch;
    if (num_warp) {
        const auto count = shape.count();
        scratch.warped.resize(count);
        if (prefilter) scratch.median.resize(2 * count);
        if (gaussian || radius) {
            scratch.unfiltered.resize(5 * count);
            scratch.filtered.resize(5 * count);
        }
    }
    return coarse_to_fine(reference, moving, flow, shape,
                   [&](const T* ref, const T* mov, T* level, Shape dims, int workers) {
        return solve_ilk(ref, mov, level, dims, workers, radius, num_warp, gaussian, prefilter, passes, scratch);
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
    Tvl1Scratch<T> scratch(num_warp ? shape.count() : 0);
    coarse_to_fine(reference, moving, flow, shape,
                   [&](const T* ref, const T* mov, T* level, Shape dims, int workers) {
        solve_tvl1(ref, mov, level, dims, workers, T(attachment * tightness), T(0.25 / tightness),
                   num_warp, num_iter, tol * double(dims.count()), prefilter, scratch);
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
