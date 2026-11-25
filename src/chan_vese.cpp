#include "skimagexx/chan_vese.hpp"
#include "skimagexx/tv_reductions.hpp"
#include "execution.hpp"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
#include <xmmintrin.h>
#endif

namespace skimagexx {
namespace {

class FloatingEnvironment {
public:
    FloatingEnvironment() {
        if (std::fegetround() != FE_TONEAREST)
            throw std::domain_error("chan_vese requires nearest rounding");
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
        if (_mm_getcsr() & ((1u << 15) | (1u << 6)))
            throw std::domain_error("chan_vese requires gradual underflow");
#endif
        if (std::feholdexcept(&saved_))
            throw std::domain_error("chan_vese cannot capture the floating environment");
    }

    ~FloatingEnvironment() { std::fesetenv(&saved_); }

private:
    std::fenv_t saved_;
};

bool in_range(double value, double limit) {
    return 0 <= value && value <= limit;
}

template<class T>
T sum(const std::vector<T>& values, TvReductionPlan plan) {
    constexpr auto type = std::is_same_v<T, float> ? TvReductionType::float32 : TvReductionType::float64;
    const auto bits = tv_sum_contiguous({values.data(), values.size(), type}, plan).bits;
    T result;
    if constexpr (std::is_same_v<T, float>) {
        const auto float_bits = static_cast<std::uint32_t>(bits);
        std::memcpy(&result, &float_bits, sizeof(result));
    } else {
        std::memcpy(&result, &bits, sizeof(result));
    }
    return result;
}

template<class T>
std::vector<T> normalize(const T* image, const T* initial, std::size_t count) {
    constexpr T image_limit = std::numeric_limits<T>::max() / T(4);
    T low = image[0];
    T high = image[0];
    for (std::size_t i = 0; i < count; ++i) {
        if (!(std::abs(image[i]) <= image_limit) || !(std::abs(initial[i]) <= T(1e6)))
            throw std::domain_error("chan_vese requires finite bounded image and level set");
        low = std::min(low, image[i]);
        high = std::max(high, image[i]);
    }
    std::vector<T> output(count);
    if (low != high) {
        const T range = high - low;
        for (std::size_t i = 0; i < count; ++i)
            output[i] = (image[i] - low) / range;
    }
    return output;
}

template<class T>
constexpr T square(T value) {
    return value * value;
}

template<class T>
T delta(T value) {
    return T(1) / (T(1) + square(value));
}

template<class T>
std::pair<T, T> binary_means(const std::vector<T>& image, const std::vector<T>& level_set,
                             std::vector<T>& inside, std::vector<T>& outside, TvReductionPlan plan) {
    std::size_t positives = 0;
    for (std::size_t i = 0; i < image.size(); ++i) {
        const bool selected = level_set[i] > T(0);
        positives += static_cast<std::size_t>(selected);
        inside[i] = selected ? image[i] : T(0);
        outside[i] = selected ? T(0) : image[i];
    }
    const auto negatives = image.size() - positives;
    return {positives ? sum(inside, plan) / static_cast<T>(positives) : T(0),
            negatives ? sum(outside, plan) / static_cast<T>(negatives) : T(0)};
}

template<class T, class Op>
void visit_neighbors(const std::vector<T>& level_set, std::size_t height, std::size_t width,
                     int workers, Op operation) {
    detail::partition(height, workers, [&](std::size_t begin, std::size_t end, std::size_t) {
        for (std::size_t row = begin; row < end; ++row) {
            const auto center = row * width;
            const auto up = row == 0 ? center : center - width;
            const auto down = row + 1 == height ? center : center + width;
            for (std::size_t col = 0; col < width; ++col) {
                const auto i = center + col;
                const auto left = col == 0 ? i : i - 1;
                const auto right = col + 1 == width ? i : i + 1;
                operation(i, level_set[i], level_set[left], level_set[right],
                          level_set[up + col], level_set[down + col]);
            }
        }
    });
}

template<class T>
T energy(const std::vector<T>& image, const std::vector<T>& level_set, std::size_t height,
         std::size_t width, T mu, T lambda1, T lambda2, int workers, TvReductionPlan plan,
         std::vector<T>& h, std::vector<T>& hinv, std::vector<T>& inside, std::vector<T>& outside) {
    constexpr T scale = T(0.636619772367581343075535053490057448);
    for (std::size_t i = 0; i < image.size(); ++i) {
        h[i] = T(0.5) * (T(1) + scale * std::atan(level_set[i]));
        hinv[i] = T(1) - h[i];
        inside[i] = image[i] * h[i];
        outside[i] = image[i] * hinv[i];
    }
    const auto inside_count = sum(h, plan);
    const auto outside_count = sum(hinv, plan);
    const auto c1 = inside_count != T(0) ? sum(inside, plan) / inside_count : T(0);
    const auto c2 = outside_count != T(0) ? sum(outside, plan) / outside_count : T(0);
    visit_neighbors(level_set, height, width, workers, [&](std::size_t i, T center, T left, T right, T up, T down) {
        inside[i] = lambda1 * square(image[i] - c1) * h[i] + lambda2 * square(image[i] - c2) * hinv[i];
        const T fx = (right - left) / T(2);
        const T fy = (down - up) / T(2);
        outside[i] = mu * delta(center) * std::sqrt(square(fx) + square(fy));
    });
    return sum(inside, plan) + sum(outside, plan);
}

template<class T>
void evolve(const std::vector<T>& image, const std::vector<T>& level_set, std::size_t height,
            std::size_t width, T mu, T lambda1, T lambda2, T dt, T mu_dt, int workers, T c1, T c2,
            std::vector<T>& next, std::vector<T>& difference) {
    constexpr T eta = T(1e-16);
    visit_neighbors(level_set, height, width, workers, [&](std::size_t i, T center, T left, T right, T up, T down) {
        const auto dx = (right - left) / T(2);
        const auto dy = (down - up) / T(2);
        const auto c_right = T(1) / std::sqrt(eta + square(right - center) + square(dy));
        const auto c_left = T(1) / std::sqrt(eta + square(center - left) + square(dy));
        const auto c_down = T(1) / std::sqrt(eta + square(dx) + square(down - center));
        const auto c_up = T(1) / std::sqrt(eta + square(dx) + square(center - up));
        const T curvature = right * c_right + left * c_left + down * c_down + up * c_up;
        const T region = -lambda1 * square(image[i] - c1) + lambda2 * square(image[i] - c2);
        const T d = delta(center);
        next[i] = (center + (dt * d) * (mu * curvature + region)) /
                  (T(1) + mu_dt * d * (c_right + c_left + c_down + c_up));
        difference[i] = square(next[i] - center);
    });
}

}  // namespace

template<class T>
ChanVeseResult<T> chan_vese(const T* image, const T* initial_level_set, std::size_t count,
                           std::size_t height, std::size_t width, const ChanVeseOptions& options) {
    if (!image || !initial_level_set || reinterpret_cast<std::uintptr_t>(image) % alignof(T) ||
        reinterpret_cast<std::uintptr_t>(initial_level_set) % alignof(T) || !height || !width ||
        height > chan_vese_max_pixels / width || count != height * width)
        throw std::invalid_argument("chan_vese requires matching aligned nonempty image and level set spans");
    if (!in_range(options.mu, 16) || !in_range(options.lambda1, 16) || !in_range(options.lambda2, 16) ||
        !in_range(options.dt, 16) || !in_range(options.tol, 2) || options.max_num_iter > chan_vese_max_iterations)
        throw std::invalid_argument("chan_vese options are outside the supported range");

    const FloatingEnvironment environment;
    const auto normalized = normalize(image, initial_level_set, count);
    std::vector<T> level_set(initial_level_set, initial_level_set + count);
    const auto max_iterations = options.max_num_iter;
    ChanVeseResult<T> result;
    const auto workers = static_cast<int>(std::min(height, static_cast<std::size_t>(detail::active_workers(count))));
    const TvReductionPlan plan{
        options.reduction_buffer ? TvReductionProfile::numpy_1244_buffered
                                 : TvReductionProfile::numpy_254_aligned_contiguous,
        options.reduction_buffer ? options.reduction_buffer : count};
    const auto mu = static_cast<T>(options.mu);
    const auto lambda1 = static_cast<T>(options.lambda1);
    const auto lambda2 = static_cast<T>(options.lambda2);
    const auto dt = static_cast<T>(options.dt);
    const auto mu_dt = static_cast<T>(options.mu * options.dt);
    if (max_iterations) {
        std::vector<T> next(count), difference(count), inside(count), outside(count);
        std::vector<T> h(options.extended_output ? count : 0), hinv(h.size());
        // NumPy 1.x compares the float32 scalar variation at double
        // precision. NumPy 2.x weakens a Python float to float32.
        const auto threshold = options.reduction_buffer ? options.tol : static_cast<double>(static_cast<T>(options.tol));
        for (std::size_t iteration = 0; iteration < max_iterations; ++iteration) {
            if (options.extended_output)
                result.energies.push_back(energy(normalized, level_set, height, width, mu,
                                                 lambda1, lambda2, workers, plan, h, hinv, inside, outside));
            const auto [c1, c2] = binary_means(normalized, level_set, inside, outside, plan);
            evolve(normalized, level_set, height, width, mu, lambda1, lambda2, dt, mu_dt,
                   workers, c1, c2, next, difference);
            level_set.swap(next);
            const auto mean_difference = static_cast<T>(static_cast<double>(sum(difference, plan)) /
                                                       static_cast<double>(count));
            if (!(static_cast<double>(std::sqrt(mean_difference)) > threshold))
                break;
        }
    }
    result.segmentation.resize(count);
    for (std::size_t i = 0; i < count; ++i)
        result.segmentation[i] = static_cast<std::uint8_t>(level_set[i] > T(0));
    if (options.extended_output)
        result.level_set = std::move(level_set);
    if (std::fetestexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW))
        throw std::domain_error("chan_vese encountered exceptional arithmetic");
    return result;
}

template ChanVeseResult<float> chan_vese(
    const float*, const float*, std::size_t, std::size_t, std::size_t, const ChanVeseOptions&);
template ChanVeseResult<double> chan_vese(
    const double*, const double*, std::size_t, std::size_t, std::size_t, const ChanVeseOptions&);

}  // namespace skimagexx
