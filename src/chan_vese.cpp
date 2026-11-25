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

#ifdef SKIMAGEXX_AVX2
#include <immintrin.h>
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable: 4752)
#define CHAN_VESE_AVX2 __declspec(noinline)
#define CHAN_VESE_AVX_INLINE __forceinline
#else
#define CHAN_VESE_AVX2 __attribute__((target("avx2"), noinline))
#define CHAN_VESE_AVX_INLINE __attribute__((target("avx2"), always_inline)) inline
#endif
#endif

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

#ifdef SKIMAGEXX_AVX2
template<class T>
CHAN_VESE_AVX2 std::size_t normalize_avx2(const T* image, T* output, std::size_t count,
                                         T low, T range) {
    std::size_t i = 0;
    if constexpr (std::is_same_v<T, double>) {
        const auto minimum = _mm256_set1_pd(low);
        const auto span = _mm256_set1_pd(range);
        for (; count - i >= 4; i += 4)
            _mm256_storeu_pd(output + i, _mm256_div_pd(_mm256_sub_pd(_mm256_loadu_pd(image + i), minimum), span));
    } else {
        const auto minimum = _mm256_set1_ps(low);
        const auto span = _mm256_set1_ps(range);
        for (; count - i >= 8; i += 8)
            _mm256_storeu_ps(output + i, _mm256_div_ps(_mm256_sub_ps(_mm256_loadu_ps(image + i), minimum), span));
    }
    return i;
}
#endif

template<class T>
std::vector<T> normalize(const T* image, const T* initial, std::size_t count, [[maybe_unused]] bool vectorized) {
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
        std::size_t i = 0;
#ifdef SKIMAGEXX_AVX2
        if (vectorized)
            i = normalize_avx2(image, output.data(), count, low, range);
#endif
        for (; i < count; ++i)
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

// Match tv_reductions.cpp's NumPy addition tree, including its profiled
// buffer boundaries and the distinction between the outer +0 and leaf -0.
constexpr std::size_t reduction_leaf_size = 128;

template<class V, class Load>
V leaf_sum(std::size_t count, V negative_zero, const Load& load) {
    if (count < 8) {
        for (std::size_t i = 0; i < count; ++i)
            negative_zero = negative_zero + load(i);
        return negative_zero;
    }
    V lane[8];
    for (std::size_t j = 0; j < 8; ++j)
        lane[j] = load(j);
    std::size_t i = 8;
    for (; i < count - count % 8; i += 8)
        for (std::size_t j = 0; j < 8; ++j)
            lane[j] = lane[j] + load(i + j);
    V value = ((lane[0] + lane[1]) + (lane[2] + lane[3])) +
              ((lane[4] + lane[5]) + (lane[6] + lane[7]));
    for (; i < count; ++i)
        value = value + load(i);
    return value;
}

template<class Leaf>
auto sum_tree(std::size_t first, std::size_t count, const Leaf& leaf) {
    if (count <= reduction_leaf_size)
        return leaf(first, count);
    const auto half = (count / 2) & ~std::size_t(7);
    const auto left = sum_tree(first, half, leaf);
    const auto right = sum_tree(first + half, count - half, leaf);
    return left + right;
}

template<class V, class Leaf>
V sum_chunks(std::size_t count, TvReductionPlan plan, V value, const Leaf& leaf) {
    const auto block = plan.profile == TvReductionProfile::numpy_1244_buffered ? plan.buffer_elements : count;
    for (std::size_t first = 0; first < count;) {
        const auto size = std::min(block, count - first);
        value = value + sum_tree(first, size, leaf);
        first += size;
    }
    return value;
}

template<class T>
struct BinarySums {
    T inside;
    T outside;

    BinarySums operator+(BinarySums other) const {
        return {inside + other.inside, outside + other.outside};
    }
};

template<class T>
std::pair<T, T> binary_means(const std::vector<T>& image, const std::vector<T>& level_set,
                             TvReductionPlan plan) {
    std::size_t positives = 0;
    const auto sums = sum_chunks(image.size(), plan, BinarySums<T>{}, [&](std::size_t first, std::size_t count) {
        return leaf_sum(count, BinarySums<T>{-T(0), -T(0)}, [&](std::size_t offset) {
            const auto i = first + offset;
            const bool selected = level_set[i] > T(0);
            positives += static_cast<std::size_t>(selected);
            return selected ? BinarySums<T>{image[i], T(0)} : BinarySums<T>{T(0), image[i]};
        });
    });
    const auto negatives = image.size() - positives;
    return {positives ? sums.inside / static_cast<T>(positives) : T(0),
            negatives ? sums.outside / static_cast<T>(negatives) : T(0)};
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
struct Evolution {
    T mu, lambda1, lambda2, dt, mu_dt, c1, c2;
};

template<class T>
T evolve_pixel(T input, T center, T left, T right, T up, T down, const Evolution<T>& params) {
    constexpr T eta = T(1e-16);
    const T dx = (right - left) / T(2);
    const T dy = (down - up) / T(2);
    const T c_right = T(1) / std::sqrt(eta + square(right - center) + square(dy));
    const T c_left = T(1) / std::sqrt(eta + square(center - left) + square(dy));
    const T c_down = T(1) / std::sqrt(eta + square(dx) + square(down - center));
    const T c_up = T(1) / std::sqrt(eta + square(dx) + square(center - up));
    const T curvature = right * c_right + left * c_left + down * c_down + up * c_up;
    const T region = -params.lambda1 * square(input - params.c1) + params.lambda2 * square(input - params.c2);
    const T d = delta(center);
    return (center + (params.dt * d) * (params.mu * curvature + region)) /
           (T(1) + params.mu_dt * d * (c_right + c_left + c_down + c_up));
}

#ifdef SKIMAGEXX_AVX2
template<class T> struct AvxLanes;
template<> struct AvxLanes<float> { using Type = __m256; static constexpr std::size_t count = 8; };
template<> struct AvxLanes<double> { using Type = __m256d; static constexpr std::size_t count = 4; };

template<class T>
struct AvxValue {
    using Lanes = typename AvxLanes<T>::Type;
    Lanes v;

    CHAN_VESE_AVX_INLINE static AvxValue broadcast(T value) {
        if constexpr (std::is_same_v<T, double>) return {_mm256_set1_pd(value)};
        else return {_mm256_set1_ps(value)};
    }
    CHAN_VESE_AVX_INLINE static AvxValue load(const T* source) {
        if constexpr (std::is_same_v<T, double>) return {_mm256_loadu_pd(source)};
        else return {_mm256_loadu_ps(source)};
    }
    CHAN_VESE_AVX_INLINE void store(T* output) const {
        if constexpr (std::is_same_v<T, double>) _mm256_storeu_pd(output, v);
        else _mm256_storeu_ps(output, v);
    }
    CHAN_VESE_AVX_INLINE AvxValue operator+(AvxValue other) const {
        if constexpr (std::is_same_v<T, double>) return {_mm256_add_pd(v, other.v)};
        else return {_mm256_add_ps(v, other.v)};
    }
    CHAN_VESE_AVX_INLINE AvxValue operator-(AvxValue other) const {
        if constexpr (std::is_same_v<T, double>) return {_mm256_sub_pd(v, other.v)};
        else return {_mm256_sub_ps(v, other.v)};
    }
    CHAN_VESE_AVX_INLINE AvxValue operator*(AvxValue other) const {
        if constexpr (std::is_same_v<T, double>) return {_mm256_mul_pd(v, other.v)};
        else return {_mm256_mul_ps(v, other.v)};
    }
    CHAN_VESE_AVX_INLINE AvxValue operator/(AvxValue other) const {
        if constexpr (std::is_same_v<T, double>) return {_mm256_div_pd(v, other.v)};
        else return {_mm256_div_ps(v, other.v)};
    }
    CHAN_VESE_AVX_INLINE AvxValue sqrt() const {
        if constexpr (std::is_same_v<T, double>) return {_mm256_sqrt_pd(v)};
        else return {_mm256_sqrt_ps(v)};
    }
    CHAN_VESE_AVX_INLINE AvxValue squared() const { return *this * *this; }
};

template<class T>
CHAN_VESE_AVX2 std::size_t evolve_avx2(const T* image, const T* phi, const T* up, const T* down,
                                      T* next, T* difference, std::size_t count, const Evolution<T>& params) {
    using V = AvxValue<T>;
    constexpr auto lanes = AvxLanes<T>::count;
    const auto one = V::broadcast(T(1));
    const auto half = V::broadcast(T(0.5));
    const auto eta = V::broadcast(T(1e-16));
    const auto mu = V::broadcast(params.mu);
    const auto negative_lambda1 = V::broadcast(-params.lambda1);
    const auto lambda2 = V::broadcast(params.lambda2);
    const auto dt = V::broadcast(params.dt);
    const auto mu_dt = V::broadcast(params.mu_dt);
    const auto c1 = V::broadcast(params.c1);
    const auto c2 = V::broadcast(params.c2);
    std::size_t i = 0;
    for (; count - i >= lanes; i += lanes) {
        const auto center = V::load(phi + i);
        const auto left = V::load(phi + i - 1);
        const auto right = V::load(phi + i + 1);
        const auto above = V::load(up + i);
        const auto below = V::load(down + i);
        const auto dx2 = ((right - left) * half).squared();
        const auto dy2 = ((below - above) * half).squared();
        const auto c_right = one / (eta + (right - center).squared() + dy2).sqrt();
        const auto c_left = one / (eta + (center - left).squared() + dy2).sqrt();
        const auto c_down = one / (eta + dx2 + (below - center).squared()).sqrt();
        const auto c_up = one / (eta + dx2 + (center - above).squared()).sqrt();
        const auto curvature = right * c_right + left * c_left + below * c_down + above * c_up;
        const auto input = V::load(image + i);
        const auto region = negative_lambda1 * (input - c1).squared() + lambda2 * (input - c2).squared();
        const auto d = one / (one + center.squared());
        const auto updated = (center + (dt * d) * (mu * curvature + region)) /
                             (one + mu_dt * d * (c_right + c_left + c_down + c_up));
        updated.store(next + i);
        (updated - center).squared().store(difference + i);
    }
    return i;
}

#undef CHAN_VESE_AVX2
#undef CHAN_VESE_AVX_INLINE
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
#endif

template<class T>
T evolve_tile(const T* image, const T* phi, std::size_t height, std::size_t width,
              const Evolution<T>& params, T* next, std::size_t first, std::size_t count,
              [[maybe_unused]] bool vectorized) {
    T difference[reduction_leaf_size];
    const auto end = first + count;
    std::size_t row = first / width;
    std::size_t col = first % width;
    for (auto i = first; i < end; ++row, col = 0) {
        const auto row_start = i - col;
        const auto up = row ? row_start - width : row_start;
        const auto down = row + 1 < height ? row_start + width : row_start;
        const auto row_end = std::min(row_start + width, end);
        const auto pixel = [&](std::size_t left, std::size_t right) {
            next[i] = evolve_pixel(image[i], phi[i], phi[left], phi[right],
                                   phi[up + col], phi[down + col], params);
            difference[i - first] = square(next[i] - phi[i]);
            ++i;
            ++col;
        };
        if (col == 0)
            pixel(i, width == 1 ? i : i + 1);
        const auto interior_end = std::min(row_start + width - 1, row_end);
#ifdef SKIMAGEXX_AVX2
        if (vectorized && i < interior_end && interior_end - i >= AvxLanes<T>::count) {
            const auto processed = evolve_avx2(image + i, phi + i, phi + up + col, phi + down + col,
                                               next + i, difference + (i - first), interior_end - i, params);
            i += processed;
            col += processed;
        }
#endif
        for (; i < interior_end;)
            pixel(i - 1, i + 1);
        if (i < row_end)
            pixel(i - 1, i);
    }
    return leaf_sum(count, -T(0), [&](std::size_t offset) { return difference[offset]; });
}

struct ReductionTile {
    std::size_t first;
    std::size_t count;
};

template<class T>
T evolve(const std::vector<T>& image, const std::vector<T>& level_set, std::size_t height,
         std::size_t width, const Evolution<T>& params, int workers, bool vectorized,
         std::vector<T>& next, TvReductionPlan plan, const std::vector<ReductionTile>& tiles,
         std::vector<T>& partials) {
    const auto tile_sum = [&](std::size_t first, std::size_t count) {
        return evolve_tile(image.data(), level_set.data(), height, width, params, next.data(),
                           first, count, vectorized);
    };
    if (tiles.empty())
        return sum_chunks(image.size(), plan, T(0), tile_sum);

    detail::partition(tiles.size(), workers, [&](std::size_t begin, std::size_t end, std::size_t) {
        for (std::size_t i = begin; i < end; ++i)
            partials[i] = tile_sum(tiles[i].first, tiles[i].count);
    });
    std::size_t i = 0;
    return sum_chunks(image.size(), plan, T(0), [&](std::size_t, std::size_t) { return partials[i++]; });
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
    bool vectorized = false;
#ifdef SKIMAGEXX_AVX2
    vectorized = has_avx2();
#endif
    const auto normalized = normalize(image, initial_level_set, count, vectorized);
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
        std::vector<T> next(count);
        const auto energy_size = options.extended_output ? count : 0;
        std::vector<T> h(energy_size), hinv(energy_size), inside(energy_size), outside(energy_size);
        std::vector<ReductionTile> tiles;
        if (workers > 1)
            sum_chunks(count, plan, std::size_t(0), [&](std::size_t first, std::size_t size) {
                tiles.push_back({first, size});
                return std::size_t(0);
            });
        std::vector<T> partials(tiles.size());
        // NumPy 1.x compares the float32 scalar variation at double
        // precision. NumPy 2.x weakens a Python float to float32.
        const auto threshold = options.reduction_buffer ? options.tol : static_cast<double>(static_cast<T>(options.tol));
        for (std::size_t iteration = 0; iteration < max_iterations; ++iteration) {
            if (options.extended_output)
                result.energies.push_back(energy(normalized, level_set, height, width, mu,
                                                 lambda1, lambda2, workers, plan, h, hinv, inside, outside));
            const auto [c1, c2] = binary_means(normalized, level_set, plan);
            const auto difference = evolve(normalized, level_set, height, width,
                                           Evolution<T>{mu, lambda1, lambda2, dt, mu_dt, c1, c2},
                                           workers, vectorized, next, plan, tiles, partials);
            level_set.swap(next);
            const auto mean_difference = static_cast<T>(static_cast<double>(difference) / static_cast<double>(count));
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
