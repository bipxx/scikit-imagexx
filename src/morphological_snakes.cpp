#include "skimagexx/morphological_snakes.hpp"
#include "execution.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <utility>

#if defined(SKIMAGEXX_AVX2) && (defined(_M_X64) || defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#define SKIMAGEXX_SNAKES_AVX2 1
#endif

namespace skimagexx {
namespace {

struct Grid {
    std::size_t depth, height, width, size;
    int workers;

    Grid(const std::vector<std::size_t>& shape, std::size_t minimum)
        : depth(shape.size() == 3 ? shape[0] : 1), height(0), width(0), size(1), workers(1) {
        if (shape.size() != 2 && shape.size() != 3)
            throw std::invalid_argument("morphological snakes require rank two or three");
        height = shape[shape.size() - 2];
        width = shape.back();
        for (auto length : shape) {
            if (length < minimum || length > (8 * 1024 * 1024) / size)
                throw std::invalid_argument("morphological snakes axis or size out of range");
            size *= length;
        }
        workers = detail::active_workers(size);
    }

    template<class F>
    void rows(F f) const {
        detail::partition(depth * height, workers, [&](auto first, auto last, auto) {
            for (auto row = first; row < last; ++row)
                f(row / height, row % height, row * width);
        });
    }
};

using Mask = std::vector<std::int8_t>;

// The curvature operators are unions of erosions and intersections of dilations.
// Each structuring element contains its center.
class Morphology {
public:
    explicit Morphology(const Grid& grid)
        : grid_(grid), pitch_(grid.width + 2), plane_pitch_(pitch_ * (grid.height + 2)),
          padded_((grid.depth + 2) * plane_pitch_) {
        const auto y = static_cast<std::ptrdiff_t>(pitch_);
        const auto z = static_cast<std::ptrdiff_t>(plane_pitch_);
        const std::array<std::pair<std::ptrdiff_t, std::ptrdiff_t>, 9> spanning{{
            {z, y}, {z, 1}, {y, 1}, {z, y + 1}, {z, y - 1},
            {z + 1, y}, {z - 1, y}, {z + y, 1}, {z - y, 1}
        }};
        for (std::size_t d = 0; d < spanning.size(); ++d) {
            const auto a = spanning[d].first, b = spanning[d].second;
            planes_[d] = {-a-b, -a, -a+b, -b, b, a-b, a, a+b};
        }
    }

    template<bool Erode, bool Curvature>
    void transform(const Mask& input, Mask& out) {
        grid_.rows([&](auto z, auto y, auto i) {
            std::copy_n(input.data() + i, grid_.width, row(z, y));
        });
        const auto dy = static_cast<std::ptrdiff_t>(pitch_);
        const auto dz = static_cast<std::ptrdiff_t>(plane_pitch_);
        grid_.rows([&](auto z, auto y, auto i) {
            const auto* center = row(z, y);
            for (std::size_t x = 0; x < grid_.width; ++x, ++center) {
                if (bool(*center) != Erode) {
                    out[i + x] = *center;
                    continue;
                }
                bool value;
                if (grid_.depth == 1) {
                    const auto pair_matches = [&](auto offset) {
                        return combine<Erode>(center[-offset], center[offset]);
                    };
                    const bool cross = pair_matches(1);
                    const bool vertical = pair_matches(dy);
                    const bool rising = pair_matches(dy - 1);
                    const bool falling = pair_matches(dy + 1);
                    if constexpr (Curvature)
                        value = combine<!Erode>(combine<!Erode>(cross, vertical),
                                               combine<!Erode>(rising, falling));
                    else
                        value = combine<Erode>(combine<Erode>(cross, vertical),
                                              combine<Erode>(rising, falling));
                } else if constexpr (Curvature) {
                    value = !Erode;
                    for (const auto& plane : planes_) {
                        const auto matches = [&](auto offset) { return bool(center[offset]) == Erode; };
                        if (std::all_of(plane.begin(), plane.end(), matches)) {
                            value = Erode;
                            break;
                        }
                    }
                } else {
                    value = Erode;
                    for (int a = -1; a <= 1 && value == Erode; ++a)
                        for (int b = -1; b <= 1 && value == Erode; ++b)
                            for (int c = -1; c <= 1 && value == Erode; ++c)
                                value = bool(center[a * dz + b * dy + c]);
                }
                out[i + x] = static_cast<std::int8_t>(value);
            }
        });
    }

    void smooth(Mask& state, Mask& temporary, std::size_t count, bool& reversed) {
        for (std::size_t i = 0; i < count; ++i) {
            if (reversed) {
                transform<true, true>(state, temporary);
                transform<false, true>(temporary, state);
            } else {
                transform<false, true>(state, temporary);
                transform<true, true>(temporary, state);
            }
            reversed = !reversed;
        }
    }

private:
    template<bool Erode>
    static bool combine(bool a, bool b) {
        if constexpr (Erode)
            return a & b;
        else
            return a | b;
    }

    std::int8_t* row(std::size_t z, std::size_t y) {
        return padded_.data() + (z + 1) * plane_pitch_ + (y + 1) * pitch_ + 1;
    }

    const Grid& grid_;
    const std::size_t pitch_, plane_pitch_;
    Mask padded_;
    std::array<std::array<std::ptrdiff_t, 8>, 9> planes_;
};

template<class T>
T gradient(T const* row, std::size_t i, std::size_t coordinate, std::size_t length,
           std::size_t stride) {
    if (coordinate == 0)
        return T(row[i + stride] - row[i]);
    if (coordinate + 1 == length)
        return T(row[i] - row[i - stride]);
    return T((row[i + stride] - row[i - stride]) * T(0.5));
}

// For a binary level set, only the direction and zero/nonzero status are needed
// in Chan-Vese. Twice the finite difference preserves that information exactly.
int mask_gradient(const Mask& state, std::size_t i, std::size_t coordinate, std::size_t length,
                  std::size_t stride) {
    if (coordinate == 0)
        return 2 * (state[i + stride] - state[i]);
    if (coordinate + 1 == length)
        return 2 * (state[i] - state[i - stride]);
    return state[i + stride] - state[i - stride];
}

template<class T>
struct Regions {
    T outside = T(0), inside = T(0);
    std::size_t inside_count = 0;
};

template<class T>
Regions<T> operator+(Regions<T> a, Regions<T> b) {
    return {a.outside + b.outside, a.inside + b.inside, a.inside_count + b.inside_count};
}

// Ordered NumPy pairwise reduction, whose BSD notice is in third_party.
template<class T>
Regions<T> region_sums(const T* image, const std::int8_t* mask, std::size_t count) {
    const auto term = [&](std::size_t i) {
        const auto value = image[i];
        return mask[i] ? Regions<T>{T(0), value, 1} : Regions<T>{value, T(0), 0};
    };
    if (count < 8) {
        Regions<T> total;
        for (std::size_t i = 0; i < count; ++i)
            total = total + term(i);
        return total;
    }
    if (count <= 128) {
        std::array<Regions<T>, 8> lanes;
        for (std::size_t j = 0; j < lanes.size(); ++j)
            lanes[j] = term(j);
        std::size_t i = 8;
        for (; i + 8 <= count; i += 8)
            for (std::size_t j = 0; j < lanes.size(); ++j)
                lanes[j] = lanes[j] + term(i + j);
        auto total = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) +
                     ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
        for (; i < count; ++i)
            total = total + term(i);
        return total;
    }
    const auto half = count / 2 / 8 * 8;
    return region_sums(image, mask, half) + region_sums(image + half, mask + half, count - half);
}

struct GaussianWeights {
    std::vector<double> smooth, derivative;

    explicit GaussianWeights(double sigma) {
        if (!(sigma >= 0.5 && sigma <= 16))
            throw std::invalid_argument("Gaussian sigma out of range");
        const auto radius = static_cast<std::size_t>(4 * sigma + 0.5);
        const double squared = sigma * sigma;
        smooth.resize(radius + 1);
        derivative.resize(radius + 1);
        double total = 1;
        smooth[0] = 1;
        for (std::size_t d = 1; d <= radius; ++d) {
            smooth[d] = std::exp((-0.5 / squared) * double(d * d));
            total += 2 * smooth[d];
        }
        for (std::size_t d = 0; d <= radius; ++d) {
            smooth[d] /= total;
            derivative[d] = (double(d) / squared) * smooth[d];
        }
    }
};

template<bool Derivative>
double gaussian_pair(double left, double right) {
    if constexpr (Derivative)
        return right - left;
    else
        return left + right;
}

enum class GaussianOutput { filtered, square, accumulate, inverse_magnitude };

template<GaussianOutput Mode, class T>
void gaussian_result(T& target, T value, T alpha) {
    if constexpr (Mode == GaussianOutput::filtered) {
        target = value;
    } else if constexpr (Mode == GaussianOutput::square) {
        target = T(0) + value * value;
    } else if constexpr (Mode == GaussianOutput::accumulate) {
        target += value * value;
    } else {
        const T squared = target + value * value;
        target = T(1) / std::sqrt(T(1) + alpha * std::sqrt(squared));
    }
}

#ifdef SKIMAGEXX_SNAKES_AVX2
#if defined(_MSC_VER)
// The rest of this translation unit must retain the baseline ISA.
#pragma warning(push)
#pragma warning(disable: 4752)
#define SNAKES_AVX_INLINE __forceinline
#define SNAKES_AVX_ENTRY __declspec(noinline)
#else
#define SNAKES_AVX_INLINE __attribute__((target("avx2"), always_inline)) inline
#define SNAKES_AVX_ENTRY __attribute__((target("avx2"), noinline))
#endif

template<class T>
SNAKES_AVX_INLINE __m256d gaussian_load(const T* input) {
    if constexpr (sizeof(T) == sizeof(float))
        return _mm256_cvtps_pd(_mm_loadu_ps(input));
    else
        return _mm256_loadu_pd(input);
}

template<GaussianOutput Mode, class T>
SNAKES_AVX_INLINE void gaussian_store(T* output, __m256d values, T alpha) {
    if constexpr (sizeof(T) == sizeof(float)) {
        auto result = _mm256_cvtpd_ps(values);
        if constexpr (Mode != GaussianOutput::filtered)
            result = _mm_mul_ps(result, result);
        if constexpr (Mode == GaussianOutput::square)
            result = _mm_add_ps(_mm_setzero_ps(), result);
        if constexpr (Mode == GaussianOutput::accumulate || Mode == GaussianOutput::inverse_magnitude)
            result = _mm_add_ps(_mm_loadu_ps(output), result);
        if constexpr (Mode == GaussianOutput::inverse_magnitude) {
            const auto one = _mm_set1_ps(1.0f);
            const auto scaled = _mm_mul_ps(_mm_set1_ps(alpha), _mm_sqrt_ps(result));
            result = _mm_div_ps(one, _mm_sqrt_ps(_mm_add_ps(one, scaled)));
        }
        _mm_storeu_ps(output, result);
    } else {
        auto result = values;
        if constexpr (Mode != GaussianOutput::filtered)
            result = _mm256_mul_pd(result, result);
        if constexpr (Mode == GaussianOutput::square)
            result = _mm256_add_pd(_mm256_setzero_pd(), result);
        if constexpr (Mode == GaussianOutput::accumulate || Mode == GaussianOutput::inverse_magnitude)
            result = _mm256_add_pd(_mm256_loadu_pd(output), result);
        if constexpr (Mode == GaussianOutput::inverse_magnitude) {
            const auto one = _mm256_set1_pd(1.0);
            const auto scaled = _mm256_mul_pd(_mm256_set1_pd(alpha), _mm256_sqrt_pd(result));
            result = _mm256_div_pd(one, _mm256_sqrt_pd(_mm256_add_pd(one, scaled)));
        }
        _mm256_storeu_pd(output, result);
    }
}

template<bool Derivative, class T>
SNAKES_AVX_INLINE __m256d gaussian_initial(const T* input, __m256d weight) {
    if constexpr (Derivative)
        return _mm256_setzero_pd();
    else
        return _mm256_mul_pd(gaussian_load(input), weight);
}

template<bool Derivative, class T>
SNAKES_AVX_INLINE __m256d gaussian_add(__m256d sum, const T* left, const T* right,
                                     __m256d weight) {
    const auto a = gaussian_load(left), b = gaussian_load(right);
    const auto pair = Derivative ? _mm256_sub_pd(b, a) : _mm256_add_pd(a, b);
    return _mm256_add_pd(sum, _mm256_mul_pd(pair, weight));
}

template<bool Derivative, GaussianOutput Mode, class T>
SNAKES_AVX_ENTRY std::size_t gaussian_span_avx2(
    const T* input, T* output, std::size_t count, const T* const* lower,
    const T* const* upper, std::size_t radius, const double* weights, T alpha) {
    const auto center = _mm256_set1_pd(weights[0]);
    std::size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        auto a = gaussian_initial<Derivative>(input + i, center);
        auto b = gaussian_initial<Derivative>(input + i + 4, center);
        auto c = gaussian_initial<Derivative>(input + i + 8, center);
        auto d = gaussian_initial<Derivative>(input + i + 12, center);
        for (auto k = radius; k > 0; --k) {
            const auto* left = lower[k - 1] + i;
            const auto* right = upper[k - 1] + i;
            const auto weight = _mm256_set1_pd(weights[k]);
            a = gaussian_add<Derivative>(a, left, right, weight);
            b = gaussian_add<Derivative>(b, left + 4, right + 4, weight);
            c = gaussian_add<Derivative>(c, left + 8, right + 8, weight);
            d = gaussian_add<Derivative>(d, left + 12, right + 12, weight);
        }
        gaussian_store<Mode>(output + i, a, alpha);
        gaussian_store<Mode>(output + i + 4, b, alpha);
        gaussian_store<Mode>(output + i + 8, c, alpha);
        gaussian_store<Mode>(output + i + 12, d, alpha);
    }
    for (; i + 4 <= count; i += 4) {
        auto sum = gaussian_initial<Derivative>(input + i, center);
        for (auto k = radius; k > 0; --k)
            sum = gaussian_add<Derivative>(sum, lower[k - 1] + i, upper[k - 1] + i,
                                           _mm256_set1_pd(weights[k]));
        gaussian_store<Mode>(output + i, sum, alpha);
    }
    _mm256_zeroupper();
    return i;
}

#undef SNAKES_AVX_INLINE
#undef SNAKES_AVX_ENTRY
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
#endif

// Each span has fixed offsets, including when a tap clamps to the edge.
// Evaluate several independent accumulators without reordering any sum.
template<bool Derivative, GaussianOutput Mode, class T>
void gaussian_span(const T* input, T* output, std::size_t count, std::size_t stride,
                   std::size_t left, std::size_t right, const std::vector<double>& weights,
                   bool vectorize, T alpha) {
    constexpr std::size_t lanes = 8;
    std::array<const T*, 64> lower, upper;
    const auto radius = weights.size() - 1;
    for (std::size_t d = 1; d <= radius; ++d) {
        lower[d - 1] = input - std::min(d, left) * stride;
        upper[d - 1] = input + std::min(d, right) * stride;
    }

    const auto initial = [&](std::size_t i) {
        if constexpr (Derivative)
            return 0.0;
        else
            return double(input[i]) * weights[0];
    };
    std::size_t i = 0;
#ifdef SKIMAGEXX_SNAKES_AVX2
    if (vectorize && count >= 4)
        i = gaussian_span_avx2<Derivative, Mode>(input, output, count, lower.data(), upper.data(),
                                                 radius, weights.data(), alpha);
#else
    (void)vectorize;
#endif
    for (; i + lanes <= count; i += lanes) {
        std::array<double, lanes> sum;
        for (std::size_t lane = 0; lane < lanes; ++lane)
            sum[lane] = initial(i + lane);
        for (auto d = radius; d > 0; --d) {
            const auto* a = lower[d - 1] + i;
            const auto* b = upper[d - 1] + i;
            const double weight = weights[d];
            for (std::size_t lane = 0; lane < lanes; ++lane)
                sum[lane] += gaussian_pair<Derivative>(a[lane], b[lane]) * weight;
        }
        for (std::size_t lane = 0; lane < lanes; ++lane)
            gaussian_result<Mode>(output[i + lane], T(sum[lane]), alpha);
    }
    for (; i < count; ++i) {
        auto sum = initial(i);
        for (auto d = radius; d > 0; --d)
            sum += gaussian_pair<Derivative>(lower[d - 1][i], upper[d - 1][i]) * weights[d];
        gaussian_result<Mode>(output[i], T(sum), alpha);
    }
}

template<bool Derivative, GaussianOutput Mode = GaussianOutput::filtered, class T>
void gaussian_axis(const T* input, T* output, const Grid& grid, std::size_t length,
                   std::size_t stride, const std::vector<double>& weights, T alpha = T(0)) {
    const std::size_t period = length * stride;
    const std::size_t radius = weights.size() - 1;
    const auto unclamped_begin = std::min(radius, length) * stride;
    const auto unclamped_end = length > radius ? (length - radius) * stride : 0;
#ifdef SKIMAGEXX_SNAKES_AVX2
    const bool vectorize = has_avx2();
#else
    constexpr bool vectorize = false;
#endif
    detail::partition(grid.size, grid.workers, [&](auto begin, auto end, auto) {
        while (begin < end) {
            const auto block = begin / period * period;
            const auto stop = std::min(end, block + period);
            while (begin < stop) {
                const auto offset = begin - block;
                const auto coordinate = offset / stride;
                const bool unclamped = offset >= unclamped_begin && offset < unclamped_end;
                const auto span_end = unclamped ? unclamped_end : (coordinate + 1) * stride;
                const auto next = std::min(stop, block + span_end);
                gaussian_span<Derivative, Mode>(input + begin, output + begin, next - begin, stride,
                                                coordinate, length - 1 - coordinate, weights,
                                                vectorize, alpha);
                begin = next;
            }
        }
    });
}

} // namespace

template<class T>
void inverse_gaussian_gradient_into(const T* image, T* output,
    const std::vector<std::size_t>& shape, double alpha, double sigma) {
    const Grid grid(shape, 1);
    const GaussianWeights weights(sigma);
    const auto rank = shape.size();
    std::vector<T> first(grid.size);
    std::vector<T> second(rank == 3 ? grid.size : 0), smooth_first(rank == 3 ? grid.size : 0);
    const auto first_stride = grid.size / shape[0];
    if (rank == 3)
        gaussian_axis<false>(image, smooth_first.data(), grid, shape[0], first_stride, weights.smooth);

    for (std::size_t differentiated = 0; differentiated < rank; ++differentiated) {
        const bool shared_first = rank == 3 && differentiated > 0;
        const T* source = shared_first ? smooth_first.data() : image;
        T* target = first.data();
        std::size_t stride = shared_first ? first_stride : grid.size;
        // Preserve the provider's axis order and round each pass to T.
        for (std::size_t axis = shared_first ? 1 : 0; axis + 1 < rank; ++axis) {
            stride /= shape[axis];
            if (axis == differentiated)
                gaussian_axis<true>(source, target, grid, shape[axis], stride, weights.derivative);
            else
                gaussian_axis<false>(source, target, grid, shape[axis], stride, weights.smooth);
            source = target;
            target = target == first.data() ? second.data() : first.data();
        }
        // Round the last convolution to T before squaring, just like the
        // stored intermediate. Combine components in derivative-axis order.
        if (differentiated == 0)
            gaussian_axis<false, GaussianOutput::square>(source, output, grid, shape.back(), 1,
                                                          weights.smooth);
        else if (differentiated + 1 == rank)
            gaussian_axis<true, GaussianOutput::inverse_magnitude>(source, output, grid, shape.back(), 1,
                                                                   weights.derivative, T(alpha));
        else
            gaussian_axis<false, GaussianOutput::accumulate>(source, output, grid, shape.back(), 1,
                                                              weights.smooth);
    }
}

template<class T>
void morphological_chan_vese_into(const T* image, const std::int8_t* initial,
    std::int8_t* output, const std::vector<std::size_t>& shape, std::size_t iterations,
    std::size_t smoothing, double lambda1, double lambda2, bool reverse_curvature) {
    const Grid grid(shape, 2);
    Mask state(initial, initial + grid.size), next(grid.size);
    Morphology morphology(grid);
    for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
        const auto sums = region_sums(image, state.data(), grid.size);
        const auto outside = sums.outside / T(double(grid.size - sums.inside_count) + 1e-8);
        const auto inside = sums.inside / T(double(sums.inside_count) + 1e-8);
        grid.rows([&](auto z, auto y, auto row) {
            for (std::size_t x = 0; x < grid.width; ++x) {
                const auto i = row + x;
                next[i] = state[i];
                if (mask_gradient(state, i, x, grid.width, 1) ||
                    mask_gradient(state, i, y, grid.height, grid.width) ||
                    (grid.depth > 1 && mask_gradient(state, i, z, grid.depth, grid.width * grid.height))) {
                    const T inner = image[i] - inside, outer = image[i] - outside;
                    const T energy = T(lambda1) * (inner * inner) - T(lambda2) * (outer * outer);
                    if (energy != T(0))
                        next[i] = std::int8_t(energy < T(0));
                }
            }
        });
        state.swap(next);
        morphology.smooth(state, next, smoothing, reverse_curvature);
    }
    std::copy(state.begin(), state.end(), output);
}

template<class T>
void morphological_geodesic_active_contour_into(const T* image, const std::int8_t* initial,
    std::int8_t* output, const std::vector<std::size_t>& shape, std::size_t iterations,
    std::size_t smoothing, double threshold, int balloon, bool reverse_curvature) {
    const Grid grid(shape, 2);
    Mask state(initial, initial + grid.size), next(grid.size);
    Morphology morphology(grid);
    std::vector<T> dx(grid.size), dy(grid.size);
    std::vector<T> dz(grid.depth > 1 ? grid.size : 0);
    const auto plane = grid.width * grid.height;
    grid.rows([&](auto z, auto y, auto row) {
        for (std::size_t x = 0; x < grid.width; ++x) {
            const auto i = row + x;
            dx[i] = gradient(image, i, x, grid.width, 1);
            dy[i] = gradient(image, i, y, grid.height, grid.width);
            if (grid.depth > 1)
                dz[i] = gradient(image, i, z, grid.depth, plane);
        }
    });
    for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
        if (balloon != 0) {
            if (balloon < 0)
                morphology.transform<true, false>(state, next);
            else
                morphology.transform<false, false>(state, next);
            detail::partition(grid.size, grid.workers, [&](auto begin, auto end, auto) {
                for (auto i = begin; i < end; ++i)
                    if (double(image[i]) > threshold)
                        state[i] = next[i];
            });
        }
        grid.rows([&](auto z, auto y, auto row) {
            for (std::size_t x = 0; x < grid.width; ++x) {
                const auto i = row + x;
                const auto add = [&](T energy, T dimage, int twice_dmask) {
                    return T(double(energy) + double(dimage) * (double(twice_dmask) * 0.5));
                };
                T energy = T(0);
                if (grid.depth > 1)
                    energy = add(energy, dz[i], mask_gradient(state, i, z, grid.depth, plane));
                energy = add(energy, dy[i], mask_gradient(state, i, y, grid.height, grid.width));
                energy = add(energy, dx[i], mask_gradient(state, i, x, grid.width, 1));
                next[i] = energy == T(0) ? state[i] : std::int8_t(energy > T(0));
            }
        });
        state.swap(next);
        morphology.smooth(state, next, smoothing, reverse_curvature);
    }
    std::copy(state.begin(), state.end(), output);
}

template void inverse_gaussian_gradient_into<float>(const float*, float*,
    const std::vector<std::size_t>&, double, double);
template void inverse_gaussian_gradient_into<double>(const double*, double*,
    const std::vector<std::size_t>&, double, double);
template void morphological_chan_vese_into<float>(const float*, const std::int8_t*,
    std::int8_t*, const std::vector<std::size_t>&, std::size_t, std::size_t, double, double, bool);
template void morphological_chan_vese_into<double>(const double*, const std::int8_t*,
    std::int8_t*, const std::vector<std::size_t>&, std::size_t, std::size_t, double, double, bool);
template void morphological_geodesic_active_contour_into<float>(const float*, const std::int8_t*,
    std::int8_t*, const std::vector<std::size_t>&, std::size_t, std::size_t, double, int, bool);
template void morphological_geodesic_active_contour_into<double>(const double*, const std::int8_t*,
    std::int8_t*, const std::vector<std::size_t>&, std::size_t, std::size_t, double, int, bool);

} // namespace skimagexx
