// The homography interpolation follows scikit-image's BSD-licensed
// _warps_cy.pyx and interpolation.pxd. See third_party/scikit-image-LICENSE.txt.
#include "skimagexx/warp.hpp"
#include "execution.hpp"
#include "resize_cast.hpp"
#include "resize_clip.hpp"
#include "simd.hpp"
#include "spline_detail.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>
#ifdef SKIMAGEXX_AVX2
#include <immintrin.h>
#endif

namespace skimagexx {
namespace {
using Index = std::ptrdiff_t;
constexpr auto max_count = std::size_t(std::numeric_limits<Index>::max()) / sizeof(double);

template<class T>
Index floor_index(T x) {
    const auto integer = Index(x);
    return integer - (T(integer) > x);
}

std::size_t count(WarpShape shape) {
    if (!shape.rows || !shape.cols || !shape.channels ||
        shape.rows > max_count / shape.cols ||
        shape.rows * shape.cols > max_count / shape.channels)
        throw std::invalid_argument("nonempty addressable warp dimensions required");
    return shape.rows * shape.cols * shape.channels;
}

Index tap_index(Index index, Index length, GaussianBorder border) {
    if (index >= 0 && index < length) return index;
    switch (border) {
        case GaussianBorder::constant: return -1;
        case GaussianBorder::nearest: return index < 0 ? 0 : length - 1;
        case GaussianBorder::mirror: {
            if (length == 1) return 0;
            const Index period = 2 * (length - 1);
            const Index offset = std::abs(index % period);
            return offset < length ? offset : period - offset;
        }
        case GaussianBorder::reflect: {
            const Index period = 2 * length;
            Index offset = index % period;
            if (offset < 0) offset += period;
            return offset < length ? offset : period - 1 - offset;
        }
        case GaussianBorder::wrap: {
            const Index offset = index % length;
            return offset < 0 ? offset + length : offset;
        }
    }
    return -1;
}

// ndimage extends fractional coordinates before choosing spline knots.
double extend(double x, Index n, GaussianBorder border) {
    if (x >= 0 && x <= double(n - 1)) return x;
    switch (border) {
        case GaussianBorder::nearest: return std::clamp(x, 0., double(n - 1));
        case GaussianBorder::mirror: {
            if (n == 1) return 0;
            const double period = double(2 * n - 2);
            const double offset = std::abs(std::fmod(x, period));
            return offset < double(n) ? offset : period - offset;
        }
        case GaussianBorder::reflect: {
            if (n == 1) return 0;
            const double period = double(2 * n);
            if (x < 0) {
                if (x < -period) x = std::fmod(x, period);
                if (x < -double(n)) return x + period;
                return (x > -1e-15 ? 1e-15 : -x) - 1.;
            }
            x = std::fmod(x, period);
            return x >= double(n) ? period - x - 1. : x;
        }
        case GaussianBorder::wrap:
            if (n == 1) return 0;
            // The negative branch uses ndimage's truncation convention.
            if (x < 0) return x + double(n * (Index((-1. - x) / double(n)) + 1));
            return std::fmod(x, double(n));
        case GaussianBorder::constant: return x;
    }
    return x;
}

template<class T>
struct Matrix {
    std::array<T, 9> p;
    bool projective;
    explicit Matrix(const WarpMap& map) {
        std::transform(map.params.begin(), map.params.end(), p.begin(),
                       [](double x) { return T(x); });
        projective = p[6] != 0 || p[7] != 0 || p[8] != 1;
    }
    std::array<T, 2> operator()(T x, T y) const {
        const T a = p[0] * x + p[1] * y + p[2];
        const T b = p[3] * x + p[4] * y + p[5];
        if (!projective) return {a, b};
        const T z = p[6] * x + p[7] * y + p[8];
        return {a / z, b / z};
    }
    bool bounded(WarpShape shape) const {
        const T last_x = T(shape.cols - 1);
        const T last_y = T(shape.rows - 1);
        const bool positive = p[8] > 0;
        for (T y : {T(0), last_y}) {
            for (T x : {T(0), last_x}) {
                const T z = p[6] * x + p[7] * y + p[8];
                // Stay clear of projective horizons and integer tap limits.
                if (!std::isfinite(z) || std::abs(z) < T(1e-12) || (z > 0) != positive)
                    return false;
                const auto xy = (*this)(x, y);
                if (!(std::abs(xy[0]) < T(1e9) && std::abs(xy[1]) < T(1e9)))
                    return false;
            }
        }
        return true;
    }
};

template<class T>
struct Sampler {
    const T* input;
    WarpShape shape;
    GaussianBorder border;
    T fill;
    T at(Index r, Index c, std::size_t k) const {
        const auto row = tap_index(r, Index(shape.rows), border);
        const auto col = tap_index(c, Index(shape.cols), border);
        return row < 0 || col < 0 ? fill : direct(row, col, k);
    }
    T direct(Index r, Index c, std::size_t k) const {
        return input[(std::size_t(r) * shape.cols + std::size_t(c)) * shape.channels + k];
    }
};

template<class T>
T cubic(T x, T a, T b, T c, T d) {
    return b + T(.5) * x * (c - a + x * (T(2) * a - T(5) * b + T(4) * c - d
                                        + x * (T(3) * (b - c) + d - a)));
}

template<int Order, bool Gray, class T>
void fast_pixel(const Sampler<T>& s, T x, T y, T* output) {
    const auto c = floor_index(x);
    const auto r = floor_index(y);
    const T dx = x - T(c), dy = y - T(r);
    const auto channels = Gray ? std::size_t(1) : s.shape.channels;
    if constexpr (Order == 1) {
        const auto right = c + (dx != 0), bottom = r + (dy != 0);
        const bool inside = r >= 0 && c >= 0 && bottom < Index(s.shape.rows) &&
                            right < Index(s.shape.cols);
        const auto interpolate = [=](T tl, T tr, T bl, T br) {
            const double a = (T(1) - dx) * tl + dx * tr;
            const double b = (T(1) - dx) * bl + dx * br;
            return T((T(1) - dy) * a + dy * b);
        };
        if (inside) {
            const auto stride = s.shape.cols * channels;
            const auto* top = s.input + std::size_t(r) * stride + std::size_t(c) * channels;
            const auto* lower = top + std::size_t(bottom - r) * stride;
            const auto next = std::size_t(right - c) * channels;
            for (std::size_t k = 0; k < channels; ++k)
                output[k] = interpolate(top[k], top[k + next], lower[k], lower[k + next]);
        } else {
            for (std::size_t k = 0; k < channels; ++k)
                output[k] = interpolate(s.at(r, c, k), s.at(r, right, k),
                                        s.at(bottom, c, k), s.at(bottom, right, k));
        }
    } else {
        const bool inside = r > 0 && c > 0 && r + 2 < Index(s.shape.rows) &&
                            c + 2 < Index(s.shape.cols);
        for (std::size_t k = 0; k < channels; ++k) {
            T rows[4];
            for (Index i = 0; i < 4; ++i) {
                const auto j = r + i - 1;
                if (inside)
                    rows[i] = cubic(dx, s.direct(j, c - 1, k), s.direct(j, c, k),
                                   s.direct(j, c + 1, k), s.direct(j, c + 2, k));
                else
                    rows[i] = cubic(dx, s.at(j, c - 1, k), s.at(j, c, k),
                                   s.at(j, c + 1, k), s.at(j, c + 2, k));
            }
            output[k] = cubic(dy, rows[0], rows[1], rows[2], rows[3]);
        }
    }
}

template<class Body>
void rows(WarpShape s, Body body) {
    detail::partition(s.rows, detail::active_workers(count(s)),
        [&](std::size_t begin, std::size_t end, std::size_t) {
            for (auto row = begin; row < end; ++row) body(row);
        });
}

#ifdef SKIMAGEXX_AVX2
template<class T> struct AffineVector;
template<> struct AffineVector<float> { using Type = __m256; };
template<> struct AffineVector<double> { using Type = __m256d; };

#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(push)
#pragma warning(disable: 4752)
#pragma float_control(precise, on, push)
#pragma fp_contract(off)
#define WARP_AVX_ENTRY __declspec(noinline)
#define WARP_AVX_INLINE __forceinline
#elif defined(__clang__)
#define WARP_AVX_ENTRY __attribute__((target("avx2"), noinline))
#define WARP_AVX_INLINE __attribute__((target("avx2"), always_inline)) inline
#else
#define WARP_AVX_ENTRY __attribute__((target("avx2"), optimize("fp-contract=off"), noinline))
#define WARP_AVX_INLINE __attribute__((target("avx2"), optimize("fp-contract=off"), always_inline)) inline
#endif

WARP_AVX_INLINE __m128 bilinear_vertical(__m128 top, __m128 bottom, __m128 dy,
                                        __m128 one_minus_dy) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    // The scalar float path rounds each horizontal blend to float and
    // carries out the vertical blend in double.
    return _mm256_cvtpd_ps(_mm256_add_pd(
        _mm256_mul_pd(_mm256_cvtps_pd(one_minus_dy), _mm256_cvtps_pd(top)),
        _mm256_mul_pd(_mm256_cvtps_pd(dy), _mm256_cvtps_pd(bottom))));
}

template<bool Gray, class T>
WARP_AVX_ENTRY std::size_t affine_row_avx2(const Sampler<T>& sampler, const Matrix<T>& matrix,
                                         std::size_t row, std::size_t width, T* output) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    constexpr bool single = std::is_same_v<T, float>;
    constexpr std::size_t lanes = single ? 8 : 4;
    using Vector = typename AffineVector<T>::Type;
    const auto& p = matrix.p;
    const T row_x = p[1] * T(row), row_y = p[4] * T(row);
    Vector slope_x, slope_y, offset_x, offset_y, shift_x, shift_y, one;
    if constexpr (single) {
        slope_x = _mm256_set1_ps(p[0]); slope_y = _mm256_set1_ps(p[3]);
        offset_x = _mm256_set1_ps(row_x); offset_y = _mm256_set1_ps(row_y);
        shift_x = _mm256_set1_ps(p[2]); shift_y = _mm256_set1_ps(p[5]);
        one = _mm256_set1_ps(1);
    } else {
        slope_x = _mm256_set1_pd(p[0]); slope_y = _mm256_set1_pd(p[3]);
        offset_x = _mm256_set1_pd(row_x); offset_y = _mm256_set1_pd(row_y);
        shift_x = _mm256_set1_pd(p[2]); shift_y = _mm256_set1_pd(p[5]);
        one = _mm256_set1_pd(1);
    }
    const auto zero = _mm256_setzero_si256();
    const auto last_col = _mm256_set1_epi32(int(sampler.shape.cols - 1));
    const auto last_row = _mm256_set1_epi32(int(sampler.shape.rows - 1));
    const auto channels = Gray ? std::size_t(1) : sampler.shape.channels;
    const auto pixel_stride = _mm256_set1_epi32(int(channels));
    const auto row_stride = _mm256_set1_epi32(int(sampler.shape.cols * channels));
    const bool nearest = sampler.border == GaussianBorder::nearest;
    const auto step = _mm256_set1_epi32(int(lanes));
    auto columns = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    std::size_t c = 0;
    for (; c + lanes <= width; c += lanes, columns = _mm256_add_epi32(columns, step)) {
        // Step the integer output coordinates. Keep the original affine
        // operation order so rounding cannot accumulate across the row.
        Vector x, y, dx, dy, left_weight, top_weight;
        __m256i left, right, top, bottom;
        if constexpr (single) {
            const auto cx = _mm256_cvtepi32_ps(columns);
            x = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(slope_x, cx), offset_x), shift_x);
            y = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(slope_y, cx), offset_y), shift_y);
            left = _mm256_cvttps_epi32(_mm256_floor_ps(x));
            right = _mm256_cvttps_epi32(_mm256_ceil_ps(x));
            top = _mm256_cvttps_epi32(_mm256_floor_ps(y));
            bottom = _mm256_cvttps_epi32(_mm256_ceil_ps(y));
            dx = _mm256_sub_ps(x, _mm256_cvtepi32_ps(left));
            dy = _mm256_sub_ps(y, _mm256_cvtepi32_ps(top));
            left_weight = _mm256_sub_ps(one, dx);
            top_weight = _mm256_sub_ps(one, dy);
        } else {
            const auto cx = _mm256_cvtepi32_pd(_mm256_castsi256_si128(columns));
            x = _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(slope_x, cx), offset_x), shift_x);
            y = _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(slope_y, cx), offset_y), shift_y);
            left = _mm256_zextsi128_si256(_mm256_cvttpd_epi32(_mm256_floor_pd(x)));
            right = _mm256_zextsi128_si256(_mm256_cvttpd_epi32(_mm256_ceil_pd(x)));
            top = _mm256_zextsi128_si256(_mm256_cvttpd_epi32(_mm256_floor_pd(y)));
            bottom = _mm256_zextsi128_si256(_mm256_cvttpd_epi32(_mm256_ceil_pd(y)));
            dx = _mm256_sub_pd(x, _mm256_cvtepi32_pd(_mm256_castsi256_si128(left)));
            dy = _mm256_sub_pd(y, _mm256_cvtepi32_pd(_mm256_castsi256_si128(top)));
            left_weight = _mm256_sub_pd(one, dx);
            top_weight = _mm256_sub_pd(one, dy);
        }
        if (nearest) {
            left = _mm256_min_epi32(last_col, _mm256_max_epi32(zero, left));
            right = _mm256_min_epi32(last_col, _mm256_max_epi32(zero, right));
            top = _mm256_min_epi32(last_row, _mm256_max_epi32(zero, top));
            bottom = _mm256_min_epi32(last_row, _mm256_max_epi32(zero, bottom));
        } else {
            const auto outside = _mm256_or_si256(
                _mm256_or_si256(_mm256_cmpgt_epi32(zero, left), _mm256_cmpgt_epi32(right, last_col)),
                _mm256_or_si256(_mm256_cmpgt_epi32(zero, top), _mm256_cmpgt_epi32(bottom, last_row)));
            if (!_mm256_testz_si256(outside, outside)) {
                T xs[lanes], ys[lanes];
                if constexpr (single) {
                    _mm256_storeu_ps(xs, x); _mm256_storeu_ps(ys, y);
                } else {
                    _mm256_storeu_pd(xs, x); _mm256_storeu_pd(ys, y);
                }
                for (std::size_t lane = 0; lane < lanes; ++lane)
                    fast_pixel<1, Gray>(sampler, xs[lane], ys[lane], output + (c + lane) * channels);
                continue;
            }
        }
        if constexpr (!Gray) {
            left = _mm256_mullo_epi32(left, pixel_stride);
            right = _mm256_mullo_epi32(right, pixel_stride);
        }
        top = _mm256_mullo_epi32(top, row_stride);
        bottom = _mm256_mullo_epi32(bottom, row_stride);
        const auto tl = _mm256_add_epi32(top, left), tr = _mm256_add_epi32(top, right);
        const auto bl = _mm256_add_epi32(bottom, left), br = _mm256_add_epi32(bottom, right);
        for (std::size_t k = 0; k < channels; ++k) {
            const T* source = sampler.input + k;
            T scratch[lanes];
            T* dest = Gray ? output + c : scratch;
            if constexpr (single) {
                const auto a = _mm256_add_ps(_mm256_mul_ps(left_weight, _mm256_i32gather_ps(source, tl, 4)),
                                             _mm256_mul_ps(dx, _mm256_i32gather_ps(source, tr, 4)));
                const auto b = _mm256_add_ps(_mm256_mul_ps(left_weight, _mm256_i32gather_ps(source, bl, 4)),
                                             _mm256_mul_ps(dx, _mm256_i32gather_ps(source, br, 4)));
                _mm_storeu_ps(dest, bilinear_vertical(_mm256_castps256_ps128(a), _mm256_castps256_ps128(b),
                    _mm256_castps256_ps128(dy), _mm256_castps256_ps128(top_weight)));
                _mm_storeu_ps(dest + 4, bilinear_vertical(_mm256_extractf128_ps(a, 1), _mm256_extractf128_ps(b, 1),
                    _mm256_extractf128_ps(dy, 1), _mm256_extractf128_ps(top_weight, 1)));
            } else {
                const auto a = _mm256_add_pd(
                    _mm256_mul_pd(left_weight, _mm256_i32gather_pd(source, _mm256_castsi256_si128(tl), 8)),
                    _mm256_mul_pd(dx, _mm256_i32gather_pd(source, _mm256_castsi256_si128(tr), 8)));
                const auto b = _mm256_add_pd(
                    _mm256_mul_pd(left_weight, _mm256_i32gather_pd(source, _mm256_castsi256_si128(bl), 8)),
                    _mm256_mul_pd(dx, _mm256_i32gather_pd(source, _mm256_castsi256_si128(br), 8)));
                _mm256_storeu_pd(dest, _mm256_add_pd(_mm256_mul_pd(top_weight, a), _mm256_mul_pd(dy, b)));
            }
            if constexpr (!Gray)
                for (std::size_t lane = 0; lane < lanes; ++lane)
                    output[(c + lane) * channels + k] = scratch[lane];
        }
    }
    _mm256_zeroupper();
    return c;
}

#undef WARP_AVX_ENTRY
#undef WARP_AVX_INLINE
#if defined(_MSC_VER) && !defined(__clang__)
#pragma float_control(pop)
#pragma warning(pop)
#endif
#endif

template<int Order, class T>
bool fast(const T* input, T* output, WarpShape source, WarpShape target,
          const WarpMap& map, const WarpOptions& options) {
    const Matrix<T> matrix(map);
    if (!matrix.bounded(target)) return false;
    const Sampler<T> sampler{input, source, options.border, T(options.cval)};
#ifdef SKIMAGEXX_AVX2
    constexpr auto gather_limit = std::size_t(std::numeric_limits<int>::max());
    constexpr std::size_t lanes = std::is_same_v<T, float> ? 8 : 4;
    const bool vectorized = Order == 1 && !matrix.projective &&
                            target.cols >= lanes && target.cols <= gather_limit &&
                            count(source) <= gather_limit && has_avx2();
#endif
    const auto run = [&](auto gray) {
        const auto channels = decltype(gray)::value ? std::size_t(1) : target.channels;
        rows(target, [&](std::size_t r) {
            T* dest = output + r * target.cols * channels;
            std::size_t c = 0;
#ifdef SKIMAGEXX_AVX2
            if (vectorized)
                c = affine_row_avx2<decltype(gray)::value>(sampler, matrix, r, target.cols, dest);
#endif
            for (dest += c * channels; c < target.cols; ++c, dest += channels) {
                const auto xy = matrix(T(c), T(r));
                fast_pixel<Order, decltype(gray)::value>(sampler, xy[0], xy[1], dest);
            }
        });
    };
    if (source.channels == 1) run(std::true_type{});
    else run(std::false_type{});
    return true;
}

template<class T, bool Boolean>
bool clip_output(const T* input, T* output, std::size_t n, std::size_t m,
                 const WarpOptions& options) {
    if (!options.clip) return true;
    T lo = std::numeric_limits<T>::max(), hi = std::numeric_limits<T>::lowest();
    if constexpr (std::is_floating_point_v<T>) {
        lo = std::numeric_limits<T>::infinity();
        hi = -lo;
    }
    bool any = false, input_nan = false;
#ifdef SKIMAGEXX_AVX2
    const bool vectorized = has_avx2();
    if (vectorized) {
        double minimum, maximum;
        simd::bounds(input, n, minimum, maximum);
        if (!std::isnan(minimum)) {
            lo = T(minimum);
            hi = T(maximum);
            any = true;
        }
    }
#endif
    if (!any) {
        for (std::size_t i = 0; i < n; ++i) {
            const auto value = input[i];
            if constexpr (std::is_floating_point_v<T>) {
                if (std::isnan(value)) {
                    input_nan = true;
                    continue;
                }
            }
            lo = std::min(lo, value);
            hi = std::max(hi, value);
            any = true;
        }
    }
    if (!any) return false;
    double fill = options.cval;
    if constexpr (std::is_same_v<T, float>) {
        if (options.round_clip_cval) fill = double(float(fill));
    }
    if (options.border == GaussianBorder::constant && (fill < lo || fill > hi)) {
        bool below = false, above = false, nan = false;
        if (input_nan) {
            for (std::size_t i = 0; i < m && !(below && above); ++i) {
                below |= output[i] <= fill;
                above |= output[i] >= fill;
            }
        } else {
#ifdef SKIMAGEXX_AVX2
            if (vectorized) {
                double minimum, maximum;
                simd::bounds(output, m, minimum, maximum);
                below = minimum <= fill;
                above = maximum >= fill;
            } else
#endif
            {
                for (std::size_t i = 0; i < m && !nan; ++i) {
                    below |= output[i] <= fill;
                    above |= output[i] >= fill;
                    if constexpr (std::is_floating_point_v<T>) nan = std::isnan(output[i]);
                }
            }
        }
        if (below && above && !nan) {
            const T cast = Boolean ? T(fill != 0) : detail::resize_raw_cast<T>(fill);
            lo = std::min(lo, cast);
            hi = std::max(hi, cast);
        }
    }
    std::size_t i = 0;
#ifdef SKIMAGEXX_AVX2
    if (vectorized) {
        constexpr auto type = std::is_same_v<T, float> ? PixelType::float32 :
                              std::is_same_v<T, double> ? PixelType::float64 : PixelType::uint8;
        i = simd::resize_clip(output, m, type, &lo, &hi, false);
    }
#endif
    for (; i < m; ++i) output[i] = std::clamp(output[i], lo, hi);
    return true;
}

struct Axis {
    std::array<Index, 4> index;
    std::array<double, 6> weight;
};

template<int Order>
Axis spline_axis(double coord, Index length, GaussianBorder border) {
    coord = extend(coord, length, border);
    // A whole footprint outside grid-constant storage has only fill taps.
    coord = std::clamp(coord, -8., double(length) + 8.);
    const Index origin = floor_index(coord + (Order & 1 ? 0. : .5));
    const Index first = origin - Order / 2;
    Axis axis{};
    if constexpr (Order == 1) {
        axis.weight[0] = 1. - (coord - double(origin));
        axis.weight[1] = 1. - axis.weight[0];
    } else if constexpr (Order > 1) {
        axis.weight = detail::spline_weights(coord, Order);
    }
    for (int i = 0; i <= Order; ++i) axis.index[i] = tap_index(first + i, length, border);
    return axis;
}

template<int Order, class Input, class Output, bool Boolean>
void generic_pixel(const Sampler<Input>& sampler, double x, double y,
                   double fill, Output* output) {
    if constexpr (Order <= 1) {
        if (x >= 0. && y >= 0. && x < double(sampler.shape.cols - 1) &&
            y < double(sampler.shape.rows - 1)) {
            const Index row = Index(y + (Order == 0 ? .5 : 0.));
            const Index col = Index(x + (Order == 0 ? .5 : 0.));
            if constexpr (Order == 0) {
                const auto* at = sampler.input + (std::size_t(row) * sampler.shape.cols +
                    std::size_t(col)) * sampler.shape.channels;
                std::copy_n(at, sampler.shape.channels, output);
            } else {
                const double top = 1. - (y - double(row)), bottom = 1. - top;
                const double left = 1. - (x - double(col)), right = 1. - left;
                for (std::size_t k = 0; k < sampler.shape.channels; ++k) {
                    double value = 0;
                    value += double(sampler.direct(row, col, k)) * top * left;
                    value += double(sampler.direct(row, col + 1, k)) * top * right;
                    value += double(sampler.direct(row + 1, col, k)) * bottom * left;
                    value += double(sampler.direct(row + 1, col + 1, k)) * bottom * right;
                    output[k] = Output(value);
                }
            }
            return;
        }
    }
    // The map's bound is checked before evaluation. These expressions also
    // handle a grid-constant coordinate outside either padded image axis.
    const auto c = spline_axis<Order>(x, Index(sampler.shape.cols), sampler.border);
    const auto r = spline_axis<Order>(y, Index(sampler.shape.rows), sampler.border);
    for (std::size_t k = 0; k < sampler.shape.channels; ++k) {
        double value = 0;
        for (int i = 0; i <= Order; ++i) {
            for (int j = 0; j <= Order; ++j) {
                double tap = r.index[i] < 0 || c.index[j] < 0
                    ? fill : double(sampler.direct(r.index[i], c.index[j], k));
                if constexpr (Order > 0) tap = tap * r.weight[i] * c.weight[j];
                value += tap;
            }
        }
        if constexpr (Boolean) output[k] = Output(detail::resize_sample_cast<Output, true>(value) != 0);
        else output[k] = detail::resize_sample_cast<Output>(value);
    }
}

// Hoist the polar angle and radius calculations out of the pixel loop.
// The arithmetic order within each map matches its public counterpart.
template<bool Nearest = false, class Render>
bool mapped_rows(WarpShape target, const WarpMap& map, Render render) {
    const auto& p = map.params;
    if (map.kind == WarpMapping::matrix) {
        const Matrix<double> matrix(map);
        if (!matrix.bounded(target)) return false;
        if constexpr (Nearest) {
            if (map.homogeneous) {
                std::atomic<bool> valid{true};
                rows(target, [&](std::size_t first) {
                    for (std::size_t c = 0; c < target.cols; ++c) {
                        const auto* h = map.homogeneous + 3 * (c * target.rows + first);
                        const double w = h[2] == 0 ? std::numeric_limits<double>::epsilon() : h[2];
                        const double x = h[0] / w, y = h[1] / w;
                        if (std::abs(x) < 1e9 && std::abs(y) < 1e9) render(first, c, x, y);
                        else valid.store(false, std::memory_order_relaxed);
                    }
                });
                return valid.load(std::memory_order_relaxed);
            }
        }
        rows(target, [&](std::size_t r) {
            for (std::size_t c = 0; c < target.cols; ++c) {
                const auto xy = matrix(double(c), double(r));
                render(r, c, xy[0], xy[1]);
            }
        });
        return true;
    }
    if (map.kind == WarpMapping::swirl) {
        if (p[4] <= 0 || std::abs(p[0]) + std::abs(p[1]) + double(target.rows) +
                         double(target.cols) >= 1e8) return false;
        rows(target, [&](std::size_t r) {
            const double y = double(r) - p[1];
            for (std::size_t c = 0; c < target.cols; ++c) {
                const double x = double(c) - p[0];
                const double rho = std::sqrt(x*x + y*y);
                const double turn = p[2] + p[3] * std::exp(-rho/p[4]);
                if constexpr (!Nearest) {
                    if (std::abs(turn) <= 32.) {
                        const double sine = std::sin(turn), cosine = std::cos(turn);
                        render(r, c, p[0] + x * cosine - y * sine, p[1] + x * sine + y * cosine);
                        continue;
                    }
                }
                const double theta = turn + std::atan2(y, x);
                render(r, c, p[0] + rho * std::cos(theta), p[1] + rho * std::sin(theta));
            }
        });
        return true;
    }
    if (p[2] <= 0 || p[3] == 0 || std::abs(p[0]) + std::abs(p[1]) >= 1e8 ||
        !std::isfinite(double(target.rows) / p[2])) return false;
    const auto radius = [&](double c) {
        const double x = c / p[3];
        return map.kind == WarpMapping::polar_log ? std::exp(x) : x;
    };
    if (!(std::abs(radius(double(target.cols - 1))) < 1e8)) return false;
    std::vector<double> radii(target.cols);
    for (std::size_t i = 0; i < radii.size(); ++i) radii[i] = radius(double(i));
    rows(target, [&](std::size_t r) {
        const double angle = double(r) / p[2];
        const double sine = std::sin(angle), cosine = std::cos(angle);
        for (std::size_t c = 0; c < target.cols; ++c)
            render(r, c, radii[c] * cosine + p[0], radii[c] * sine + p[1]);
    });
    return true;
}

template<int Order, class Input, class Output, bool Boolean = false>
bool generic(const Input* input, Output* output, WarpShape source, WarpShape target,
             const WarpMap& map, const WarpOptions& options) {
    auto render = [&](auto sampler, double fill, double padding) {
        return mapped_rows<Order == 0>(target, map, [&](std::size_t r, std::size_t c, double x, double y) {
            generic_pixel<Order, std::remove_const_t<std::remove_pointer_t<decltype(sampler.input)>>,
                          Output, Boolean>(sampler, x + padding, y + padding, fill,
                              output + (r * target.cols + c) * target.channels);
        });
    };
    if constexpr (Order <= 1) {
        return render(Sampler<Input>{input, source, options.border, Input{}}, options.cval, 0.);
    } else {
        const auto border = options.border;
        const std::size_t pad = border == GaussianBorder::nearest ||
                                border == GaussianBorder::constant ? 12 : 0;
        const WarpShape padded{source.rows + 2 * pad, source.cols + 2 * pad};
        std::vector<double> coefficients(count(padded));
        const double fill = double(Input(options.cval));
        const Sampler<Input> original{input, source, border, Input(options.cval)};
        rows(padded, [&](std::size_t r) {
            for (std::size_t c = 0; c < padded.cols; ++c)
                coefficients[r * padded.cols + c] = original.at(Index(r) - Index(pad),
                                                               Index(c) - Index(pad), 0);
        });
        detail::spline_filter_into(coefficients.data(), coefficients.data(),
                                   {padded.rows, padded.cols}, Order, border, options.legacy_reflect);
        return render(Sampler<double>{coefficients.data(), padded, border, fill},
                      options.cval, double(pad));
    }
}

template<class T, bool Boolean = false>
bool warp_typed(const T* input, T* output, WarpShape source, WarpShape target,
                const WarpMap& map, const WarpOptions& options) {
    bool done;
    if constexpr (std::is_floating_point_v<T>) {
        if (map.kind == WarpMapping::matrix && options.order == 1)
            done = fast<1>(input, output, source, target, map, options);
        else if (map.kind == WarpMapping::matrix && options.order == 3)
            done = fast<3>(input, output, source, target, map, options);
        else if (options.order == 1)
            done = generic<1>(input, output, source, target, map, options);
        else if (options.order == 2)
            done = generic<2>(input, output, source, target, map, options);
        else if (options.order == 3)
            done = generic<3>(input, output, source, target, map, options);
        else done = generic<0, T, T, Boolean>(input, output, source, target, map, options);
    } else done = generic<0, T, T, Boolean>(input, output, source, target, map, options);
    return done && clip_output<T, Boolean>(input, output, count(source), count(target), options);
}
} // namespace

bool warp_into(const void* input, void* output, PixelType type, WarpShape source,
               WarpShape target, const WarpMap& map, const WarpOptions& options) {
    if (!input || !output || input == output || source.channels != target.channels ||
        options.order < 0 || options.order > 3 || !std::isfinite(options.cval))
        throw std::invalid_argument("invalid warp storage or options");
    const auto n = count(source);
    const auto m = count(target);
    if (map.homogeneous && m / target.channels > max_count / 3) return false;
    if ((options.order > 1 && source.channels != 1 &&
         !(options.order == 3 && map.kind == WarpMapping::matrix)) ||
        !std::all_of(map.params.begin(), map.params.end(), [](double x) { return std::isfinite(x); }))
        return false;
    switch (type) {
        case PixelType::float32:
            if (std::abs(options.cval) > std::numeric_limits<float>::max()) return false;
            return warp_typed(static_cast<const float*>(input), static_cast<float*>(output),
                              source, target, map, options);
        case PixelType::float64:
            return warp_typed(static_cast<const double*>(input), static_cast<double*>(output),
                              source, target, map, options);
        case PixelType::boolean:
            if (options.order != 0) return false;
            return warp_typed<std::uint8_t, true>(static_cast<const std::uint8_t*>(input),
                static_cast<std::uint8_t*>(output), source, target, map, options);
        case PixelType::uint8: {
            const auto* data = static_cast<const std::uint8_t*>(input);
            if (options.order == 0)
                return warp_typed(data, static_cast<std::uint8_t*>(output), source, target, map, options);
            std::vector<double> converted(n);
            const double scale = options.preserve_range ? 1. : 1. / 255.;
            for (std::size_t i = 0; i < n; ++i) converted[i] = data[i] * scale;
            return warp_typed(converted.data(), static_cast<double*>(output),
                              source, target, map, options);
        }
        default: return false;
    }
}

template<class T>
bool warp_coords_into(T* output, WarpShape shape, bool color, const WarpMap& map) {
    const auto plane = count(shape);
    if (!output || (!color && shape.channels != 1) || plane > max_count / 3)
        throw std::invalid_argument("invalid warp coordinate storage");
    return mapped_rows(shape, map, [&](std::size_t r, std::size_t c, double x, double y) {
        const auto i = (r * shape.cols + c) * shape.channels;
        for (std::size_t k = 0; k < shape.channels; ++k) {
            output[i + k] = T(y);
            output[plane + i + k] = T(x);
            if (color) output[2 * plane + i + k] = T(k);
        }
    });
}
template bool warp_coords_into(float*, WarpShape, bool, const WarpMap&);
template bool warp_coords_into(double*, WarpShape, bool, const WarpMap&);
}
