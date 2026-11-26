#include <skimagexx/restoration_extra.hpp>
#include <skimagexx/image.hpp>
#include "fft_internal.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iterator>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <type_traits>

#if defined(SKIMAGEXX_AVX2) && defined(__AVX2__)
#include <immintrin.h>
#define RESTORATION_AVX2
#define RESTORATION_AVX_TARGET
#elif defined(SKIMAGEXX_AVX2) && (defined(_M_X64) || defined(__SSE2__))
#include <emmintrin.h>
#define RESTORATION_SSE2
#endif

namespace skimagexx {
namespace {
std::size_t size_of(const RestorationShape& shape) {
    if (shape.empty()) throw std::invalid_argument("restoration requires a nonempty shape");
    std::size_t size = 1;
    for (const auto extent : shape) {
        if (extent == 0 || size > std::size_t(std::numeric_limits<std::ptrdiff_t>::max()) / extent)
            throw std::invalid_argument("invalid restoration shape extent");
        size *= extent;
    }
    return size;
}

RestorationShape axes_of(const RestorationShape& shape) {
    RestorationShape axes(shape.size());
    std::iota(axes.begin(), axes.end(), std::size_t{0});
    return axes;
}

RestorationShape element_strides(const RestorationShape& shape) {
    RestorationShape strides(shape.size(), 1);
    for (auto axis = shape.size() - 1; axis > 0; --axis)
        strides[axis - 1] = strides[axis] * shape[axis];
    return strides;
}

// Advance the source coordinates of a contiguous row without dividing by any
// dimension. The final axis is handled as contiguous spans by the caller.
void advance_row(RestorationShape& coordinates, const RestorationShape& shape) {
    for (auto axis = coordinates.size(); axis-- > 0;) {
        if (++coordinates[axis] < shape[axis]) break;
        coordinates[axis] = 0;
    }
}

#ifdef RESTORATION_AVX2
template<class T>
RESTORATION_AVX_TARGET std::size_t add_span_simd(const T* input, T* output, std::size_t size) {
    constexpr std::size_t lanes = 32 / sizeof(T);
    std::size_t i = 0;
    for (; size - i >= lanes; i += lanes) {
        if constexpr (std::is_same_v<T, float>) {
            _mm256_storeu_ps(output + i, _mm256_add_ps(_mm256_loadu_ps(output + i),
                                                      _mm256_loadu_ps(input + i)));
        } else {
            _mm256_storeu_pd(output + i, _mm256_add_pd(_mm256_loadu_pd(output + i),
                                                      _mm256_loadu_pd(input + i)));
        }
    }
    return i;
}

template<class T>
RESTORATION_AVX_TARGET __m256d interpolate_load_avx2(const T* p) {
    if constexpr (std::is_same_v<T, float>) return _mm256_cvtps_pd(_mm_loadu_ps(p));
    else return _mm256_loadu_pd(p);
}

template<class T>
RESTORATION_AVX_TARGET std::size_t interpolate_span_simd(const T* up, const T* left,
    const T* right, const T* down, T* output, std::size_t size) {
    const auto weight = _mm256_set1_pd(0.25);
    std::size_t i = 0;
    for (; size - i >= 4; i += 4) {
        auto sum = _mm256_add_pd(_mm256_setzero_pd(),
                                _mm256_mul_pd(interpolate_load_avx2(up + i), weight));
        sum = _mm256_add_pd(sum, _mm256_mul_pd(interpolate_load_avx2(left + i), weight));
        sum = _mm256_add_pd(sum, _mm256_mul_pd(interpolate_load_avx2(right + i), weight));
        sum = _mm256_add_pd(sum, _mm256_mul_pd(interpolate_load_avx2(down + i), weight));
        if constexpr (std::is_same_v<T, float>) _mm_storeu_ps(output + i, _mm256_cvtpd_ps(sum));
        else _mm256_storeu_pd(output + i, sum);
    }
    return i;
}
#elif defined(RESTORATION_SSE2)
template<class T>
std::size_t add_span_simd(const T* input, T* output, std::size_t size) {
    constexpr std::size_t lanes = 16 / sizeof(T);
    std::size_t i = 0;
    for (; size - i >= lanes; i += lanes) {
        if constexpr (std::is_same_v<T, float>) {
            _mm_storeu_ps(output + i, _mm_add_ps(_mm_loadu_ps(output + i),
                                                 _mm_loadu_ps(input + i)));
        } else {
            _mm_storeu_pd(output + i, _mm_add_pd(_mm_loadu_pd(output + i),
                                                 _mm_loadu_pd(input + i)));
        }
    }
    return i;
}

std::size_t interpolate_span_simd(const double* up, const double* left, const double* right,
                                   const double* down, double* output, std::size_t size) {
    const auto weight = _mm_set1_pd(0.25);
    std::size_t i = 0;
    for (; size - i >= 2; i += 2) {
        auto sum = _mm_add_pd(_mm_setzero_pd(), _mm_mul_pd(_mm_loadu_pd(up + i), weight));
        sum = _mm_add_pd(sum, _mm_mul_pd(_mm_loadu_pd(left + i), weight));
        sum = _mm_add_pd(sum, _mm_mul_pd(_mm_loadu_pd(right + i), weight));
        sum = _mm_add_pd(sum, _mm_mul_pd(_mm_loadu_pd(down + i), weight));
        _mm_storeu_pd(output + i, sum);
    }
    return i;
}
#endif

template<class T>
T interpolate_pixel(T up, T left, T right, T down) {
    double sum = 0;
    sum += double(up) * 0.25;
    sum += double(left) * 0.25;
    sum += double(right) * 0.25;
    sum += double(down) * 0.25;
    return T(sum);
}

template<class T>
void interpolate_2d(const T* image, const RestorationShape& shape,
                    std::size_t block, T* output) {
    const auto width = shape[1], row_size = width * block;
#if defined(RESTORATION_AVX2) || defined(RESTORATION_SSE2)
    const bool vectorize = has_avx2();
#endif
    for (std::size_t y = 0; y < shape[0]; ++y) {
        const auto* row = image + y * row_size;
        const auto* up = y == 0 ? row + (shape[0] > 1 ? row_size : 0) : row - row_size;
        const auto* down = y + 1 == shape[0] ? row - (shape[0] > 1 ? row_size : 0) : row + row_size;
        const auto span = [&](std::size_t start, std::size_t left, std::size_t right,
                              std::size_t count) {
            auto* out = output + y * row_size + start;
            std::size_t i = 0;
#ifdef RESTORATION_AVX2
            if (vectorize)
                i = interpolate_span_simd(up + start, row + left, row + right, down + start, out, count);
#elif defined(RESTORATION_SSE2)
            if constexpr (std::is_same_v<T, double>) {
                if (vectorize)
                    i = interpolate_span_simd(up + start, row + left, row + right, down + start, out, count);
            }
#endif
            for (; i < count; ++i)
                out[i] = interpolate_pixel(up[start + i], row[left + i], row[right + i], down[start + i]);
        };
        if (width == 1) span(0, 0, 0, block);
        else {
            span(0, block, block, block);
            span(block, 0, 2 * block, row_size - 2 * block);
            span(row_size - block, row_size - 2 * block, row_size - 2 * block, block);
        }
    }
}

template<class T>
void interpolate_phase_2d(const T* image, std::size_t height, std::size_t width,
                          std::array<std::size_t, 2> phase, std::size_t stride, T* output) {
    for (auto y = phase[0]; y < height; y += stride) {
        const auto* row = image + y * width;
        const auto* up = height == 1 ? row : y == 0 ? row + width : row - width;
        const auto* down = height == 1 ? row : y + 1 == height ? row - width : row + width;
        for (auto x = phase[1]; x < width; x += stride) {
            const auto left = width == 1 ? x : x == 0 ? x + 1 : x - 1;
            const auto right = width == 1 ? x : x + 1 == width ? x - 1 : x + 1;
            *output++ = interpolate_pixel(up[x], row[left], row[right], down[x]);
        }
    }
}
double laplacian_value(std::size_t frequency, std::size_t extent) {
    constexpr double pi = 3.1415926535897932384626433832795;
    const auto sine = std::sin(pi * double(frequency) / double(extent));
    return 4 * sine * sine;
}

template<class Visit>
bool visit_laplacian(const RestorationShape& shape, std::size_t count, Visit visit) {
    const auto width = shape.back() / 2 + 1;
    std::vector<double> last(width);
    for (std::size_t j = 0; j < width; ++j)
        last[j] = laplacian_value(j, shape.back());
    auto spectrum = shape;
    spectrum.back() = width;
    const auto strides = element_strides(spectrum);
    std::vector<std::vector<double>> eigenvalues(shape.size() - 1);
    for (std::size_t axis = 0; axis < eigenvalues.size(); ++axis) {
        auto& values = eigenvalues[axis];
        const auto extent = shape[axis];
        values.resize(extent);
        for (std::size_t j = 0; j < extent; ++j)
            values[j] = laplacian_value(std::min(j, extent - j), extent);
    }
    for (std::size_t i = 0; i < count; i += width) {
        double offset = 0;
        for (std::size_t axis = 0; axis < eigenvalues.size(); ++axis)
            offset += eigenvalues[axis][i / strides[axis] % spectrum[axis]];
        for (std::size_t j = 0; j < width; ++j)
            if (!visit(i + j, offset + last[j])) return false;
    }
    return true;
}
} // namespace

template<class T>
void restoration_transfer(const T* kernel, const RestorationShape& kernel_shape,
                          const RestorationShape& shape, std::complex<T>* output,
                          int workers) {
    const auto size = size_of(shape);
    const auto kernel_size = size_of(kernel_shape);
    if (!kernel || !output || kernel_shape.size() != shape.size())
        throw std::invalid_argument("invalid restoration kernel buffer or rank");
    for (std::size_t axis = 0; axis < shape.size(); ++axis)
        if (kernel_shape[axis] > shape[axis])
            throw std::invalid_argument("restoration kernel exceeds image shape");
    const auto strides = element_strides(shape);
    std::vector<T> padded(size);
    for (std::size_t i = 0; i < kernel_size; ++i) {
        auto remaining = i;
        std::size_t target = 0;
        for (auto axis = shape.size(); axis-- > 0;) {
            const auto coordinate = remaining % kernel_shape[axis];
            remaining /= kernel_shape[axis];
            const auto center = kernel_shape[axis] / 2;
            const auto wrapped = coordinate >= center
                ? coordinate - center : shape[axis] - (center - coordinate);
            target += wrapped * strides[axis];
        }
        padded[target] = kernel[i];
    }
    detail::fft_r2c(shape, axes_of(shape), padded.data(), output, workers);
}

template<class T>
void restoration_laplacian(const RestorationShape& shape, T* output) {
    const auto size = size_of(shape);
    if (!output) throw std::invalid_argument("null restoration Laplacian buffer");
    visit_laplacian(shape, size / shape.back() * (shape.back() / 2 + 1),
                    [output](std::size_t i, double value) {
        output[i] = T(value);
        return true;
    });
}

template<class T>
bool restoration_wiener(const T* image, const RestorationShape& shape,
                        const std::complex<T>* transfer, const T* reg,
                        T balance, bool clip, T* output, int workers) {
    const auto size = size_of(shape);
    if (!image || !output || !transfer)
        throw std::invalid_argument("null restoration Wiener buffer");
    const auto count = size / shape.back() * (shape.back() / 2 + 1);
    std::vector<std::complex<T>> data(count);
    const auto axes = axes_of(shape);
    detail::fft_r2c(shape, axes, image, data.data(), workers);
    const auto apply = [&](std::size_t i, T regularizer) {
        const T re = transfer[i].real(), im = transfer[i].imag();
        const T denominator = (re * re + im * im) + balance * (regularizer * regularizer);
        if (denominator == 0 || !std::isfinite(denominator)) return false;
        const T fr = re / denominator, fi = -im / denominator;
        const auto value = data[i];
        data[i] = {fr * value.real() - fi * value.imag(), fr * value.imag() + fi * value.real()};
        return std::isfinite(data[i].real()) && std::isfinite(data[i].imag());
    };
    if (reg) {
        for (std::size_t i = 0; i < count; ++i)
            if (!apply(i, reg[i])) return false;
    } else if (!visit_laplacian(shape, count, [&](std::size_t i, double value) {
        return apply(i, T(value));
    })) {
        return false;
    }
    detail::fft_c2r(shape, axes, data.data(), output, workers);
    if (clip) {
        for (std::size_t i = 0; i < size; ++i)
            output[i] = std::clamp(output[i], T(-1), T(1));
    }
    return true;
}

template<class T>
void restoration_gibbs_sample(const std::complex<T>* data,
                              const std::complex<T>* weighted_transfer,
                              const T* precision, const double* normal_real,
                              const double* normal_imag, std::size_t count,
                              std::complex<T>* output) {
    if (!data || !weighted_transfer || !precision || !normal_real || !normal_imag || !output)
        throw std::invalid_argument("null restoration Gibbs buffer");
    for (std::size_t i = 0; i < count; ++i) {
        const T reciprocal = T(1) / precision[i];
        const T fr = weighted_transfer[i].real() * reciprocal;
        const T fi = weighted_transfer[i].imag() * reciprocal;
        const T re = data[i].real(), im = data[i].imag();
        const T scale = std::sqrt(T(0.5) / precision[i]);
        output[i] = {(fr * re - fi * im) + scale * T(normal_real[i]),
                     (fr * im + fi * re) + scale * T(normal_imag[i])};
    }
}

template<class T>
void restoration_gibbs_step(const std::complex<T>* data, const std::complex<T>* transfer,
                             const T* reg, const T* power, const T* reg_power,
                             T noise, T prior, const double* normals, std::size_t count,
                             std::complex<T>* sample, std::complex<T>* residuals) {
    if (!data || !transfer || !reg || !power || !reg_power || !normals || !sample || !residuals)
        throw std::invalid_argument("null restoration Gibbs buffer");
    for (std::size_t i = 0; i < count; ++i) {
        const T precision = noise * power[i] + prior * reg_power[i];
        const T reciprocal = T(1) / precision;
        const T hr = transfer[i].real(), hi = transfer[i].imag();
        const T fr = (noise * hr) * reciprocal;
        const T fi = (noise * -hi) * reciprocal;
        const T re = data[i].real(), im = data[i].imag();
        const T scale = std::sqrt(T(0.5) / precision);
        const T sr = (fr * re - fi * im) + scale * T(normals[i]);
        const T si = (fr * im + fi * re) + scale * T(normals[count + i]);
        sample[i] = {sr, si};
        residuals[i] = {re - (sr * hr - si * hi), im - (sr * hi + si * hr)};
        residuals[count + i] = {sr * reg[i], si * reg[i]};
    }
}

namespace {
struct GibbsSums {
    double noise = 0, prior = 0, change = 0, magnitude = 0;
};

template<class T>
struct GibbsStep {
    const std::complex<T>* data;
    const std::complex<T>* transfer;
    const T* reg;
    const T* power;
    const T* reg_power;
    const double* normal_real;
    const double* normal_imag;
    std::complex<T>* total;
    T noise, prior;
    T previous, divisor;
    bool accumulate;
    bool measure_delta;
};

template<class T>
void gibbs_scalar(const GibbsStep<T>& step, std::size_t i, GibbsSums& sums) {
    const auto [data, transfer, reg, power, reg_power, nr, ni, total, noise, prior,
                previous, divisor, accumulate, measure_delta] = step;
    const T precision = noise * power[i] + prior * reg_power[i];
    const T reciprocal = T(1) / precision;
    const T hr = transfer[i].real(), hi = transfer[i].imag();
    const T fr = (noise * hr) * reciprocal, fi = (noise * -hi) * reciprocal;
    const T re = data[i].real(), im = data[i].imag();
    // Multiplication by a power of two preserves the rounded quotient when
    // both results are normal. Keep the division near overflow and underflow.
    const T variance = reciprocal >= T(2) * std::numeric_limits<T>::min() &&
                       reciprocal <= std::numeric_limits<T>::max()
        ? T(0.5) * reciprocal : T(0.5) / precision;
    const T scale = std::sqrt(variance);
    const T sr = (fr * re - fi * im) + scale * T(nr[i]);
    const T si = (fr * im + fi * re) + scale * T(ni[i]);
    const T rr = re - (sr * hr - si * hi), ri = im - (sr * hi + si * hr);
    const T pr = sr * reg[i], pi = si * reg[i];
    sums.noise += rr * rr + ri * ri;
    sums.prior += pr * pr + pi * pi;
    if (accumulate) {
        const auto old = total[i];
        total[i] = {old.real() + sr, old.imag() + si};
        if (measure_delta) {
            const T dr = (previous * sr - old.real()) / divisor;
            const T di = (previous * si - old.imag()) / divisor;
            sums.change += std::hypot(dr, di);
            sums.magnitude += std::abs(total[i]);
        }
    }
}

#if defined(RESTORATION_AVX2) || defined(RESTORATION_SSE2)
template<class T> struct GibbsVector;

#ifdef RESTORATION_AVX2
#define GIBBS_NORMAL_RECIPROCAL(SUFFIX, VALUE, LOW, HIGH) \
    _mm256_movemask_##SUFFIX(_mm256_and_##SUFFIX( \
        _mm256_cmp_##SUFFIX(VALUE, LOW, _CMP_GE_OQ), _mm256_cmp_##SUFFIX(VALUE, HIGH, _CMP_LE_OQ)))
#else
#define GIBBS_NORMAL_RECIPROCAL(SUFFIX, VALUE, LOW, HIGH) \
    _mm_movemask_##SUFFIX(_mm_and_##SUFFIX(_mm_cmpge_##SUFFIX(VALUE, LOW), _mm_cmple_##SUFFIX(VALUE, HIGH)))
#endif
#define GIBBS_VECTOR_OPS(PREFIX, SUFFIX, LANES) \
    static constexpr std::size_t lanes = LANES; \
    static V set(T x) { return PREFIX##_set1_##SUFFIX(x); } \
    static V load(const T* p) { return PREFIX##_loadu_##SUFFIX(p); } \
    static V add(V a, V b) { return PREFIX##_add_##SUFFIX(a, b); } \
    static V sub(V a, V b) { return PREFIX##_sub_##SUFFIX(a, b); } \
    static V mul(V a, V b) { return PREFIX##_mul_##SUFFIX(a, b); } \
    static V div(V a, V b) { return PREFIX##_div_##SUFFIX(a, b); } \
    static V sqrt(V a) { return PREFIX##_sqrt_##SUFFIX(a); } \
    static V variance(V reciprocal, V precision, V half) { \
        const auto low = set(T(2) * std::numeric_limits<T>::min()); \
        const auto high = set(std::numeric_limits<T>::max()); \
        if (GIBBS_NORMAL_RECIPROCAL(SUFFIX, reciprocal, low, high) == (1 << lanes) - 1) \
            return mul(half, reciprocal); \
        return div(half, precision); \
    } \
    static V norm2(V re, V im) { return add(mul(re, re), mul(im, im)); } \
    static double sum(V a) { \
        T values[lanes]; \
        PREFIX##_storeu_##SUFFIX(values, a); \
        return std::accumulate(std::begin(values), std::end(values), 0.0); \
    }

#ifdef RESTORATION_AVX2
template<> struct GibbsVector<double> {
    using T = double;
    using V = __m256d;
    GIBBS_VECTOR_OPS(_mm256, pd, 4)

    static RESTORATION_AVX_TARGET V normals(const double* p) { return load(p); }
    static RESTORATION_AVX_TARGET void complex_load(const std::complex<T>* p, V& re, V& im) {
        const auto* values = reinterpret_cast<const T*>(p);
        const auto lo = load(values), hi = load(values + 4);
        re = _mm256_permute4x64_pd(_mm256_unpacklo_pd(lo, hi), 0xd8);
        im = _mm256_permute4x64_pd(_mm256_unpackhi_pd(lo, hi), 0xd8);
    }
    static RESTORATION_AVX_TARGET void complex_store(std::complex<T>* p, V re, V im) {
        auto* values = reinterpret_cast<T*>(p);
        const auto lo = _mm256_unpacklo_pd(re, im), hi = _mm256_unpackhi_pd(re, im);
        _mm256_storeu_pd(values, _mm256_permute2f128_pd(lo, hi, 0x20));
        _mm256_storeu_pd(values + 4, _mm256_permute2f128_pd(lo, hi, 0x31));
    }
};

template<> struct GibbsVector<float> {
    using T = float;
    using V = __m256;
    GIBBS_VECTOR_OPS(_mm256, ps, 8)

    static RESTORATION_AVX_TARGET V normals(const double* p) {
        return _mm256_set_m128(_mm256_cvtpd_ps(_mm256_loadu_pd(p + 4)),
                              _mm256_cvtpd_ps(_mm256_loadu_pd(p)));
    }
    static RESTORATION_AVX_TARGET void complex_load(const std::complex<T>* p, V& re, V& im) {
        const auto* values = reinterpret_cast<const T*>(p);
        const auto lo = load(values), hi = load(values + 8);
        const auto order = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);
        re = _mm256_permutevar8x32_ps(_mm256_shuffle_ps(lo, hi, 0x88), order);
        im = _mm256_permutevar8x32_ps(_mm256_shuffle_ps(lo, hi, 0xdd), order);
    }
    static RESTORATION_AVX_TARGET void complex_store(std::complex<T>* p, V re, V im) {
        auto* values = reinterpret_cast<T*>(p);
        const auto lo = _mm256_unpacklo_ps(re, im), hi = _mm256_unpackhi_ps(re, im);
        _mm256_storeu_ps(values, _mm256_permute2f128_ps(lo, hi, 0x20));
        _mm256_storeu_ps(values + 8, _mm256_permute2f128_ps(lo, hi, 0x31));
    }
};
#else
// The generic x64 TU has baseline SSE2 even when it has no AVX2 compiler
// target. Use the same runtime switch so disabling SIMD retains scalar math.
template<> struct GibbsVector<double> {
    using T = double;
    using V = __m128d;
    GIBBS_VECTOR_OPS(_mm, pd, 2)

    static V normals(const double* p) { return load(p); }
    static void complex_load(const std::complex<T>* p, V& re, V& im) {
        const auto* values = reinterpret_cast<const T*>(p);
        const auto lo = load(values), hi = load(values + 2);
        re = _mm_unpacklo_pd(lo, hi);
        im = _mm_unpackhi_pd(lo, hi);
    }
    static void complex_store(std::complex<T>* p, V re, V im) {
        auto* values = reinterpret_cast<T*>(p);
        _mm_storeu_pd(values, _mm_unpacklo_pd(re, im));
        _mm_storeu_pd(values + 2, _mm_unpackhi_pd(re, im));
    }
};

template<> struct GibbsVector<float> {
    using T = float;
    using V = __m128;
    GIBBS_VECTOR_OPS(_mm, ps, 4)

    static V normals(const double* p) {
        return _mm_movelh_ps(_mm_cvtpd_ps(_mm_loadu_pd(p)), _mm_cvtpd_ps(_mm_loadu_pd(p + 2)));
    }
    static void complex_load(const std::complex<T>* p, V& re, V& im) {
        const auto* values = reinterpret_cast<const T*>(p);
        const auto lo = load(values), hi = load(values + 4);
        re = _mm_shuffle_ps(lo, hi, 0x88);
        im = _mm_shuffle_ps(lo, hi, 0xdd);
    }
    static void complex_store(std::complex<T>* p, V re, V im) {
        auto* values = reinterpret_cast<T*>(p);
        _mm_storeu_ps(values, _mm_unpacklo_ps(re, im));
        _mm_storeu_ps(values + 4, _mm_unpackhi_ps(re, im));
    }
};
#endif
#undef GIBBS_VECTOR_OPS
#undef GIBBS_NORMAL_RECIPROCAL

template<class T>
std::size_t gibbs_vector(const GibbsStep<T>& step, std::size_t i,
                         std::size_t end, GibbsSums& sums) {
    using A = GibbsVector<T>;
    using V = typename A::V;
    constexpr auto lanes = A::lanes;
    const auto [data, transfer, reg, power, reg_power, nr, ni, total, noise, prior,
                previous_count, delta_divisor, accumulate, measure_delta] = step;
    const auto one = A::set(1), half = A::set(0.5);
    const auto noise_work = A::set(noise), neg_noise = A::set(-noise), prior_work = A::set(prior);
    const auto previous = A::set(previous_count), divisor = A::set(delta_divisor);
    auto rn = A::set(0), pn = rn, change = rn, magnitude = rn;
    for (; end - i >= lanes; i += lanes) {
        V hr, hi, re, im;
        A::complex_load(transfer + i, hr, hi);
        A::complex_load(data + i, re, im);
        const auto precision = A::add(A::mul(noise_work, A::load(power + i)),
                                      A::mul(prior_work, A::load(reg_power + i)));
        const auto reciprocal = A::div(one, precision);
        const auto fr = A::mul(A::mul(noise_work, hr), reciprocal);
        const auto fi = A::mul(A::mul(neg_noise, hi), reciprocal);
        const auto scale = A::sqrt(A::variance(reciprocal, precision, half));
        const auto sr = A::add(A::sub(A::mul(fr, re), A::mul(fi, im)),
                               A::mul(scale, A::normals(nr + i)));
        const auto si = A::add(A::add(A::mul(fr, im), A::mul(fi, re)),
                               A::mul(scale, A::normals(ni + i)));
        const auto rr = A::sub(re, A::sub(A::mul(sr, hr), A::mul(si, hi)));
        const auto ri = A::sub(im, A::add(A::mul(sr, hi), A::mul(si, hr)));
        const auto regularizer = A::load(reg + i);
        rn = A::add(rn, A::norm2(rr, ri));
        pn = A::add(pn, A::norm2(A::mul(sr, regularizer), A::mul(si, regularizer)));
        if (accumulate) {
            V old_re, old_im;
            A::complex_load(total + i, old_re, old_im);
            const auto new_re = A::add(old_re, sr), new_im = A::add(old_im, si);
            A::complex_store(total + i, new_re, new_im);
            if (measure_delta) {
                const auto dr = A::div(A::sub(A::mul(previous, sr), old_re), divisor);
                const auto di = A::div(A::sub(A::mul(previous, si), old_im), divisor);
                change = A::add(change, A::sqrt(A::norm2(dr, di)));
                magnitude = A::add(magnitude, A::sqrt(A::norm2(new_re, new_im)));
            }
        }
    }
    sums.noise += A::sum(rn);
    sums.prior += A::sum(pn);
    if (measure_delta) {
        sums.change += A::sum(change);
        sums.magnitude += A::sum(magnitude);
    }
    return i;
}
#endif
} // namespace

template<class T>
std::array<double, 3> restoration_gibbs_accumulate(const std::complex<T>* data,
    const std::complex<T>* transfer, const T* reg, const T* power, const T* reg_power,
    T noise, T prior, const double* normals, std::size_t height, std::size_t width,
    std::size_t count, std::complex<T>* total, bool measure_delta) {
    measure_delta = measure_delta && count > 1;
#if defined(RESTORATION_AVX2) || defined(RESTORATION_SSE2)
    const bool vectorize = has_avx2();
#endif
    const auto previous = measure_delta ? T(count - 1) : T(0);
    const auto divisor = measure_delta ? T(count) * previous : T(0);
    const GibbsStep<T> step{data, transfer, reg, power, reg_power, normals,
                            normals + height * width, total, noise, prior, previous, divisor,
                            count != 0, measure_delta};
    const bool half_weight = height != width;
    GibbsSums sums, first_column;
    for (std::size_t row = 0; row < height; ++row) {
        auto i = row * width;
        const auto end = i + width;
        if (half_weight) gibbs_scalar(step, i++, first_column);
#if defined(RESTORATION_AVX2) || defined(RESTORATION_SSE2)
        if (vectorize) i = gibbs_vector(step, i, end, sums);
#endif
        for (; i < end; ++i) gibbs_scalar(step, i, sums);
    }
    const auto weight = half_weight ? 2 : 1;
    const auto delta = measure_delta
        ? (sums.change + first_column.change) / (sums.magnitude + first_column.magnitude) / count
        : std::numeric_limits<double>::quiet_NaN();
    return {double(T(weight * sums.noise + first_column.noise)),
            double(T(weight * sums.prior + first_column.prior)), delta};
}

template<class T>
void restoration_gibbs_forward(const T* image, const RestorationShape& shape,
                               std::complex<T>* data, int workers) {
    const auto size = size_of(shape);
    const auto count = size / shape.back() * (shape.back() / 2 + 1);
    detail::fft_r2c(shape, axes_of(shape), image, data, workers);
    const auto scale = T(1 / std::sqrt(double(size)));
    for (std::size_t i = 0; i < count; ++i) data[i] *= scale;
}

template<class T>
void restoration_gibbs_inverse(const std::complex<T>* data, const RestorationShape& shape,
                               T* output, int workers) {
    const auto size = size_of(shape);
    detail::fft_c2r(shape, axes_of(shape), data, output, workers);
    const auto scale = T(std::sqrt(double(size)));
    for (std::size_t i = 0; i < size; ++i) output[i] *= scale;
}

template<class T>
void restoration_interpolate(const T* image, const RestorationShape& shape,
                             std::size_t spatial_rank, T* output) {
    const auto size = size_of(shape);
    if (!image || !output || spatial_rank == 0 || spatial_rank > shape.size())
        throw std::invalid_argument("invalid restoration interpolation buffers or rank");
    const auto strides = element_strides(shape);
    const auto block = strides[spatial_rank - 1];
    if (spatial_rank == 2) {
        interpolate_2d(image, shape, block, output);
        return;
    }
    const T weight = T(1) / T(2 * spatial_rank);
    const auto row_size = strides[spatial_rank - 1] * shape[spatial_rank - 1];
    RestorationShape before(spatial_rank), after(spatial_rank);
    RestorationShape coordinates(spatial_rank - 1);
    for (std::size_t row = 0; row < size; row += row_size) {
        for (std::size_t axis = 0; axis + 1 < spatial_rank; ++axis) {
            const auto step = strides[axis], extent = shape[axis];
            const auto coord = coordinates[axis];
            before[axis] = extent == 1 ? row : (coord == 0 ? row + step : row - step);
            after[axis] = extent == 1 ? row : (coord + 1 == extent ? row - step : row + step);
        }
        for (std::size_t j = 0; j < row_size; ++j) {
            const auto pos = row + j;
            const auto left = row_size == block ? pos : (j < block ? pos + block : pos - block);
            const auto right = row_size == block ? pos :
                (j >= row_size - block ? pos - block : pos + block);
            double sum = 0;
            for (std::size_t axis = 0; axis + 1 < spatial_rank; ++axis)
                sum += double(image[before[axis] + j]) * weight;
            sum += double(image[left]) * weight;
            sum += double(image[right]) * weight;
            for (auto axis = spatial_rank - 1; axis-- > 0;)
                sum += double(image[after[axis] + j]) * weight;
            output[pos] = T(sum);
        }
        advance_row(coordinates, shape);
    }
}

void restoration_interpolate_phase(const float* image, std::size_t height, std::size_t width,
                                   std::array<std::size_t, 2> phase, std::size_t stride, float* out) {
    interpolate_phase_2d(image, height, width, phase, stride, out);
}

void restoration_interpolate_phase(const double* image, std::size_t height, std::size_t width,
                                   std::array<std::size_t, 2> phase, std::size_t stride, double* out) {
    interpolate_phase_2d(image, height, width, phase, stride, out);
}

template<class T>
void restoration_roll(const T* image, const RestorationShape& shape,
                      const RestorationShape& shifts, T* output, bool add) {
    const auto size = size_of(shape);
    if (!image || !output || shifts.size() != shape.size())
        throw std::invalid_argument("invalid restoration roll buffers or rank");
    std::size_t last = 0;
    for (std::size_t axis = 0; axis < shape.size(); ++axis) {
        if (shifts[axis] >= shape[axis])
            throw std::invalid_argument("restoration roll shift is not reduced");
        if (shifts[axis]) last = axis;
    }
#if defined(RESTORATION_AVX2) || defined(RESTORATION_SSE2)
    const bool vectorize = add && has_avx2();
#endif
    const auto span = [&](const T* input, T* out, std::size_t count) {
        if (!add) {
            std::copy_n(input, count, out);
            return;
        }
        std::size_t i = 0;
#if defined(RESTORATION_AVX2) || defined(RESTORATION_SSE2)
        if (vectorize) i = add_span_simd(input, out, count);
#endif
        for (; i < count; ++i) out[i] += input[i];
    };
    if (last == 0) {
        const auto split = shifts[0] * (size / shape[0]);
        if (split) span(image + size - split, output, split);
        span(image, output + split, size - split);
        return;
    }
    if (last == 1) {
        const auto height = shape[0], width = size / height;
        const auto split = shifts[1] * (width / shape[1]);
        const auto tail = width - split;
        const auto first = shifts[0] ? height - shifts[0] : 0;
        auto source = first;
        for (std::size_t row = 0; row < size; row += width) {
            const auto* input = image + source * width;
            span(input + tail, output + row, split);
            span(input, output + row + split, tail);
            if (++source == height) source = 0;
        }
        return;
    }
    const auto strides = element_strides(shape);
    const auto row_size = shape[last] * strides[last];
    const auto split = shifts[last] * strides[last];
    RestorationShape coordinates(last);
    std::size_t source = 0;
    for (std::size_t axis = 0; axis < last; ++axis) {
        coordinates[axis] = shifts[axis] ? shape[axis] - shifts[axis] : 0;
        source += coordinates[axis] * strides[axis];
    }
    for (std::size_t row = 0; row < size; row += row_size) {
        const auto tail = row_size - split;
        if (split) span(image + source + tail, output + row, split);
        span(image + source, output + row + split, tail);
        // Source wrapping and destination carry occur at different positions.
        for (auto axis = last; axis-- > 0;) {
            if (++coordinates[axis] == shape[axis]) {
                coordinates[axis] = 0;
                source -= (shape[axis] - 1) * strides[axis];
            } else {
                source += strides[axis];
            }
            const auto start = shifts[axis] ? shape[axis] - shifts[axis] : 0;
            if (coordinates[axis] != start) break;
        }
    }
}

#define INSTANTIATE(T) \
    template void restoration_transfer(const T*, const RestorationShape&, const RestorationShape&, std::complex<T>*, int); \
    template void restoration_laplacian(const RestorationShape&, T*); \
    template bool restoration_wiener(const T*, const RestorationShape&, const std::complex<T>*, const T*, T, bool, T*, int); \
    template void restoration_gibbs_sample(const std::complex<T>*, const std::complex<T>*, const T*, const double*, const double*, std::size_t, std::complex<T>*); \
    template void restoration_gibbs_step(const std::complex<T>*, const std::complex<T>*, const T*, const T*, const T*, T, T, const double*, std::size_t, std::complex<T>*, std::complex<T>*); \
    template std::array<double, 3> restoration_gibbs_accumulate(const std::complex<T>*, const std::complex<T>*, const T*, const T*, const T*, T, T, const double*, std::size_t, std::size_t, std::size_t, std::complex<T>*, bool); \
    template void restoration_gibbs_forward(const T*, const RestorationShape&, std::complex<T>*, int); \
    template void restoration_gibbs_inverse(const std::complex<T>*, const RestorationShape&, T*, int); \
    template void restoration_interpolate(const T*, const RestorationShape&, std::size_t, T*); \
    template void restoration_roll(const T*, const RestorationShape&, const RestorationShape&, T*, bool)
INSTANTIATE(float);
INSTANTIATE(double);
#undef INSTANTIATE
} // namespace skimagexx
