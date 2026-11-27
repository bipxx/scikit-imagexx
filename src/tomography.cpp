// Tomography conventions follow scikit-image 0.26 (BSD-3-Clause).
// See third_party/scikit-image-LICENSE.txt.
#include <skimagexx/tomography.hpp>
#include <skimagexx/order_angles_golden_ratio.hpp>
#include "execution.hpp"
#include "fft_internal.hpp"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <complex>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(SKIMAGEXX_AVX2) && (defined(_M_X64) || defined(__x86_64__))
#include <immintrin.h>
#define SKIMAGEXX_TOMOGRAPHY_SIMD
#ifdef _MSC_VER
#define SKIMAGEXX_TOMOGRAPHY_AVX2 __declspec(noinline)
#else
#define SKIMAGEXX_TOMOGRAPHY_AVX2 __attribute__((target("avx2"), noinline))
#endif
#endif

namespace skimagexx {
namespace {
constexpr double pi = 3.14159265358979323846;

template<class T>
bool outside_circle(const T* image, int rows, int columns) {
    const std::int64_t radius = std::min(rows, columns) / 2;
    for (int r = 0; r < rows; ++r) {
        const std::int64_t y = r - rows / 2;
        for (int c = 0; c < columns; ++c) {
            const std::int64_t x = c - columns / 2;
            if (x * x + y * y > radius * radius && image[std::size_t(r) * columns + c] != T(0)) {
                return true;
            }
        }
    }
    return false;
}

template<class T> int floor_index(T coordinate) {
    const int truncated = static_cast<int>(coordinate);
    return truncated - (coordinate < T(truncated));
}

template<class Predicate>
int first_true(int begin, int end, const Predicate& predicate) {
    while (begin < end) {
        const int middle = begin + (end - begin) / 2;
        if (predicate(middle)) end = middle;
        else begin = middle + 1;
    }
    return begin;
}

template<bool lower_inclusive, bool upper_inclusive = lower_inclusive, class T, class Coordinate>
std::pair<int, int> axis_span(int size, T lower, T upper, const Coordinate& coordinate) {
    const auto below = [=](T value) { return lower_inclusive ? value < lower : value <= lower; };
    const auto above = [=](T value) { return upper_inclusive ? value > upper : value >= upper; };
    const T first = coordinate(0), last = coordinate(size - 1);
    // Search the sampled coordinates using the same rounded expression as
    // interpolation. Inverting that expression can shift a boundary by a tap.
    if (first <= last) {
        if (above(first) || below(last)) return {0, 0};
        const int begin = below(first) ? first_true(0, size, [&](int i) { return !below(coordinate(i)); }) : 0;
        const int end = above(last) ? first_true(begin, size, [&](int i) { return above(coordinate(i)); }) : size;
        return {begin, end};
    }
    if (below(first) || above(last)) return {0, 0};
    const int begin = above(first) ? first_true(0, size, [&](int i) { return !above(coordinate(i)); }) : 0;
    const int end = below(last) ? first_true(begin, size, [&](int i) { return below(coordinate(i)); }) : size;
    return {begin, end};
}

template<class T>
T bilinear(const T* square, int pitch, T r, T c) {
    const int row = floor_index(r);
    const int column = floor_index(c);
    const T dr = r - T(row), dc = c - T(column);
    const auto* pixels = square + std::ptrdiff_t(row) * pitch + column;
    const double top = (T(1) - dc) * pixels[0] + dc * pixels[1];
    const double bottom = (T(1) - dc) * pixels[pitch] + dc * pixels[pitch + 1];
    return T((T(1) - dr) * top + dr * bottom);
}

#ifdef SKIMAGEXX_TOMOGRAPHY_SIMD
SKIMAGEXX_TOMOGRAPHY_AVX2
int radon_span(const double* square, int pitch, const double* column_x, const double* column_y,
               double row_x, double row_y, double offset_x, double offset_y,
               double* sums, int begin, int end) {
    const auto one = _mm256_set1_pd(1.0);
    const auto stride = _mm_set1_epi32(pitch);
    const auto next = _mm_set1_epi32(1);
    const auto x_row = _mm256_set1_pd(row_x), y_row = _mm256_set1_pd(row_y);
    const auto x_offset = _mm256_set1_pd(offset_x), y_offset = _mm256_set1_pd(offset_y);
    for (; begin + 4 <= end; begin += 4) {
        const auto x = _mm256_add_pd(_mm256_add_pd(_mm256_loadu_pd(column_x + begin), x_row), x_offset);
        const auto y = _mm256_add_pd(_mm256_add_pd(_mm256_loadu_pd(column_y + begin), y_row), y_offset);
        const auto left = _mm256_cvttpd_epi32(_mm256_floor_pd(x));
        const auto top = _mm256_cvttpd_epi32(_mm256_floor_pd(y));
        const auto index = _mm_add_epi32(_mm_mullo_epi32(top, stride), left);
        const auto below = _mm_add_epi32(index, stride);
        const auto dc = _mm256_sub_pd(x, _mm256_cvtepi32_pd(left));
        const auto dr = _mm256_sub_pd(y, _mm256_cvtepi32_pd(top));
        const auto lc = _mm256_sub_pd(one, dc);
        const auto a = _mm256_add_pd(_mm256_mul_pd(lc, _mm256_i32gather_pd(square, index, 8)),
                                     _mm256_mul_pd(dc, _mm256_i32gather_pd(square, _mm_add_epi32(index, next), 8)));
        const auto b = _mm256_add_pd(_mm256_mul_pd(lc, _mm256_i32gather_pd(square, below, 8)),
                                     _mm256_mul_pd(dc, _mm256_i32gather_pd(square, _mm_add_epi32(below, next), 8)));
        const auto value = _mm256_add_pd(_mm256_mul_pd(_mm256_sub_pd(one, dr), a), _mm256_mul_pd(dr, b));
        _mm256_storeu_pd(sums + begin, _mm256_add_pd(_mm256_loadu_pd(sums + begin), value));
    }
    return begin;
}
#endif

std::vector<double> frequency_filter(std::size_t size, TomographyFilter filter) {
    std::vector<double> result(size / 2 + 1, 1.0);
    if (filter == TomographyFilter::none) return result;
    std::vector<double> impulse(size);
    impulse[0] = 0.25;
    for (std::size_t index = 1; index < size; index += 2) {
        const double x = pi * double(std::min(index, size - index));
        impulse[index] = -1 / (x * x);
    }
    std::vector<std::complex<double>> spectrum(result.size());
    detail::fft_r2c({size}, {0}, impulse.data(), spectrum.data(), 1);
    const double cosine_step = 2 * pi / double(size - 1);
    for (std::size_t i = 0; i < result.size(); ++i) {
        const double omega = pi * double(i) / double(size);
        const auto shifted = (i + size / 2) % size;
        double window = 1;
        switch (filter) {
        case TomographyFilter::shepp_logan: if (i) window = std::sin(omega) / omega; break;
        case TomographyFilter::cosine: window = std::sin(pi * double(shifted) / double(size)); break;
        case TomographyFilter::hamming:
        case TomographyFilter::hann: {
            // These even-length windows are symmetric about a half-sample.
            // The real part of the full inverse FFT averages the two weights
            // on each conjugate pair. A real FFT needs that average up front.
            const auto opposite = (size / 2 + size - i) % size;
            const double mean_cosine = (std::cos(cosine_step * double(shifted)) +
                                        std::cos(cosine_step * double(opposite))) / 2;
            window = filter == TomographyFilter::hamming
                ? 0.54 - 0.46 * mean_cosine : 0.5 - 0.5 * mean_cosine;
            break;
        }
        default: break;
        }
        result[i] = 2 * spectrum[i].real() * window;
    }
    return result;
}

template<class T>
std::vector<double> filtered_projections(const T* data, int detectors, int angles,
                                         int padded, TomographyFilter filter, int workers) {
    const int offset = padded / 2 - detectors / 2;
    std::size_t size = 64;
    while (size < 2 * std::size_t(padded)) size *= 2;
    const auto response = frequency_filter(size, filter);
    const std::size_t bins = response.size();
    const std::vector<std::size_t> shape{std::size_t(angles), size}, axes{1};
    std::vector<T> spatial(std::size_t(angles) * size);
    for (int k = 0; k < angles; ++k) {
        auto* column = spatial.data() + std::size_t(k) * size + offset;
        for (int r = 0; r < detectors; ++r) column[r] = data[std::size_t(r) * angles + k];
    }
    std::vector<std::complex<double>> spectrum(std::size_t(angles) * bins);
    const auto apply_response = [&](const auto* input) {
        for (int k = 0; k < angles; ++k) {
            const auto base = std::size_t(k) * bins;
            for (std::size_t f = 0; f < bins; ++f) {
                auto value = std::complex<double>(input[base + f]);
                value *= response[f];
                spectrum[base + f] = value;
            }
        }
    };
    if constexpr (std::is_same_v<T, double>) {
        detail::fft_r2c(shape, axes, spatial.data(), spectrum.data(), workers);
        apply_response(spectrum.data());
    } else {
        std::vector<std::complex<T>> single(spectrum.size());
        detail::fft_r2c(shape, axes, spatial.data(), single.data(), workers);
        apply_response(single.data());
    }
    // Keep a compact prefix after inversion, one contiguous detector array per angle.
    std::vector<double> inverse(std::size_t(angles) * size);
    detail::fft_c2r(shape, axes, spectrum.data(), inverse.data(), workers);
    for (int k = 1; k < angles; ++k) {
        std::copy_n(inverse.data() + std::size_t(k) * size, padded,
                    inverse.data() + std::size_t(k) * padded);
    }
    inverse.resize(std::size_t(angles) * padded);
    return inverse;
}

struct CubicSegment {
    double value, slope, quadratic, cubic;
};

std::vector<CubicSegment> cubic_segments(const std::vector<double>& projections,
                                         int detectors, int angles) {
    // On a uniform grid, eliminating the not-a-knot end conditions leaves
    // known second derivatives at nodes 1 and n-2 and a tridiagonal interior.
    std::vector<CubicSegment> result(std::size_t(angles) * (detectors - 1));
    std::vector<double> second(detectors), reciprocal(detectors);
    for (int i = 2; i < detectors - 2; ++i) reciprocal[i] = 1 / (4 - reciprocal[i - 1]);
    for (int k = 0; k < angles; ++k) {
        const auto* col = projections.data() + std::size_t(k) * detectors;
        auto* segments = result.data() + std::size_t(k) * (detectors - 1);
        const auto difference = [&](int i) { return col[i + 1] - 2 * col[i] + col[i - 1]; };
        second[1] = difference(1);
        second[detectors - 2] = difference(detectors - 2);
        for (int i = 2; i < detectors - 2; ++i) {
            double rhs = 6 * difference(i) - second[i - 1];
            if (i == detectors - 3) rhs -= second[detectors - 2];
            second[i] = rhs * reciprocal[i];
        }
        for (int i = detectors - 4; i >= 2; --i) second[i] -= reciprocal[i] * second[i + 1];
        second.front() = 2 * second[1] - second[2];
        second.back() = 2 * second[detectors - 2] - second[detectors - 3];
        for (int i = 0; i < detectors - 1; ++i) {
            segments[i] = {col[i], col[i + 1] - col[i] - (2 * second[i] + second[i + 1]) / 6,
                           second[i] / 2, (second[i + 1] - second[i]) / 6};
        }
    }
    return result;
}

template<TomographyInterpolation interpolation, class T>
void backproject(const std::vector<double>& projections, int detectors, int angles,
                 const double* cosines, const double* sines, int size, bool circle, T* image,
                 int workers) {
    std::vector<CubicSegment> segments;
    if constexpr (interpolation == TomographyInterpolation::cubic) {
        segments = cubic_segments(projections, detectors, angles);
    }
    const int radius = size / 2, center = detectors / 2;
    const double lower = -center, upper = detectors - 1 - center;
    const T scale = T(2.0 * angles);
    const auto radius_squared = std::int64_t(radius) * radius;
    // Keep strips of the output in cache while accumulating angles in their
    // original order. Reuse the column coordinates and circular row limits.
    constexpr std::size_t strip_height = 16;
    const auto strips = (std::size_t(size) + strip_height - 1) / strip_height;
    const int strip_workers = static_cast<int>(std::min(strips, std::size_t(workers)));
    const std::unique_ptr<double[]> horizontal(new double[std::size_t(strip_workers) * size]);
    std::vector<std::pair<int, int>> row_limits(size, {0, size});
    if (circle) {
        for (int r = 0; r < size; ++r) {
            const std::int64_t y = r - radius;
            const auto remaining = radius_squared - y * y;
            int x = static_cast<int>(std::sqrt(double(remaining)));
            if (std::int64_t(x) * x > remaining) --x;
            row_limits[r] = {radius - x, std::min(size, radius + x + 1)};
        }
    }
    detail::partition(strips, strip_workers, [&](std::size_t begin, std::size_t end, std::size_t worker) {
        auto* column_coordinates = horizontal.get() + worker * size;
        for (auto strip = begin; strip < end; ++strip) {
            const auto first = strip * strip_height;
            const auto last = std::min(first + strip_height, std::size_t(size));
            std::fill_n(image + first * size, (last - first) * size, T(0));
            for (int k = 0; k < angles; ++k) {
                const auto* col = projections.data() + std::size_t(k) * detectors;
                const CubicSegment* segment = nullptr;
                if constexpr (interpolation == TomographyInterpolation::cubic) {
                    segment = segments.data() + std::size_t(k) * (detectors - 1);
                }
                const double cosine = cosines[k], sine = sines[k];
                for (int c = 0; c < size; ++c) column_coordinates[c] = double(c - radius) * cosine;
                for (auto r = first; r < last; ++r) {
                    const double vertical = (double(r) - radius) * sine;
                    auto* row = image + r * size;
                    const auto coordinate = [&](int c) { return column_coordinates[c] - vertical; };
                    const auto [first_column, last_column] = axis_span<true>(size, lower, upper, coordinate);
                    const auto [row_begin, row_end] = row_limits[r];
                    for (int c = std::max(first_column, row_begin); c < std::min(last_column, row_end); ++c) {
                        const double t = coordinate(c);
                        const int left = floor_index(t);
                        const int i = left + center;
                        double value;
                        if constexpr (interpolation == TomographyInterpolation::nearest) {
                            value = col[i + (t > double(left) + 0.5)];
                        } else if (i == detectors - 1) {
                            value = col[i];
                        } else if constexpr (interpolation == TomographyInterpolation::cubic) {
                            const auto& p = segment[i];
                            const double fraction = t - left;
                            value = p.value + fraction * (p.slope + fraction * (p.quadratic + fraction * p.cubic));
                        } else {
                            const double fraction = t - left;
                            value = col[i] + fraction * (col[i + 1] - col[i]);
                        }
                        row[c] = T(double(row[c]) + value);
                    }
                }
            }
            for (auto r = first; r < last; ++r) {
                auto* row = image + r * size;
                const auto [row_begin, row_end] = row_limits[r];
                for (int c = row_begin; c < row_end; ++c) row[c] = (row[c] * T(pi)) / scale;
            }
        }
    });
}

template<class T> struct RaySample {
    std::size_t base;
    T di, dj;
    bool has_top, has_left, has_bottom, has_right;
};

template<class T> struct RayChord {
    T position, half_length;
    int steps;
};

template<class T>
RayChord<T> ray_chord(T ray_position, T center, T radius) {
    const T t = ray_position - center;
    const T length_squared = radius * radius - t * t;
    if (!(length_squared > T(0))) return {t, T(0), 0};
    const T half_length = T(std::sqrt(double(length_squared)));
    return {t, half_length, 2 * static_cast<int>(std::ceil(2 * double(half_length)))};
}

template<class T>
struct RayWindows {
    std::vector<std::size_t> offsets;
    std::vector<T> values;
};

template<class T>
RayWindows<T> ray_windows(const T* shifts, int angles, int detectors, T center, T radius) {
    RayWindows<T> table;
    table.offsets.resize(std::size_t(detectors) + 1);
    auto& offsets = table.offsets;
    std::vector<T> distinct;
    distinct.reserve(angles);
    for (int k = 0; k < angles; ++k) {
        if (std::isfinite(shifts[k])) distinct.push_back(shifts[k]);
    }
    std::sort(distinct.begin(), distinct.end());
    const auto end = std::unique(distinct.begin(), distinct.end());
    // Mark the window sizes used by the projections, then replace those marks
    // with offsets into one allocation. Zero-step chords have no window.
    for (auto shift = distinct.begin(); shift != end; ++shift) {
        for (int detector = 0; detector < detectors; ++detector) {
            const auto chord = ray_chord(T(detector) + *shift, center, radius);
            offsets[std::size_t(chord.steps) / 2] = 1;
        }
    }
    std::size_t count = 0;
    for (std::size_t i = 1; i < offsets.size(); ++i) {
        if (offsets[i]) count += 2 * i + 1;
    }
    auto& windows = table.values;
    windows.reserve(count);
    const T beta = T(0.46164);
    for (std::size_t i = 1; i < offsets.size(); ++i) {
        if (!offsets[i]) continue;
        offsets[i] = windows.size();
        const int steps = static_cast<int>(2 * i);
        for (int k = 0; k <= steps; ++k) {
            windows.push_back(T((T(1) - beta) - beta * std::cos(2 * pi * k / (steps - 1))));
        }
    }
    return table;
}

template<bool vector_taps, class T>
void project_ray(const T* image, T* update, int size, double cosine, double sine,
                 const RayChord<T>& chord, T projection, RaySample<T>* samples,
                 const T* window) {
    const T center = T(size / 2), t = chord.position, half_length = chord.half_length;
    const int steps = chord.steps;
    const T ds = (T(2) * half_length) / T(steps);
    const T dx = T(-ds * cosine), dy = T(-ds * sine);
    const T x0 = T(half_length * cosine - t * sine);
    const T y0 = T(half_length * sine + t * cosine);
    T total = 0, norm = 0;
    const auto accumulate = [&](std::size_t index, T weight) {
        total += weight * image[index];
        norm += weight * weight;
    };
    // Only the chord's ends can cross a grid line with missing taps.
    // Clip the full-tap span using the original rounded sample coordinates.
    const auto row_at = [&](int k) { return (x0 + T(k) * dx) + center; };
    const auto col_at = [&](int k) { return (y0 + T(k) * dy) + center; };
    const int end = steps + 1;
    // The upstream top and left predicates use truncated indices.
    const auto [row_begin, row_end] = axis_span<true, false>(end, T(1), T(size - 1), row_at);
    const auto [col_begin, col_end] = axis_span<true, false>(end, T(1), T(size - 1), col_at);
    const int begin = std::max(row_begin, col_begin);
    const int stop = std::max(begin, std::min(row_end, col_end));
    const auto forward = [&](auto full_taps, int first, int last) {
        constexpr bool interior = decltype(full_taps)::value;
#ifdef SKIMAGEXX_TOMOGRAPHY_SIMD
        __m128d sums;
        if constexpr (interior && vector_taps) sums = _mm_set_pd(norm, total);
#endif
        for (int k = first; k < last; ++k) {
            const T row = row_at(k), col = col_at(k);
            // The chord lies inside the detector grid. Its endpoint can land on
            // the last grid line, where there is no bilinear neighbor beyond it.
            const int i = static_cast<int>(row), j = static_cast<int>(col);
            const T di = row - T(i), dj = col - T(j);
            const auto base = std::size_t(i) * size + j;
            auto& sample = samples[k];
            sample.base = base;
            sample.di = di;
            sample.dj = dj;
#ifdef SKIMAGEXX_TOMOGRAPHY_SIMD
            if constexpr (interior && vector_taps) {
                const auto columns = _mm_set_pd(dj, 1.0 - dj);
                const auto spacing = _mm_set1_pd(ds);
                const auto top = _mm_mul_pd(_mm_mul_pd(_mm_set1_pd(1.0 - di), columns), spacing);
                const auto bottom = _mm_mul_pd(_mm_mul_pd(_mm_set1_pd(di), columns), spacing);
                const auto top_values = _mm_mul_pd(top, _mm_loadu_pd(image + base));
                const auto bottom_values = _mm_mul_pd(bottom, _mm_loadu_pd(image + base + size));
                const auto top_norm = _mm_mul_pd(top, top), bottom_norm = _mm_mul_pd(bottom, bottom);
                // Each lane is one accumulator. Both retain the original tap order.
                sums = _mm_add_pd(sums, _mm_unpacklo_pd(top_values, top_norm));
                sums = _mm_add_pd(sums, _mm_unpackhi_pd(top_values, top_norm));
                sums = _mm_add_pd(sums, _mm_unpacklo_pd(bottom_values, bottom_norm));
                sums = _mm_add_pd(sums, _mm_unpackhi_pd(bottom_values, bottom_norm));
            } else
#endif
            {
                if constexpr (!interior) {
                    sample.has_top = i > 0;
                    sample.has_left = j > 0;
                    sample.has_bottom = i < size - 1;
                    sample.has_right = j < size - 1;
                }
                if (interior || (sample.has_top && sample.has_left)) accumulate(base, T((1.0 - di) * (1.0 - dj) * ds));
                if (interior || (sample.has_top && sample.has_right)) accumulate(base + 1, T((1.0 - di) * dj * ds));
                if (interior || (sample.has_bottom && sample.has_left)) accumulate(base + size, di * (T(1) - dj) * ds);
                if (interior || (sample.has_bottom && sample.has_right)) accumulate(base + size + 1, di * dj * ds);
            }
        }
#ifdef SKIMAGEXX_TOMOGRAPHY_SIMD
        if constexpr (interior && vector_taps) {
            total = _mm_cvtsd_f64(sums);
            norm = _mm_cvtsd_f64(_mm_unpackhi_pd(sums, sums));
        }
#endif
    };
    forward(std::false_type{}, 0, begin);
    forward(std::true_type{}, begin, stop);
    forward(std::false_type{}, stop, end);
    const T deviation = norm > T(0) ? -(total - projection) / norm : T(0);
    const auto back = [&](auto full_taps, int first, int last) {
        constexpr bool interior = decltype(full_taps)::value;
        for (int k = first; k < last; ++k) {
            const auto& sample = samples[k];
            const auto base = sample.base;
            const T di = sample.di, dj = sample.dj, w = window[k];
#ifdef SKIMAGEXX_TOMOGRAPHY_SIMD
            if constexpr (interior && vector_taps) {
                const auto columns = _mm_set_pd(dj, 1.0 - dj);
                const auto spacing = _mm_set1_pd(ds), weight = _mm_set1_pd(w);
                const auto increment = [&](double row) {
                    return _mm_mul_pd(_mm_mul_pd(_mm_mul_pd(_mm_set1_pd(deviation * row), columns), spacing), weight);
                };
                auto* out = update + base;
                _mm_storeu_pd(out, _mm_add_pd(_mm_loadu_pd(out), increment(1.0 - di)));
                out += size;
                _mm_storeu_pd(out, _mm_add_pd(_mm_loadu_pd(out), increment(di)));
            } else
#endif
            {
                if (interior || (sample.has_top && sample.has_left)) {
                    update[base] = T(update[base] + deviation * (1.0 - di) * (1.0 - dj) * ds * w);
                }
                if (interior || (sample.has_top && sample.has_right)) {
                    update[base + 1] = T(update[base + 1] + deviation * (1.0 - di) * dj * ds * w);
                }
                if (interior || (sample.has_bottom && sample.has_left)) {
                    update[base + size] += deviation * di * (T(1) - dj) * ds * w;
                }
                if (interior || (sample.has_bottom && sample.has_right))
                    update[base + size + 1] += deviation * di * dj * ds * w;
            }
        }
    };
    back(std::false_type{}, 0, begin);
    back(std::true_type{}, begin, stop);
    back(std::false_type{}, stop, end);
}
}

int radon_detector_size(int rows, int columns, bool circle) {
    return circle ? std::min(rows, columns) : int(std::ceil(std::sqrt(2.0) * std::max(rows, columns)));
}

bool radon_outside_circle(const float* image, int rows, int columns) {
    return outside_circle(image, rows, columns);
}

bool radon_outside_circle(const double* image, int rows, int columns) {
    return outside_circle(image, rows, columns);
}

template<class T>
void radon_into(const T* image, int rows, int columns, const RadonRotation* rotations,
                int angles, bool circle, T* sinogram) {
    const int size = radon_detector_size(rows, columns, circle);
    const int pitch = size + 2;
    std::vector<T> prepared(std::size_t(pitch) * pitch);
    auto* square = prepared.data() + pitch + 1;
    const int source_row = circle ? (rows - size + 1) / 2 : 0;
    const int source_col = circle ? (columns - size + 1) / 2 : 0;
    const int target_row = circle ? 0 : size / 2 - rows / 2;
    const int target_col = circle ? 0 : size / 2 - columns / 2;
    const int copied_rows = circle ? size : rows, copied_cols = circle ? size : columns;
    for (int r = 0; r < copied_rows; ++r) {
        std::copy_n(image + std::size_t(r + source_row) * columns + source_col, copied_cols,
                    square + std::size_t(r + target_row) * pitch + target_col);
    }
    int left = 0, top = 0, right = size - 1, bottom = size - 1;
    const bool round_to_nearest = std::fegetround() == FE_TONEAREST;
    if (round_to_nearest) {
        // Inverse mapping can skip the zero exterior of the nonzero bounds.
        // Keep a one-pixel margin for the bilinear interpolation taps.
        left = size;
        top = size;
        right = bottom = -1;
        for (int r = target_row; r < target_row + copied_rows; ++r) {
            const auto* row = square + std::size_t(r) * pitch;
            int first = target_col, last = target_col + copied_cols - 1;
            while (first <= last && row[first] == T(0)) ++first;
            if (first > last) continue;
            while (row[last] == T(0)) --last;
            left = std::min(left, first);
            right = std::max(right, last);
            top = std::min(top, r);
            bottom = r;
        }
        if (bottom < 0) {
            std::fill_n(sinogram, std::size_t(size) * angles, T(0));
            return;
        }
    }
    constexpr int cache_budget = 24 * 1024;
    const int tile_size = std::min(angles, std::clamp(
        cache_budget / (int(sizeof(T)) * size) - 2, 1, 8));
    const int workers = std::min(angles, detail::active_workers(std::size_t(size) * size * angles));
    std::vector<T> workspace(std::size_t(workers) * size * (tile_size + 2));
#ifdef SKIMAGEXX_TOMOGRAPHY_SIMD
    const bool vector_spans = std::is_same_v<T, double> && pitch <= 32768 && has_avx2();
#endif
    detail::partition(std::size_t(angles), workers,
        [&](std::size_t begin, std::size_t end, std::size_t worker) {
            auto* tile = workspace.data() + worker * size * (tile_size + 2);
            auto* column_x = tile + tile_size * size;
            auto* column_y = column_x + size;
            for (auto first = begin; first < end; first += tile_size) {
                const auto last = std::min(first + tile_size, end);
                for (auto k = first; k < last; ++k) {
                    auto* sums = tile + (k - first) * size;
                    const auto& rotation = rotations[k];
                    const T cosine = T(rotation.cosine), sine = T(rotation.sine);
                    const T offset_x = T(rotation.column_offset), offset_y = T(rotation.row_offset);
                    std::fill_n(sums, size, T(0));
                    // At the exact identity, every bilinear sample is a grid value.
                    if (round_to_nearest && cosine == T(1) && sine == T(0)
                        && offset_x == T(0) && offset_y == T(0)) {
                        for (int r = top; r <= bottom; ++r) {
                            const auto* row = square + std::size_t(r) * pitch;
                            for (int c = left; c <= right; ++c) sums[c] += row[c];
                        }
                        continue;
                    }
                    for (int c = 0; c < size; ++c) {
                        column_x[c] = cosine * T(c);
                        column_y[c] = -sine * T(c);
                    }
                    for (int r = 0; r < size; ++r) {
                        const T row_x = sine * T(r), row_y = cosine * T(r);
                        const auto x = [&](int c) { return (column_x[c] + row_x) + offset_x; };
                        const auto y = [&](int c) { return (column_y[c] + row_y) + offset_y; };
                        const auto [x_begin, x_end] = axis_span<false>(size, T(left - 1), T(right + 1), x);
                        const auto [y_begin, y_end] = axis_span<false>(size, T(top - 1), T(bottom + 1), y);
                        const int stop = std::min(x_end, y_end);
                        int c = std::max(x_begin, y_begin);
#ifdef SKIMAGEXX_TOMOGRAPHY_SIMD
                        if constexpr (std::is_same_v<T, double>) {
                            if (vector_spans && stop - c >= 4) {
                                c = radon_span(square, pitch, column_x, column_y, row_x, row_y,
                                               offset_x, offset_y, sums, c, stop);
                            }
                        }
#endif
                        for (; c < stop; ++c) {
                            sums[c] += bilinear(square, pitch, y(c), x(c));
                        }
                    }
                }
                for (int c = 0; c < size; ++c) {
                    auto* out = sinogram + std::size_t(c) * angles + first;
                    for (auto k = first; k < last; ++k) out[k - first] = tile[(k - first) * size + c];
                }
            }
        });
}

template<class T>
void iradon_into(const T* sinogram, int detectors, int angles, const double* cosines,
                 const double* sines, int size, TomographyFilter filter,
                 TomographyInterpolation interpolation, bool circle, T* image) {
    const int padded = radon_detector_size(detectors, detectors, !circle);
    const int workers = detail::active_workers(std::max(std::size_t(padded), std::size_t(size) * size) * angles);
    const auto projections = filtered_projections(sinogram, detectors, angles, padded, filter, workers);
    switch (interpolation) {
    case TomographyInterpolation::linear:
        backproject<TomographyInterpolation::linear>(projections, padded, angles, cosines, sines, size, circle, image, workers);
        break;
    case TomographyInterpolation::nearest:
        backproject<TomographyInterpolation::nearest>(projections, padded, angles, cosines, sines, size, circle, image, workers);
        break;
    case TomographyInterpolation::cubic:
        backproject<TomographyInterpolation::cubic>(projections, padded, angles, cosines, sines, size, circle, image, workers);
        break;
    }
}

template<class T>
void iradon_sart_into(const T* sinogram, int detectors, int angles, const T* theta,
                      const T* shifts, T relaxation, bool weak_scalars, T* image,
                      const T* clip, T* clipped_image) {
    const auto count = std::size_t(detectors) * detectors;
    const ComponentsView view{theta, std::is_same_v<T, float> ? PixelType::float32 : PixelType::float64,
                              false, false, {std::size_t(angles)}, {sizeof(T)}};
    const auto order = order_angles_golden_ratio(view, weak_scalars
        ? GoldenScalarPolicy::weak_input : GoldenScalarPolicy::wide);
    std::vector<T> update(count);
    const std::unique_ptr<RaySample<T>[]> samples(new RaySample<T>[std::size_t(detectors) * 2 + 1]);
    const T center = T(detectors / 2), radius = T(detectors / 2 - 1);
    RayWindows<T> temporary_windows;
    const RayWindows<T>* windows;
    // Retain one small, unshifted table per thread for repeated reconstructions.
    // Other rounding modes and shifted projections build their own exact table.
    if (detectors <= 256 && std::fegetround() == FE_TONEAREST &&
        std::all_of(shifts, shifts + angles, [](T shift) { return shift == T(0); })) {
        static thread_local RayWindows<T> unshifted;
        if (unshifted.offsets.size() != std::size_t(detectors) + 1) {
            unshifted = ray_windows(shifts, angles, detectors, center, radius);
        }
        windows = &unshifted;
    } else {
        temporary_windows = ray_windows(shifts, angles, detectors, center, radius);
        windows = &temporary_windows;
    }
    struct ActiveRay {
        int detector;
        RayChord<T> chord;
    };
    std::vector<ActiveRay> rays;
    rays.reserve(detectors);
    const T* previous_shift = nullptr;
    constexpr std::size_t cache_budget = 16 * 1024;
    const auto input_address = reinterpret_cast<std::uintptr_t>(sinogram);
    const auto image_address = reinterpret_cast<std::uintptr_t>(image);
    const bool overlaps = input_address < image_address + count * sizeof(T) &&
        image_address < input_address + std::size_t(detectors) * angles * sizeof(T);
    const auto tile_size = overlaps ? std::size_t{1} : std::min(order.size(), std::clamp(
        cache_budget / (sizeof(T) * detectors), std::size_t{1}, std::size_t{16}));
    std::vector<T> projections(tile_size * detectors);
#ifdef SKIMAGEXX_TOMOGRAPHY_SIMD
    const bool vector_taps = std::is_same_v<T, double> && has_avx2();
#endif
    // Pack projections in golden-ratio order. Each still observes the image
    // after the preceding projection has completed its update and clipping.
    for (std::size_t first = 0; first < order.size(); first += tile_size) {
        const auto block = std::min(tile_size, order.size() - first);
        for (std::size_t i = 0; i < block; ++i) {
            const auto k = order[first + i];
            auto* projection = projections.data() + i * detectors;
            for (int detector = 0; detector < detectors; ++detector) {
                projection[detector] = sinogram[std::size_t(detector) * angles + k];
            }
        }
        for (std::size_t j = 0; j < block; ++j) {
            const auto k = order[first + j];
            const auto* projection = projections.data() + j * detectors;
            const T radians = T(theta[k] / 180.0 * pi);
            const double cosine = std::cos(double(radians)), sine = std::sin(double(radians));
            if (!previous_shift || *previous_shift != shifts[k]) {
                rays.clear();
                for (int detector = 0; detector < detectors; ++detector) {
                    const auto chord = ray_chord(T(detector) + shifts[k], center, radius);
                    if (chord.steps) rays.push_back({detector, chord});
                }
                previous_shift = shifts + k;
            }
            for (const auto& [detector, chord] : rays) {
                const auto* window = windows->values.data() + windows->offsets[std::size_t(chord.steps) / 2];
#ifdef SKIMAGEXX_TOMOGRAPHY_SIMD
                if constexpr (std::is_same_v<T, double>) {
                    if (vector_taps) {
                        project_ray<true>(image, update.data(), detectors, cosine, sine, chord,
                                          projection[detector], samples.get(), window);
                        continue;
                    }
                }
#endif
                project_ray<false>(image, update.data(), detectors, cosine, sine, chord,
                                  projection[detector], samples.get(), window);
            }
            for (std::size_t i = 0; i < count; ++i) {
                image[i] += relaxation * update[i];
                update[i] = T(0);
            }
            if (clip) {
                for (std::size_t i = 0; i < count; ++i) {
                    clipped_image[i] = std::min(std::max(image[i], clip[0]), clip[1]);
                }
                image = clipped_image;
            }
        }
    }
}

template void radon_into(const float*, int, int, const RadonRotation*, int, bool, float*);
template void radon_into(const double*, int, int, const RadonRotation*, int, bool, double*);
template void iradon_into(const float*, int, int, const double*, const double*, int,
                          TomographyFilter, TomographyInterpolation, bool, float*);
template void iradon_into(const double*, int, int, const double*, const double*, int,
                          TomographyFilter, TomographyInterpolation, bool, double*);
template void iradon_sart_into(const float*, int, int, const float*, const float*, float,
                               bool, float*, const float*, float*);
template void iradon_sart_into(const double*, int, int, const double*, const double*, double,
                               bool, double*, const double*, double*);
}
