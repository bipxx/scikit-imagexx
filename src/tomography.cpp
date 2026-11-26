// Tomography conventions follow scikit-image 0.26 (BSD-3-Clause).
// See third_party/scikit-image-LICENSE.txt.
#include <skimagexx/tomography.hpp>
#include <skimagexx/order_angles_golden_ratio.hpp>
#include "execution.hpp"
#include "fft_internal.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

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

template<bool inclusive, class T, class Coordinate>
std::pair<int, int> axis_span(int size, T lower, T upper, const Coordinate& coordinate) {
    const auto below = [=](T value) { return inclusive ? value < lower : value <= lower; };
    const auto above = [=](T value) { return inclusive ? value > upper : value >= upper; };
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
T bilinear(const T* square, int size, T r, T c) {
    const int row = floor_index(r);
    const int column = floor_index(c);
    const T dr = r - T(row), dc = c - T(column);
    T tl, tr, bl, br;
    if (unsigned(row) < unsigned(size - 1) && unsigned(column) < unsigned(size - 1)) {
        const auto* top = square + std::size_t(row) * size + column;
        tl = top[0];
        tr = top[1];
        bl = top[size];
        br = top[size + 1];
    } else {
        const auto read = [&](int y, int x) {
            if (unsigned(y) >= unsigned(size) || unsigned(x) >= unsigned(size)) return T(0);
            return square[std::size_t(y) * size + x];
        };
        tl = read(row, column);
        tr = read(row, column + 1);
        bl = read(row + 1, column);
        br = read(row + 1, column + 1);
    }
    const double top = (T(1) - dc) * tl + dc * tr;
    const double bottom = (T(1) - dc) * bl + dc * br;
    return T((T(1) - dr) * top + dr * bottom);
}

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
    // original order. Reuse the column coordinates for all rows of a strip.
    constexpr std::size_t strip_height = 16;
    const auto strips = (std::size_t(size) + strip_height - 1) / strip_height;
    const int strip_workers = static_cast<int>(std::min(strips, std::size_t(workers)));
    const std::unique_ptr<double[]> horizontal(new double[std::size_t(strip_workers) * size]);
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
                    for (int c = first_column; c < last_column; ++c) {
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
                const std::int64_t y = std::int64_t(r) - radius;
                auto* row = image + r * size;
                for (int c = 0; c < size; ++c) {
                    const std::int64_t x = c - radius;
                    row[c] = circle && x * x + y * y > radius_squared
                        ? T(0) : (row[c] * T(pi)) / scale;
                }
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
std::vector<T> ray_windows(const T* shifts, int angles, int detectors, T center, T radius,
                           std::vector<std::size_t>& offsets) {
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
    std::vector<T> windows;
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
    return windows;
}

template<bool interior, class T>
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
    // The circular chord already clips the ray to the grid. Its sampled
    // endpoints can touch the final grid line, so only interior chords can
    // drop the neighbor predicates.
    if constexpr (interior) {
        const T row_first = x0 + center, row_last = (x0 + T(steps) * dx) + center;
        const T col_first = y0 + center, col_last = (y0 + T(steps) * dy) + center;
        const T last = T(size - 1);
        const auto inside = [last](T a, T b) {
            return std::min(a, b) >= T(1) && std::max(a, b) < last;
        };
        if (!inside(row_first, row_last) || !inside(col_first, col_last)) {
            project_ray<false>(image, update, size, cosine, sine, chord, projection, samples, window);
            return;
        }
    }
    for (int k = 0; k <= steps; ++k) {
        const T row = (x0 + T(k) * dx) + center, col = (y0 + T(k) * dy) + center;
        // The chord lies inside the detector grid. Its endpoint can land on
        // the last grid line, where there is no bilinear neighbor beyond it.
        const int i = static_cast<int>(row), j = static_cast<int>(col);
        const T di = row - T(i), dj = col - T(j);
        const auto base = std::size_t(i) * size + j;
        auto& sample = samples[k];
        sample.base = base;
        sample.di = di;
        sample.dj = dj;
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
    const T deviation = norm > T(0) ? -(total - projection) / norm : T(0);
    for (int k = 0; k <= steps; ++k) {
        const auto& sample = samples[k];
        const auto base = sample.base;
        const T di = sample.di, dj = sample.dj, w = window[k];
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
    std::vector<T> prepared;
    if (rows != size || columns != size) {
        prepared.resize(std::size_t(size) * size);
        if (circle) {
            const int r0 = (rows - size + 1) / 2, c0 = (columns - size + 1) / 2;
            for (int r = 0; r < size; ++r) {
                std::copy_n(image + std::size_t(r + r0) * columns + c0, size,
                            prepared.data() + std::size_t(r) * size);
            }
        } else {
            const int r0 = size / 2 - rows / 2, c0 = size / 2 - columns / 2;
            for (int r = 0; r < rows; ++r) {
                std::copy_n(image + std::size_t(r) * columns, columns,
                            prepared.data() + std::size_t(r + r0) * size + c0);
            }
        }
        image = prepared.data();
    }
    const int workers = detail::active_workers(std::size_t(size) * size * angles);
    std::vector<T> workspace(std::size_t(workers) * size * 3);
    detail::partition(std::size_t(angles), workers,
        [&](std::size_t begin, std::size_t end, std::size_t worker) {
            auto* sums = workspace.data() + worker * size * 3;
            auto* column_x = sums + size;
            auto* column_y = column_x + size;
            for (auto k = begin; k < end; ++k) {
                const auto& rotation = rotations[k];
                const T cosine = T(rotation.cosine), sine = T(rotation.sine);
                const T offset_x = T(rotation.column_offset), offset_y = T(rotation.row_offset);
                std::fill_n(sums, size, T(0));
                for (int c = 0; c < size; ++c) {
                    column_x[c] = cosine * T(c);
                    column_y[c] = -sine * T(c);
                }
                for (int r = 0; r < size; ++r) {
                    const T row_x = sine * T(r), row_y = cosine * T(r);
                    const auto x = [&](int c) { return (column_x[c] + row_x) + offset_x; };
                    const auto y = [&](int c) { return (column_y[c] + row_y) + offset_y; };
                    const auto [x_begin, x_end] = axis_span<false>(size, T(-1), T(size), x);
                    const auto [y_begin, y_end] = axis_span<false>(size, T(-1), T(size), y);
                    const int end = std::min(x_end, y_end);
                    for (int c = std::max(x_begin, y_begin); c < end; ++c) {
                        sums[c] += bilinear(image, size, y(c), x(c));
                    }
                }
                for (int c = 0; c < size; ++c) sinogram[std::size_t(c) * angles + k] = sums[c];
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
    std::vector<std::size_t> window_offsets(std::size_t(detectors) + 1);
    const T center = T(detectors / 2), radius = T(detectors / 2 - 1);
    const auto windows = ray_windows(shifts, angles, detectors, center, radius, window_offsets);
    std::vector<RayChord<T>> chords(detectors);
    const T* previous_shift = nullptr;
    for (const auto k : order) {
        const T radians = T(theta[k] / 180.0 * pi);
        const double cosine = std::cos(double(radians)), sine = std::sin(double(radians));
        if (!previous_shift || *previous_shift != shifts[k]) {
            for (int detector = 0; detector < detectors; ++detector) {
                chords[detector] = ray_chord(T(detector) + shifts[k], center, radius);
            }
            previous_shift = shifts + k;
        }
        for (int detector = 0; detector < detectors; ++detector) {
            const auto& chord = chords[detector];
            if (!chord.steps) continue;
            const auto* window = windows.data() + window_offsets[std::size_t(chord.steps) / 2];
            project_ray<true>(image, update.data(), detectors, cosine, sine, chord,
                              sinogram[std::size_t(detector) * angles + k], samples.get(), window);
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
