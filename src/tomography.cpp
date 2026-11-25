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
#include <type_traits>
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
            const double step = 2 * pi / double(size - 1);
            const double mean_cosine = (std::cos(step * double(shifted)) +
                                        std::cos(step * double(opposite))) / 2;
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
                                         int padded, TomographyFilter filter) {
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
    if constexpr (std::is_same_v<T, double>) {
        detail::fft_r2c(shape, axes, spatial.data(), spectrum.data(), 1);
    } else {
        std::vector<std::complex<T>> single(spectrum.size());
        detail::fft_r2c(shape, axes, spatial.data(), single.data(), 1);
        std::copy(single.begin(), single.end(), spectrum.begin());
    }
    for (int k = 0; k < angles; ++k) {
        auto* column = spectrum.data() + std::size_t(k) * bins;
        for (std::size_t f = 0; f < bins; ++f) column[f] *= response[f];
    }
    // Keep a compact prefix after inversion, one contiguous detector array per angle.
    std::vector<double> inverse(std::size_t(angles) * size);
    detail::fft_c2r(shape, axes, spectrum.data(), inverse.data(), 1);
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
                 const double* cosines, const double* sines, int size, bool circle, T* image) {
    std::vector<CubicSegment> segments;
    if constexpr (interpolation == TomographyInterpolation::cubic) {
        segments = cubic_segments(projections, detectors, angles);
    }
    const int radius = size / 2, center = detectors / 2;
    const double lower = -center, upper = detectors - 1 - center;
    const auto count = std::size_t(size) * size;
    std::fill_n(image, count, T(0));
    detail::partition(std::size_t(size), detail::active_workers(count * angles),
        [&](std::size_t begin, std::size_t end, std::size_t) {
            for (int k = 0; k < angles; ++k) {
                const auto* col = projections.data() + std::size_t(k) * detectors;
                const double cosine = cosines[k], sine = sines[k];
                for (auto r = begin; r < end; ++r) {
                    const double vertical = (double(r) - radius) * sine;
                    auto* row = image + r * size;
                    for (int x = -radius; x < size - radius; ++x) {
                        const double t = double(x) * cosine - vertical;
                        if (t < lower || t > upper) continue;
                        const int left = floor_index(t);
                        const int i = left + center;
                        const double fraction = t - left;
                        double value;
                        if constexpr (interpolation == TomographyInterpolation::nearest) {
                            value = col[i + (t > double(left) + 0.5)];
                        } else if (i == detectors - 1) {
                            value = col[i];
                        } else if constexpr (interpolation == TomographyInterpolation::cubic) {
                            const auto& p = segments[std::size_t(k) * (detectors - 1) + i];
                            value = p.value + fraction * (p.slope + fraction * (p.quadratic + fraction * p.cubic));
                        } else {
                            value = col[i] + fraction * (col[i + 1] - col[i]);
                        }
                        row[x + radius] = T(double(row[x + radius]) + value);
                    }
                }
            }
            for (auto r = begin; r < end; ++r) {
                const std::int64_t y = std::int64_t(r) - radius;
                for (int c = 0; c < size; ++c) {
                    const std::int64_t x = c - radius;
                    auto& value = image[r * size + c];
                    value = circle && x * x + y * y > std::int64_t(radius) * radius
                        ? T(0) : (value * T(pi)) / T(2.0 * angles);
                }
            }
        });
}

template<class T> struct RaySample {
    std::size_t base;
    T di, dj;
    bool has_top, has_left, has_bottom, has_right;
};

template<class T>
void project_ray(const T* image, T* update, int size, double cosine, double sine,
                 T ray_position, T projection, std::vector<RaySample<T>>& samples,
                 std::vector<std::vector<T>>& windows) {
    const T center = T(size / 2), radius = T(size / 2 - 1);
    const T t = ray_position - center;
    const T length_squared = radius * radius - t * t;
    if (!(length_squared > T(0))) return;
    const T half_length = T(std::sqrt(double(length_squared)));
    const int steps = 2 * static_cast<int>(std::ceil(2 * double(half_length)));
    const T ds = (T(2) * half_length) / T(steps);
    const T dx = T(-ds * cosine), dy = T(-ds * sine);
    const T x0 = T(half_length * cosine - t * sine);
    const T y0 = T(half_length * sine + t * cosine);
    auto& window = windows[std::size_t(steps) / 2];
    if (window.empty()) {
        window.resize(std::size_t(steps) + 1);
        const T beta = T(0.46164);
        for (int k = 0; k <= steps; ++k) {
            window[k] = T((T(1) - beta) - beta * std::cos(2 * pi * k / (steps - 1)));
        }
    }
    samples.clear();
    T total = 0, norm = 0;
    const auto accumulate = [&](std::size_t index, T weight) {
        total += weight * image[index];
        norm += weight * weight;
    };
    for (int k = 0; k <= steps; ++k) {
        const T row = (x0 + T(k) * dx) + center, col = (y0 + T(k) * dy) + center;
        // The chord lies inside the detector grid. Its endpoint can land on
        // the last grid line, where there is no bilinear neighbor beyond it.
        const int i = static_cast<int>(row), j = static_cast<int>(col);
        const T di = row - T(i), dj = col - T(j);
        const auto base = std::size_t(i) * size + j;
        samples.push_back({base, di, dj, i > 0, j > 0, i < size - 1, j < size - 1});
        if (i > 0 && j > 0) accumulate(base, T((1.0 - di) * (1.0 - dj) * ds));
        if (i > 0 && j < size - 1) accumulate(base + 1, T((1.0 - di) * dj * ds));
        if (i < size - 1 && j > 0) accumulate(base + size, di * (T(1) - dj) * ds);
        if (i < size - 1 && j < size - 1) accumulate(base + size + 1, di * dj * ds);
    }
    const T deviation = norm > T(0) ? -(total - projection) / norm : T(0);
    for (int k = 0; k <= steps; ++k) {
        const auto& sample = samples[k];
        const auto base = sample.base;
        const T di = sample.di, dj = sample.dj, w = window[k];
        if (sample.has_top && sample.has_left) {
            update[base] = T(update[base] + deviation * (1.0 - di) * (1.0 - dj) * ds * w);
        }
        if (sample.has_top && sample.has_right) {
            update[base + 1] = T(update[base + 1] + deviation * (1.0 - di) * dj * ds * w);
        }
        if (sample.has_bottom && sample.has_left) {
            update[base + size] += deviation * di * (T(1) - dj) * ds * w;
        }
        if (sample.has_bottom && sample.has_right) update[base + size + 1] += deviation * di * dj * ds * w;
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
                    for (int c = 0; c < size; ++c) {
                        sums[c] += bilinear(image, size, (column_y[c] + row_y) + offset_y,
                                                       (column_x[c] + row_x) + offset_x);
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
    const auto projections = filtered_projections(sinogram, detectors, angles, padded, filter);
    switch (interpolation) {
    case TomographyInterpolation::linear:
        backproject<TomographyInterpolation::linear>(projections, padded, angles, cosines, sines, size, circle, image);
        break;
    case TomographyInterpolation::nearest:
        backproject<TomographyInterpolation::nearest>(projections, padded, angles, cosines, sines, size, circle, image);
        break;
    case TomographyInterpolation::cubic:
        backproject<TomographyInterpolation::cubic>(projections, padded, angles, cosines, sines, size, circle, image);
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
    std::vector<RaySample<T>> samples;
    samples.reserve(std::size_t(detectors) * 2 + 1);
    std::vector<std::vector<T>> windows(std::size_t(detectors) + 1);
    for (const auto k : order) {
        const T radians = T(theta[k] / 180.0 * pi);
        const double cosine = std::cos(double(radians)), sine = std::sin(double(radians));
        std::fill(update.begin(), update.end(), T(0));
        for (int detector = 0; detector < detectors; ++detector) {
            project_ray(image, update.data(), detectors, cosine, sine, T(detector) + shifts[k],
                        sinogram[std::size_t(detector) * angles + k], samples, windows);
        }
        for (std::size_t i = 0; i < count; ++i) image[i] += relaxation * update[i];
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
