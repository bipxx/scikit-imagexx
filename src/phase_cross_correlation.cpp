#include "skimagexx/phase_cross_correlation.hpp"

#include "fft_internal.hpp"
#include "skimagexx/image.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

namespace skimagexx {
namespace {

bool valid_shape(std::size_t height, std::size_t width, PhaseCorrelationOptions options) {
    return height && width && height <= phase_correlation_max_axis &&
           width <= phase_correlation_max_axis && height <= phase_correlation_max_samples / width &&
           options.upsample_factor >= 1 &&
           options.upsample_factor <= phase_correlation_max_upsample;
}

template<class T>
T power(const std::complex<T>& value) {
    return value.real() * value.real() + value.imag() * value.imag();
}

template<class T>
T spectral_power(const std::complex<T>* values, std::size_t size) {
    // Eight accumulators and balanced blocks preserve the precision of the
    // contiguous NumPy reduction, in particular for float32 roundoff near zero error.
    if (size < 8) {
        T sum = -T(0);
        for (std::size_t i = 0; i < size; ++i) sum += power(values[i]);
        return sum;
    }
    if (size > 128) {
        const std::size_t midpoint = (size / 2) / 8 * 8;
        return spectral_power(values, midpoint) + spectral_power(values + midpoint, size - midpoint);
    }
    std::array<T, 8> lanes{};
    for (std::size_t i = 0; i < 8; ++i) lanes[i] = power(values[i]);
    const std::size_t end = size / 8 * 8;
    for (std::size_t i = 8; i < end; i += 8)
        for (std::size_t j = 0; j < 8; ++j) lanes[j] += power(values[i + j]);
    T sum = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) +
            ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
    for (std::size_t i = end; i < size; ++i) sum += power(values[i]);
    return sum;
}

template<class T>
std::vector<std::complex<T>> dft_kernel(
    std::size_t length, std::size_t region, int factor, T shift) {
    constexpr double tau = 6.283185307179586476925286766559;
    const T offset = static_cast<T>(region / 2) - shift * static_cast<T>(factor);
    const auto length_signed = static_cast<std::ptrdiff_t>(length);
    const double step = 1.0 / (static_cast<double>(length) * factor);
    std::vector<std::complex<T>> kernel(region * length);
    for (std::size_t k = 0; k < length; ++k) {
        auto frequency = static_cast<std::ptrdiff_t>(k);
        if (k >= (length + 1) / 2) frequency -= length_signed;
        const double delta = tau * (static_cast<double>(frequency) * step);
        const double rotation_real = std::cos(delta);
        const double rotation_imag = std::sin(delta);
        double real = std::cos(-static_cast<double>(offset) * delta);
        double imag = std::sin(-static_cast<double>(offset) * delta);
        for (std::size_t sample = 0; sample < region; ++sample) {
            kernel[sample * length + k] = {static_cast<T>(real), static_cast<T>(imag)};
            const double next_real = real * rotation_real - imag * rotation_imag;
            imag = real * rotation_imag + imag * rotation_real;
            real = next_real;
        }
    }
    return kernel;
}

template<std::size_t dots, bool shared_kernel = false, class T>
std::array<std::complex<T>, dots> dft_dots(const std::complex<T>* kernels,
                                          const std::complex<T>* values,
                                          std::size_t length, std::size_t stride) {
    std::array<T, dots> real{}, imag{};
    for (std::size_t k = 0; k < length; ++k) {
        const T a = values[k].real(), b = values[k].imag();
        for (std::size_t d = 0; d < dots; ++d) {
            const auto weight = kernels[d * stride + k];
            if constexpr (shared_kernel) {
                real[d] += a * weight.real() - b * weight.imag();
                imag[d] += a * weight.imag() + b * weight.real();
            } else {
                real[d] += weight.real() * a - weight.imag() * b;
                imag[d] += weight.real() * b + weight.imag() * a;
            }
        }
    }
    std::array<std::complex<T>, dots> result;
    for (std::size_t d = 0; d < dots; ++d) result[d] = {real[d], imag[d]};
    return result;
}

template<class T>
std::complex<T> refine_peak(const std::complex<T>* spectrum, std::size_t height,
                            std::size_t width, int factor, std::array<T, 2>& shift,
                            bool packed) {
    const auto region = static_cast<std::size_t>((3 * factor + 1) / 2);
    const auto kernels_x = dft_kernel(width, region, factor, shift[1]);
    const auto kernels_y = dft_kernel(height, region, factor, shift[0]);
    // Contract the contiguous frequency axis first. The intermediate is
    // transposed so each second contraction also reads contiguous values.
    std::vector<std::complex<T>> intermediate(region * height);
    const auto stride = packed ? width / 2 + 1 : width;
    std::vector<std::complex<T>> unpacked_row(packed && stride < width ? width : 0);
    for (std::size_t y = 0; y < height; ++y) {
        const auto* row = spectrum + y * stride;
        if (!unpacked_row.empty()) {
            const auto* opposite = spectrum + (y == 0 ? 0 : height - y) * stride;
            std::copy_n(row, stride, unpacked_row.data());
            for (auto x = stride; x < width; ++x)
                unpacked_row[x] = std::conj(opposite[width - x]);
            row = unpacked_row.data();
        }
        // Reuse each spectrum sample across two outputs. Every dot still
        // accumulates its terms in ascending frequency order.
        std::size_t x = 0;
        for (; x + 1 < region; x += 2) {
            const auto pair = dft_dots<2>(kernels_x.data() + x * width, row, width, width);
            intermediate[x * height + y] = pair[0];
            intermediate[(x + 1) * height + y] = pair[1];
        }
        if (x < region)
            intermediate[x * height + y] = dft_dots<1>(kernels_x.data() + x * width, row, width, 0)[0];
    }
    std::array<std::size_t, 2> best{};
    std::complex<T> peak{};
    T best_magnitude = T(-1);
    T best_power = T(-1);
    for (std::size_t y = 0; y < region; ++y) {
        const auto* kernel = kernels_y.data() + y * height;
        const auto consider = [&](std::complex<T> sum, std::size_t x) {
            const T squared = power(sum);
            if (!(squared > best_power)) return;
            best_power = squared;
            // Preserve first-index ties when distinct powers round to the
            // same magnitude in the working dtype.
            const T magnitude = std::sqrt(squared);
            if (magnitude > best_magnitude) {
                best_magnitude = magnitude;
                best = {y, x};
                peak = sum;
            }
        };
        std::size_t x = 0;
        for (; x + 1 < region; x += 2) {
            const auto pair = dft_dots<2, true>(intermediate.data() + x * height, kernel, height, height);
            consider(pair[0], x);
            consider(pair[1], x + 1);
        }
        if (x < region)
            consider(dft_dots<1, true>(intermediate.data() + x * height, kernel, height, 0)[0], x);
    }
    for (std::size_t i = 0; i < shift.size(); ++i)
        shift[i] += (static_cast<T>(best[i]) - static_cast<T>(region / 2)) / static_cast<T>(factor);
    return peak;
}

template<class T>
T signed_shift(std::size_t index, std::size_t length) {
    const T shift = static_cast<T>(index);
    return index > length / 2 ? shift - static_cast<T>(length) : shift;
}

template<class T>
T packed_power(const std::complex<T>* input, std::size_t height, std::size_t width,
               std::size_t stride) {
    const auto half = width / 2 + 1;
    double sum = 0;
    for (std::size_t y = 0; y < height; ++y) {
        const auto* row = input + y * stride;
        sum += power(row[0]);
        for (std::size_t x = 1; x < (width + 1) / 2; ++x)
            sum += 2 * static_cast<double>(power(row[x]));
        if (width % 2 == 0)
            sum += power(row[half - 1]);
    }
    return static_cast<T>(sum);
}

template<class T>
std::complex<T> cross_power(std::complex<T> a, std::complex<T> b, bool normalize) {
    T real = a.real() * b.real() + a.imag() * b.imag();
    T imag = a.imag() * b.real() - a.real() * b.imag();
    if (normalize) {
        constexpr T floor = T(100) * std::numeric_limits<T>::epsilon();
        const T magnitude = std::max(std::sqrt(real * real + imag * imag), floor);
        real /= magnitude;
        imag /= magnitude;
    }
    return {real, imag};
}

template<class T>
std::size_t real_peak_index(const T* values, std::size_t size) {
    // Separate comparison chains, with one final first-index tie break.
    std::array<T, 4> largest;
    largest.fill(std::abs(values[0]));
    std::array<std::size_t, 4> indices{};
    std::size_t i = 1;
    for (; i + 4 <= size; i += 4) {
        for (std::size_t lane = 0; lane < 4; ++lane) {
            const T magnitude = std::abs(values[i + lane]);
            if (magnitude > largest[lane]) {
                largest[lane] = magnitude;
                indices[lane] = i + lane;
            }
        }
    }
    std::size_t best = 0;
    for (std::size_t lane = 1; lane < 4; ++lane) {
        if (largest[lane] > largest[best] ||
            (largest[lane] == largest[best] && indices[lane] < indices[best]))
            best = lane;
    }
    auto index = indices[best];
    T magnitude = largest[best];
    for (; i < size; ++i) {
        const T candidate = std::abs(values[i]);
        if (candidate > magnitude) {
            magnitude = candidate;
            index = i;
        }
    }
    return index;
}

template<class T>
bool correlate(const std::complex<T>* reference, const std::complex<T>* moving,
               const std::vector<std::size_t>& shape, int workers, PhaseCorrelationOptions options,
               PhaseCorrelationResult<T>& result, std::complex<T>* packed_spectrum = nullptr) {
    const std::size_t size = shape[0] * shape[1];
    const auto half = shape[1] / 2 + 1;
    const bool packed = packed_spectrum != nullptr;
    const bool refine = options.upsample_factor > 1;
    const T divisor = refine ? T(1) : static_cast<T>(size);
    const auto energy = [&](const std::complex<T>* input) {
        return packed ? packed_power(input, shape[0], shape[1], half) : spectral_power(input, size);
    };
    const T amp = (energy(reference) / divisor) * (energy(moving) / divisor);
    // Bound the elementwise products as well as the final peak so that
    // magnitudes can use a direct sum of squares.
    const T extent = static_cast<T>(size);
    if (amp == T(0) || !(amp <= std::numeric_limits<T>::max() / extent / extent / T(8)))
        return false;

    std::size_t index = 0;
    std::complex<T> peak{};
    std::vector<std::complex<T>> full_spectrum;
    const auto* spectrum = packed_spectrum;
    if (packed) {
        for (std::size_t i = 0; i < shape[0] * half; ++i)
            packed_spectrum[i] = cross_power(reference[i], moving[i], options.normalize_phase);
        // The inverse FFT writes every sample before the peak scan reads it.
        std::unique_ptr<T[]> correlation(new T[size]);
        detail::fft_c2r(shape, {0, 1}, spectrum, correlation.get(), workers);
        index = real_peak_index(correlation.get(), size);
        peak = {correlation[index], T(0)};
    } else {
        full_spectrum.reserve(size);
        for (std::size_t i = 0; i < size; ++i)
            full_spectrum.push_back(cross_power(reference[i], moving[i], options.normalize_phase));
        spectrum = full_spectrum.data();
        std::vector<std::complex<T>> correlation(size);
        detail::fft_c2c(shape, {0, 1}, spectrum, correlation.data(), false, workers);
        T best_magnitude = T(-1);
        T best_power = T(-1);
        for (std::size_t i = 0; i < size; ++i) {
            const T squared = power(correlation[i]);
            if (!(squared > best_power)) continue;
            best_power = squared;
            const T magnitude = std::sqrt(squared);
            if (magnitude > best_magnitude) {
                best_magnitude = magnitude;
                index = i;
            }
        }
        peak = correlation[index];
    }

    std::array<T, 2> shift{signed_shift<T>(index / shape[1], shape[0]),
                           signed_shift<T>(index % shape[1], shape[1])};
    if (refine)
        peak = refine_peak(spectrum, shape[0], shape[1], options.upsample_factor, shift, packed);
    // A real inverse cannot recover the side of the branch cut chosen by
    // the small imaginary component of a complex inverse.
    if (packed && std::signbit(peak.real())) return false;
    for (std::size_t i = 0; i < shift.size(); ++i)
        if (shape[i] == 1) shift[i] = T(0);
    const T ratio = power(peak) / amp;
    if (!std::isfinite(ratio)) return false;
    result = {shift, std::sqrt(std::abs(1.0 - static_cast<double>(ratio))),
              std::atan2(peak.imag(), peak.real())};
    return true;
}

template<class T>
bool cyclic_translation(const T* reference, const T* moving, std::size_t height,
                        std::size_t width, std::array<T, 2>& shift) {
    const auto count = height * width;
    if (reference == moving) {
        // Self-registration does not need a pair of spatial maximum scans.
        shift = {T(0), T(0)};
        return std::none_of(reference, reference + count, [](T value) { return std::isnan(value); });
    }
    const auto a = static_cast<std::size_t>(std::max_element(reference, reference + count) - reference);
    const auto b = static_cast<std::size_t>(std::max_element(moving, moving + count) - moving);
    if (reference[a] != moving[b]) return false;
    const auto dy = (b / width + height - a / width) % height;
    const auto dx = (b % width + width - a % width) % width;
    for (std::size_t y = 0; y < height; ++y) {
        const T* from = reference + y * width;
        const auto next_y = y + dy < height ? y + dy : y + dy - height;
        const T* to = moving + next_y * width;
        if (!std::equal(from, from + width - dx, to + dx) ||
            !std::equal(from + width - dx, from + width, to))
            return false;
    }
    shift = {signed_shift<T>((height - dy) % height, height),
             signed_shift<T>((width - dx) % width, width)};
    return true;
}

template<class T>
bool exact_phase_ramp(const std::complex<T>* reference, const std::complex<T>* moving,
                      std::size_t height, std::size_t width, int factor,
                      PhaseCorrelationResult<T>& result) {
    if (height < 3 || width < 3) return false;
    constexpr double tau = 6.283185307179586476925286766559;
    constexpr T floor = T(100) * std::numeric_limits<T>::epsilon();
    constexpr T tolerance = T(64) * std::numeric_limits<T>::epsilon();
    const auto shift_for_axis = [&](std::size_t stride, std::size_t length) {
        const auto value = cross_power(reference[stride], moving[stride], false);
        const auto offset = -std::atan2(double(value.imag()), double(value.real())) *
                            (static_cast<double>(length) / tau);
        return std::round(offset * factor) / factor;
    };
    const double y = shift_for_axis(width, height);
    const double x = shift_for_axis(1, width);
    if (!std::isfinite(x) || !std::isfinite(y)) return false;
    if (factor > 1 && (std::abs(y) >= static_cast<double>(height) / 2 - 0.5 ||
                       std::abs(x) >= static_cast<double>(width) / 2 - 0.5))
        return false;
    const auto rotation = [&](std::size_t length, double shift) {
        std::vector<std::complex<T>> values(length);
        auto frequency = std::ptrdiff_t{0};
        const double step = (tau / static_cast<double>(length)) * shift;
        for (std::size_t i = 0; i < length; ++i, ++frequency) {
            if (i == (length + 1) / 2)
                frequency -= static_cast<std::ptrdiff_t>(length);
            const double angle = static_cast<double>(frequency) * step;
            values[i] = {static_cast<T>(std::cos(angle)), static_cast<T>(std::sin(angle))};
        }
        return values;
    };
    const auto rows = rotation(height, y);
    const auto cols = rotation(width, x);
    // A complete, floor-free phase ramp has its unique peak at the verified
    // grid shift. Complex spectra that fail this check use the general FFT.
    for (std::size_t r = 0; r < height; ++r) {
        for (std::size_t c = 0; c < width; ++c) {
            const auto i = r * width + c;
            const auto product = cross_power(reference[i], moving[i], false);
            const T angle_real = rows[r].real() * cols[c].real() - rows[r].imag() * cols[c].imag();
            const T angle_imag = rows[r].real() * cols[c].imag() + rows[r].imag() * cols[c].real();
            const T aligned_real = product.real() * angle_real - product.imag() * angle_imag;
            const T aligned_imag = product.real() * angle_imag + product.imag() * angle_real;
            if (!(aligned_real > floor && std::abs(aligned_imag) <= tolerance * aligned_real))
                return false;
        }
    }
    const auto size = height * width;
    const T divisor = factor == 1 ? static_cast<T>(size) : T(1);
    const T amp = (spectral_power(reference, size) / divisor) * (spectral_power(moving, size) / divisor);
    const T peak = static_cast<T>(size) / divisor;
    if (!(amp > T(0) && std::isfinite(amp))) return false;
    const auto positive_midpoint = [](double shift, std::size_t length) {
        const auto midpoint = static_cast<double>(length) / 2;
        return static_cast<T>(shift == -midpoint ? midpoint : shift);
    };
    result = {{positive_midpoint(y, height), positive_midpoint(x, width)},
              std::sqrt(std::abs(1.0 - static_cast<double>((peak * peak) / amp))), T(0)};
    return true;
}

} // namespace

template<class T>
bool phase_cross_correlation(
    const T* reference, const T* moving, std::size_t height, std::size_t width,
    PhaseCorrelationOptions options, PhaseCorrelationResult<T>& result) {
    if (!reference || !moving || !valid_shape(height, width, options)) return false;
    const std::vector<std::size_t> shape{height, width};
    const std::vector<std::size_t> axes{0, 1};
    const int workers = get_num_threads();
    const auto packed_size = height * (width / 2 + 1);
    std::vector<std::complex<T>> reference_spectrum(packed_size);
    detail::fft_r2c(shape, axes, reference, reference_spectrum.data(), workers);
    std::array<T, 2> shift{};
    if (options.normalize_phase && cyclic_translation(reference, moving, height, width, shift)) {
        constexpr T floor = T(100) * std::numeric_limits<T>::epsilon();
        if (std::all_of(reference_spectrum.begin(), reference_spectrum.end(),
                        [&](const auto& bin) { return power(bin) > floor; })) {
            const T count = static_cast<T>(height * width);
            const T energy = packed_power(reference_spectrum.data(), height, width, width / 2 + 1);
            const T scaled = options.upsample_factor == 1 ? energy / count : energy;
            if (scaled > T(0) && scaled <= std::sqrt(std::numeric_limits<T>::max())) {
                const T inverse_rms = count / energy;
                result = {shift, std::sqrt(std::abs(1.0 - double(inverse_rms * inverse_rms))), T(0)};
                return true;
            }
        }
    }
    std::vector<std::complex<T>> moving_spectrum(packed_size);
    detail::fft_r2c(shape, axes, moving, moving_spectrum.data(), workers);
    return correlate(reference_spectrum.data(), moving_spectrum.data(), shape, workers, options,
                     result, moving_spectrum.data());
}

template<class T>
bool phase_cross_correlation_fourier(
    const std::complex<T>* reference, const std::complex<T>* moving,
    std::size_t height, std::size_t width, PhaseCorrelationOptions options,
    PhaseCorrelationResult<T>& result) {
    if (!reference || !moving || !valid_shape(height, width, options)) return false;
    if (options.normalize_phase && exact_phase_ramp(reference, moving, height, width,
                                                    options.upsample_factor, result))
        return true;
    return correlate(reference, moving, {height, width}, get_num_threads(), options, result);
}

template bool phase_cross_correlation(
    const float*, const float*, std::size_t, std::size_t, PhaseCorrelationOptions,
    PhaseCorrelationResult<float>&);
template bool phase_cross_correlation(
    const double*, const double*, std::size_t, std::size_t, PhaseCorrelationOptions,
    PhaseCorrelationResult<double>&);
template bool phase_cross_correlation_fourier(
    const std::complex<float>*, const std::complex<float>*, std::size_t, std::size_t,
    PhaseCorrelationOptions, PhaseCorrelationResult<float>&);
template bool phase_cross_correlation_fourier(
    const std::complex<double>*, const std::complex<double>*, std::size_t, std::size_t,
    PhaseCorrelationOptions, PhaseCorrelationResult<double>&);

} // namespace skimagexx
