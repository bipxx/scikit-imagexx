#include <skimagexx/restoration_extra.hpp>
#include "fft_internal.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <limits>
#include <stdexcept>

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
    const T weight = T(1) / T(2 * spatial_rank);
    const auto row_size = strides[spatial_rank - 1] * shape[spatial_rank - 1];
    RestorationShape before(spatial_rank), after(spatial_rank);
    for (std::size_t row = 0; row < size; row += row_size) {
        for (std::size_t axis = 0; axis + 1 < spatial_rank; ++axis) {
            const auto step = strides[axis], extent = shape[axis];
            const auto coord = row / step % extent;
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
    }
}

template<class T>
void restoration_roll(const T* image, const RestorationShape& shape,
                      const RestorationShape& shifts, T* output, bool add) {
    const auto size = size_of(shape);
    if (!image || !output || shifts.size() != shape.size())
        throw std::invalid_argument("invalid restoration roll buffers or rank");
    for (std::size_t axis = 0; axis < shape.size(); ++axis)
        if (shifts[axis] >= shape[axis])
            throw std::invalid_argument("restoration roll shift is not reduced");
    const auto strides = element_strides(shape);
    auto last = shape.size() - 1;
    // Coalesce every trailing unshifted axis with the last shifted axis.
    while (last > 0 && shifts[last] == 0) --last;
    const auto row_size = shape[last] * strides[last];
    const auto split = shifts[last] * strides[last];
    for (std::size_t row = 0; row < size; row += row_size) {
        std::size_t source = 0;
        for (std::size_t axis = 0; axis < last; ++axis) {
            const auto dest = row / strides[axis] % shape[axis];
            const auto shift = shifts[axis];
            source += (dest >= shift ? dest - shift : shape[axis] - (shift - dest)) *
                      strides[axis];
        }
        const auto tail = row_size - split;
        if (add) {
            for (std::size_t j = 0; j < split; ++j) output[row + j] += image[source + tail + j];
            for (std::size_t j = 0; j < tail; ++j) output[row + split + j] += image[source + j];
        } else {
            std::copy_n(image + source + tail, split, output + row);
            std::copy_n(image + source, tail, output + row + split);
        }
    }
}

#define INSTANTIATE(T) \
    template void restoration_transfer(const T*, const RestorationShape&, const RestorationShape&, std::complex<T>*, int); \
    template void restoration_laplacian(const RestorationShape&, T*); \
    template bool restoration_wiener(const T*, const RestorationShape&, const std::complex<T>*, const T*, T, bool, T*, int); \
    template void restoration_gibbs_sample(const std::complex<T>*, const std::complex<T>*, const T*, const double*, const double*, std::size_t, std::complex<T>*); \
    template void restoration_gibbs_step(const std::complex<T>*, const std::complex<T>*, const T*, const T*, const T*, T, T, const double*, std::size_t, std::complex<T>*, std::complex<T>*); \
    template void restoration_gibbs_forward(const T*, const RestorationShape&, std::complex<T>*, int); \
    template void restoration_gibbs_inverse(const std::complex<T>*, const RestorationShape&, T*, int); \
    template void restoration_interpolate(const T*, const RestorationShape&, std::size_t, T*); \
    template void restoration_roll(const T*, const RestorationShape&, const RestorationShape&, T*, bool)
INSTANTIATE(float);
INSTANTIATE(double);
#undef INSTANTIATE
} // namespace skimagexx
