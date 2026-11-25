#include "skimagexx/thin_plate_spline.hpp"
#include "execution.hpp"

#include <atomic>
#include <cmath>
#include <limits>
#include <utility>

namespace skimagexx {
namespace {

bool bounded(const double* values, std::size_t count, double limit) {
    for (std::size_t i = 0; i < count; ++i)
        if (!std::isfinite(values[i]) || std::abs(values[i]) > limit) return false;
    return true;
}

double radial(double dx, double dy) {
    // Preserve the original Euclidean distance followed by squaring.
    const double r = std::sqrt(dx * dx + dy * dy);
    const double square = r * r;
    return square * std::log(square + 1e-8);
}

bool solve(std::vector<double>& matrix, std::vector<double>& rhs, std::size_t width) {
    const auto largest = std::max_element(matrix.begin(), matrix.end(),
        [](double a, double b) { return std::abs(a) < std::abs(b); });
    const double tolerance = std::abs(*largest) * width * std::numeric_limits<double>::epsilon();
    for (std::size_t column = 0; column < width; ++column) {
        std::size_t pivot = column;
        for (std::size_t row = column + 1; row < width; ++row)
            if (std::abs(matrix[row * width + column]) > std::abs(matrix[pivot * width + column]))
                pivot = row;
        if (std::abs(matrix[pivot * width + column]) <= tolerance) return false;
        if (pivot != column) {
            std::swap_ranges(matrix.begin() + pivot * width, matrix.begin() + (pivot + 1) * width,
                             matrix.begin() + column * width);
            for (std::size_t k = 0; k < 2; ++k) std::swap(rhs[pivot * 2 + k], rhs[column * 2 + k]);
        }
        for (std::size_t row = column + 1; row < width; ++row) {
            const double factor = matrix[row * width + column] / matrix[column * width + column];
            for (std::size_t k = column + 1; k < width; ++k)
                matrix[row * width + k] -= factor * matrix[column * width + k];
            for (std::size_t k = 0; k < 2; ++k) rhs[row * 2 + k] -= factor * rhs[column * 2 + k];
        }
    }
    for (std::size_t row = width; row-- > 0;) {
        for (std::size_t k = 0; k < 2; ++k) {
            for (std::size_t column = row + 1; column < width; ++column)
                rhs[row * 2 + k] -= matrix[row * width + column] * rhs[column * 2 + k];
            rhs[row * 2 + k] /= matrix[row * width + row];
        }
    }
    return true;
}

} // namespace

bool thin_plate_spline_fit(const double* src, const double* dst, std::size_t count,
                           std::vector<double>& weights) {
    if (!src || !dst || count < 3 || count > thin_plate_spline_max_controls ||
        !bounded(src, 2 * count, 1e6) || !bounded(dst, 2 * count, 1e12)) return false;

    const auto width = count + 3;
    std::vector<double> matrix(width * width), rhs(width * 2);
    std::copy_n(dst, count * 2, rhs.begin());
    for (std::size_t i = 0; i < count; ++i) {
        for (std::size_t j = 0; j < i; ++j) {
            const auto kernel = static_cast<float>(radial(src[2 * i] - src[2 * j],
                                                          src[2 * i + 1] - src[2 * j + 1]));
            matrix[i * width + j] = matrix[j * width + i] = kernel;
        }
        matrix[i * width + count] = matrix[count * width + i] = 1;
        for (std::size_t k = 0; k < 2; ++k) {
            const auto value = static_cast<float>(src[2 * i + k]);
            matrix[i * width + count + k + 1] = matrix[(count + k + 1) * width + i] = value;
        }
    }
    if (!solve(matrix, rhs, width) || !bounded(rhs.data(), rhs.size(), 1e18)) return false;
    weights = std::move(rhs);
    return true;
}

bool thin_plate_spline_apply(const double* src, const double* weights, std::size_t count,
                             const double* coords, std::size_t rows, double* output) {
    if (!src || !weights || count < 3 || count > thin_plate_spline_max_controls ||
        rows > std::size_t(std::numeric_limits<std::ptrdiff_t>::max()) / (2 * sizeof(double)) ||
        (rows && (!coords || !output)) || !bounded(src, 2 * count, 1e6) ||
        !bounded(weights, 2 * (count + 3), 1e18)) return false;
    const auto* affine = weights + 2 * count;
    const auto workload = rows * std::min(count, std::size_t(16));
    std::atomic<bool> supported{true};
    detail::partition(rows, detail::active_workers(workload), [&](auto begin, auto end, auto) {
        for (auto row = begin; row < end; ++row) {
            const double x = coords[2 * row], y = coords[2 * row + 1];
            if (!std::isfinite(x) || !std::isfinite(y) || std::abs(x) > 1e12 || std::abs(y) > 1e12) {
                supported.store(false, std::memory_order_relaxed);
                return;
            }
            double sum_x = 0, sum_y = 0;
            for (std::size_t i = 0; i < count; ++i) {
                const double kernel = radial(x - src[2 * i], y - src[2 * i + 1]);
                sum_x += kernel * weights[2 * i];
                sum_y += kernel * weights[2 * i + 1];
            }
            output[2 * row] = (affine[0] + (affine[2] * x + affine[4] * y)) + sum_x;
            output[2 * row + 1] = (affine[1] + (affine[3] * x + affine[5] * y)) + sum_y;
        }
    });
    return supported.load(std::memory_order_relaxed);
}

} // namespace skimagexx
