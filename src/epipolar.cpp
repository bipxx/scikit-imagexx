#include "skimagexx/epipolar.hpp"

#include <algorithm>
#include <cmath>

namespace skimagexx {
namespace {

double dot(const double* row, double x, double y) {
    return row[0] * x + row[1] * y + row[2];
}

bool normalize(const double* points, std::size_t count, EpipolarScaling scaling,
               double* matrix) {
    std::fill_n(matrix, 9, 0.0);
    matrix[0] = matrix[4] = matrix[8] = 1.0;
    if (scaling == EpipolarScaling::raw) return true;
    if (count == 0) return false;

    double center_x = 0.0;
    double center_y = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
        center_x += points[2 * i];
        center_y += points[2 * i + 1];
    }
    const auto n = static_cast<double>(count);
    center_x /= n;
    center_y /= n;
    double sum = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
        const auto x = points[2 * i] - center_x;
        const auto y = points[2 * i + 1] - center_y;
        const auto squared_radius = x * x + y * y;
        sum += scaling == EpipolarScaling::rms ? squared_radius : std::sqrt(squared_radius);
    }
    const auto scale = scaling == EpipolarScaling::rms
        ? std::sqrt(sum / (2.0 * n))
        : sum / n / std::sqrt(2.0);
    if (!(scale > 0) || !std::isfinite(scale)) return false;
    matrix[0] = matrix[4] = 1.0 / scale;
    matrix[2] = -center_x / scale;
    matrix[5] = -center_y / scale;
    return std::isfinite(matrix[0]) && std::isfinite(matrix[2]) && std::isfinite(matrix[5]);
}

}  // namespace

bool epipolar_lines(const double* points, std::size_t count,
                    const double* matrix, double* output) {
    bool finite = true;
    for (std::size_t i = 0; i < count; ++i) {
        const auto x = points[2 * i];
        const auto y = points[2 * i + 1];
        const auto a = dot(matrix, x, y);
        const auto b = dot(matrix + 3, x, y);
        const auto c = dot(matrix + 6, x, y);
        output[3 * i] = a;
        output[3 * i + 1] = b;
        output[3 * i + 2] = c;
        finite &= std::isfinite(a) & std::isfinite(b) & std::isfinite(c);
    }
    return finite;
}

bool epipolar_residuals(const double* src, const double* dst, std::size_t count,
                        const double* matrix, double* output) {
    const double transpose[] = {matrix[0], matrix[3], matrix[6],
                                matrix[1], matrix[4], matrix[7]};
    bool finite = true;
    for (std::size_t i = 0; i < count; ++i) {
        const auto x = src[2 * i];
        const auto y = src[2 * i + 1];
        const auto u = dst[2 * i];
        const auto v = dst[2 * i + 1];
        const auto a = dot(matrix, x, y);
        const auto b = dot(matrix + 3, x, y);
        const auto c = dot(matrix + 6, x, y);
        const auto d = dot(transpose, u, v);
        const auto e = dot(transpose + 3, u, v);
        const auto denominator = std::sqrt(a * a + b * b + d * d + e * e);
        output[i] = std::abs(u * a + v * b + c) / denominator;
        finite &= std::isfinite(denominator) & std::isfinite(output[i]);
    }
    return finite;
}

bool epipolar_constraints(const double* src, const double* dst, std::size_t count,
                          EpipolarScaling scaling, double* constraints,
                          double* src_matrix, double* dst_matrix) {
    if (!normalize(src, count, scaling, src_matrix) ||
        !normalize(dst, count, scaling, dst_matrix)) return false;

    bool finite = true;
    for (std::size_t i = 0; i < count; ++i) {
        const auto x = dot(src_matrix, src[2 * i], src[2 * i + 1]);
        const auto y = dot(src_matrix + 3, src[2 * i], src[2 * i + 1]);
        const auto u = dot(dst_matrix, dst[2 * i], dst[2 * i + 1]);
        const auto v = dot(dst_matrix + 3, dst[2 * i], dst[2 * i + 1]);
        const double row[] = {u * x, u * y, u, v * x, v * y, v, x, y, 1.0};
        std::copy_n(row, 9, constraints + 9 * i);
        for (const auto value : row) finite &= std::isfinite(value);
    }
    return finite;
}

void epipolar_matrix(const double* u, const double* singular_values, const double* vt,
                     const double* src_matrix, const double* dst_matrix,
                     bool essential, double* matrix) {
    const auto mean = (singular_values[0] + singular_values[1]) / 2.0;
    const auto s0 = essential ? mean : singular_values[0];
    const auto s1 = essential ? mean : singular_values[1];
    double projected[9];
    double unconditioned[9];
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t column = 0; column < 3; ++column) {
            projected[3 * row + column] =
                u[3 * row] * s0 * vt[column] + u[3 * row + 1] * s1 * vt[3 + column];
        }
    }
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t column = 0; column < 3; ++column) {
            unconditioned[3 * row + column] =
                dst_matrix[row] * projected[column] +
                dst_matrix[3 + row] * projected[3 + column] +
                dst_matrix[6 + row] * projected[6 + column];
        }
    }
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t column = 0; column < 3; ++column) {
            matrix[3 * row + column] =
                unconditioned[3 * row] * src_matrix[column] +
                unconditioned[3 * row + 1] * src_matrix[3 + column] +
                unconditioned[3 * row + 2] * src_matrix[6 + column];
        }
    }
}

}  // namespace skimagexx
