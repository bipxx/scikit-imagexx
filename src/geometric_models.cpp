#include "skimagexx/geometric_models.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace skimagexx {
namespace {

using Pair = std::array<double, 2>;
using Triple = std::array<double, 3>;
using Matrix = std::array<Triple, 3>;
constexpr double epsilon = std::numeric_limits<double>::epsilon();
constexpr double pi = 3.141592653589793238462643383279502884;

struct Normalization {
    Pair origin{};
    double scale = 0;
};

bool normalize(const double* points, std::size_t count, Normalization& output) {
    Pair origin{};
    for (std::size_t i = 0; i < count; ++i) {
        origin[0] += points[2 * i];
        origin[1] += points[2 * i + 1];
    }
    const auto samples = static_cast<double>(count);
    origin[0] /= samples;
    origin[1] /= samples;
    double norm = 0;
    for (std::size_t i = 0; i < count; ++i) {
        const double x = points[2 * i] - origin[0], y = points[2 * i + 1] - origin[1];
        norm += x * x + y * y;
    }
    const double scale = std::sqrt(norm / (2 * samples));
    if (!std::isfinite(scale) || scale < std::numeric_limits<double>::min()) {
        return false;
    }
    output = {origin, scale};
    return true;
}

double dot(const Triple& a, const Triple& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

Triple cross(const Triple& a, const Triple& b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]};
}

bool invert(const Matrix& input, Matrix& result) {
    const Matrix columns{cross(input[1], input[2]), cross(input[2], input[0]),
                         cross(input[0], input[1])};
    const double determinant = dot(input[0], columns[0]);
    const double scale = std::max({input[0][0], input[1][1], input[2][2]});
    if (!(determinant > 64 * epsilon * scale * scale * scale)) {
        return false;
    }
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) result[i][j] = columns[j][i] / determinant;
    }
    return true;
}

// Solve the real 3-by-3 reduced problem. Only one eigenvector may satisfy
// the positive ellipse constraint; nearly repeated roots defer to LAPACK.
bool ellipse_eigenvector(Matrix matrix, Triple& result) {
    const double mean = (matrix[0][0] + matrix[1][1] + matrix[2][2]) / 3;
    for (int i = 0; i < 3; ++i) matrix[i][i] -= mean;
    const double p = (dot(matrix[0], {matrix[0][0], matrix[1][0], matrix[2][0]})
        + dot(matrix[1], {matrix[0][1], matrix[1][1], matrix[2][1]})
        + dot(matrix[2], {matrix[0][2], matrix[1][2], matrix[2][2]})) / 6;
    if (!(p > 0)) return false;
    const double root = std::sqrt(p);
    const double argument = dot(matrix[0], cross(matrix[1], matrix[2])) / (2 * p * root);
    if (!(std::abs(argument) < 1 - 64 * epsilon)) return false;
    const double angle = std::acos(argument) / 3;
    bool found = false;
    for (int k = 0; k < 3; ++k) {
        const double eigenvalue = 2 * root * std::cos(angle + k * (2 * pi / 3));
        auto shifted = matrix;
        for (int i = 0; i < 3; ++i) shifted[i][i] -= eigenvalue;
        auto vector = cross(shifted[0], shifted[1]);
        double squared = dot(vector, vector);
        for (int i = 0; i < 2; ++i) {
            const auto candidate = cross(shifted[i], shifted[2]);
            const auto length = dot(candidate, candidate);
            if (length > squared) {
                vector = candidate;
                squared = length;
            }
        }
        if (4 * vector[0] * vector[2] - vector[1] * vector[1] > 64 * epsilon * squared) {
            if (found) return false;
            const double norm = std::sqrt(squared);
            for (int i = 0; i < 3; ++i) result[i] = vector[i] / norm;
            found = true;
        }
    }
    return found;
}

struct EllipseCoefficients {
    double a_cos, a_sin, b_sin, b_cos;
};

EllipseCoefficients coefficients(EllipseParameters model) {
    const double c = std::cos(model.theta), s = std::sin(model.theta);
    return {model.axis_lengths[0] * c, model.axis_lengths[0] * s,
            model.axis_lengths[1] * s, model.axis_lengths[1] * c};
}

double objective(Pair point, Pair center, EllipseCoefficients e, double t) {
    const double c = std::cos(t), s = std::sin(t);
    const double x = point[0] - (center[0] + e.a_cos * c - e.b_sin * s);
    const double y = point[1] - (center[1] + e.a_sin * c + e.b_cos * s);
    return x * x + y * y;
}

// Scalar specialization of the reference's finite-difference
// Levenberg-Marquardt objective. A scalar damped linear solve gives the
// trust-region step directly when the Gauss-Newton step is too long.
bool ellipse_distance(Pair point, EllipseParameters model, EllipseCoefficients e, double& output) {
    constexpr double tolerance = 1.49012e-8;
    constexpr double difference = 1.490116119384765625e-8;
    double t = std::atan2(point[1] - model.center[1], point[0] - model.center[0]) - model.theta;
    double value = objective(point, model.center, e, t);
    if (!std::isfinite(value)) return false;
    double scale = 0, bound = 0;
    bool first = true;
    for (int evaluations = 1; evaluations < 400;) {
        const double h = t == 0 ? difference : difference * std::abs(t);
        const double jac = (objective(point, model.center, e, t + h) - value) / h;
        ++evaluations;
        if (jac == 0 || value == 0) {
            output = std::sqrt(value);
            return true;
        }
        scale = std::max(scale, std::abs(jac));
        if (first) {
            bound = 100 * std::abs(scale * t);
            if (bound == 0) bound = 100;
        }
        double ratio;
        do {
            const double gauss = -value / jac;
            const double scaled_gauss = std::abs(scale * gauss);
            const double step = scaled_gauss - bound <= 0.1 * bound
                ? gauss : std::copysign(bound / scale, gauss);
            const double scaled_step = std::abs(scale * step);
            if (first) bound = std::min(bound, scaled_step);
            const double next_t = t + step;
            const double next_value = objective(point, model.center, e, next_t);
            ++evaluations;
            const double relative = next_value / value;
            const double actual = 0.1 * next_value < value ? 1 - relative * relative : -1;
            const double linear = std::abs(jac * step) / value;
            const double damping = std::max(0.0, linear - linear * linear);
            const double predicted = linear * linear + 2 * damping;
            ratio = predicted == 0 ? 0 : actual / predicted;
            if (ratio <= 0.25) {
                const double derivative = -(linear * linear + damping);
                const double factor = actual < 0 ? 0.5 * derivative / (derivative + 0.5 * actual) : 0.5;
                bound = (0.1 * next_value >= value ? 0.1 : std::max(0.1, factor))
                    * std::min(bound, 10 * scaled_step);
            } else if (step == gauss || ratio >= 0.75) {
                bound = 2 * scaled_step;
            }
            if (ratio >= 0.0001) {
                t = next_t;
                value = next_value;
                first = false;
            }
            if ((std::abs(actual) <= tolerance && predicted <= tolerance && 0.5 * ratio <= 1)
                || bound <= tolerance * std::abs(scale * t) || value == 0) {
                output = std::sqrt(value);
                return true;
            }
        } while (ratio < 0.0001 && evaluations < 400);
    }
    return false;
}

}  // namespace

bool fit_circle(const double* points, std::size_t count, CircleParameters& output) {
    Normalization norm;
    if (count < 3 || !normalize(points, count, norm)) return false;
    double xx = 0, xy = 0, yy = 0, xf = 0, yf = 0, ff = 0, xs = 0, ys = 0;
    for (std::size_t i = 0; i < count; ++i) {
        const double x = (points[2 * i] - norm.origin[0]) / norm.scale;
        const double y = (points[2 * i + 1] - norm.origin[1]) / norm.scale;
        const double f = x * x + y * y;
        xx += x * x;
        xy += x * y;
        yy += y * y;
        xf += x * f;
        yf += y * f;
        ff += f;
        xs += x;
        ys += y;
    }
    const double n = static_cast<double>(count);
    xx -= xs * xs / n;
    yy -= ys * ys / n;
    xy -= xs * ys / n;
    xf -= xs * ff / n;
    yf -= ys * ff / n;
    const double determinant = xx * yy - xy * xy;
    if (!(determinant > 128 * epsilon * (xx + yy) * (xx + yy))) return false;
    const double x = (yy * xf - xy * yf) / (2 * determinant);
    const double y = (xx * yf - xy * xf) / (2 * determinant);
    const double radius = std::sqrt((ff - 2 * (x * xs + y * ys)) / n + x * x + y * y);
    output = {{x * norm.scale + norm.origin[0], y * norm.scale + norm.origin[1]},
              radius * norm.scale};
    return std::isfinite(output.radius);
}

bool fit_ellipse(const double* points, std::size_t count, EllipseParameters& output) {
    Normalization norm;
    if (count < 5 || !normalize(points, count, norm)) return false;
    Matrix s1{}, s2{}, s3{};
    for (std::size_t i = 0; i < count; ++i) {
        const double x = (points[2 * i] - norm.origin[0]) / norm.scale;
        const double y = (points[2 * i + 1] - norm.origin[1]) / norm.scale;
        const Triple quadratic{x * x, x * y, y * y}, linear{x, y, 1};
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                s1[r][c] += quadratic[r] * quadratic[c];
                s2[r][c] += quadratic[r] * linear[c];
                s3[r][c] += linear[r] * linear[c];
            }
        }
    }
    Matrix inverse;
    if (!invert(s3, inverse)) return false;
    Matrix transform;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) transform[r][c] = -dot(inverse[r], s2[c]);
    }
    Matrix reduced = s1;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            reduced[r][c] += dot(s2[r], {transform[0][c], transform[1][c], transform[2][c]});
        }
    }
    // Apply the inverse conic constraint matrix.
    for (int c = 0; c < 3; ++c) {
        std::swap(reduced[0][c], reduced[2][c]);
        reduced[0][c] *= 0.5;
        reduced[1][c] *= -1;
        reduced[2][c] *= 0.5;
    }
    Triple conic;
    if (!ellipse_eigenvector(reduced, conic)) return false;
    const double a = conic[0], b = 0.5 * conic[1], c = conic[2];
    const double d = 0.5 * dot(transform[0], conic), f = 0.5 * dot(transform[1], conic);
    const double g = dot(transform[2], conic);
    const double denominator = b * b - a * c;
    const double x = (c * d - b * f) / denominator, y = (a * f - b * d) / denominator;
    const double numerator = a * f * f + c * d * d + g * b * b - 2 * b * d * f - a * c * g;
    const double term = std::sqrt((a - c) * (a - c) + 4 * b * b);
    if (!(term > 1e-10)) return false;
    double width = std::sqrt(2 * numerator / (denominator * (term - (a + c))));
    double height = std::sqrt(2 * numerator / (denominator * (-term - (a + c))));
    double theta = 0.5 * std::atan((2 * b) / (a - c));
    if (a > c) theta += pi / 2;
    if (width < height) {
        std::swap(width, height);
        theta += pi / 2;
    }
    theta = std::fmod(theta, pi);
    if (theta < 0) theta += pi;
    output = {{x * norm.scale + norm.origin[0], y * norm.scale + norm.origin[1]},
              {width * norm.scale, height * norm.scale}, theta};
    return std::isfinite(width) && std::isfinite(height) && std::isfinite(theta)
        && std::isfinite(x) && std::isfinite(y);
}

void circle_residuals(const double* points, std::size_t count, CircleParameters model,
                      double* output) {
    for (std::size_t i = 0; i < count; ++i) {
        const double x = points[2 * i] - model.center[0], y = points[2 * i + 1] - model.center[1];
        output[i] = model.radius - std::sqrt(x * x + y * y);
    }
}

bool ellipse_residuals(const double* points, std::size_t count, EllipseParameters model,
                       double* output) {
    const auto e = coefficients(model);
    for (std::size_t i = 0; i < count; ++i) {
        if (!ellipse_distance({points[2 * i], points[2 * i + 1]}, model, e, output[i])) {
            return false;
        }
    }
    return true;
}

void predict_circle(const double* cosine, const double* sine, std::size_t count,
                    CircleParameters model, double* output) {
    for (std::size_t i = 0; i < count; ++i) {
        output[2 * i] = model.center[0] + model.radius * cosine[i];
        output[2 * i + 1] = model.center[1] + model.radius * sine[i];
    }
}

void predict_ellipse(const double* cosine, const double* sine, std::size_t count,
                     EllipseParameters model, double* output) {
    const auto e = coefficients(model);
    for (std::size_t i = 0; i < count; ++i) {
        output[2 * i] = model.center[0] + e.a_cos * cosine[i] - e.b_sin * sine[i];
        output[2 * i + 1] = model.center[1] + e.a_sin * cosine[i] + e.b_cos * sine[i];
    }
}

double ransac_trial_limit(std::size_t inliers, std::size_t samples, std::size_t minimum,
                          double probability) {
    if (probability == 0) return 0;
    if (!inliers) return std::numeric_limits<double>::infinity();
    const double fraction = static_cast<double>(inliers) / static_cast<double>(samples);
    const auto numerator = std::clamp(1 - probability, epsilon, 1 - epsilon);
    const auto denominator = std::clamp(1 - std::pow(fraction, static_cast<double>(minimum)),
                                        epsilon, 1 - epsilon);
    return std::ceil(std::log(numerator) / std::log(denominator));
}

}  // namespace skimagexx
