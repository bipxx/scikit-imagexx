#include "skimagexx/polynomial.hpp"

#include <algorithm>
#include <cmath>

namespace skimagexx {
namespace {

std::size_t terms(unsigned order) {
    return (order + 1) * (order + 2) / 2;
}

void powers(double base, unsigned degree, double* values) {
    values[0] = 1.0;
    for (unsigned i = 1; i <= degree; ++i) values[i] = values[i - 1] * base;
}

}  // namespace

bool polynomial_apply(const double* coords, std::size_t count, const double* params,
                      std::size_t coefficients, double* output) {
    if (coefficients == 0) return false;
    if (coefficients >= terms(polynomial_max_order + 1)) return false;
    unsigned order = 0;
    while (terms(order + 1) <= coefficients) ++order;

    bool finite = true;
    for (std::size_t i = 0; i < count; ++i) {
        const auto x = coords[2 * i];
        const auto y = coords[2 * i + 1];
        if (order == 1) {
            output[2 * i] = params[0] + params[1] * x + params[2] * y;
            output[2 * i + 1] = params[coefficients] +
                params[coefficients + 1] * x + params[coefficients + 2] * y;
        } else {
            double xs[polynomial_max_order + 1], ys[polynomial_max_order + 1];
            powers(x, order, xs);
            powers(y, order, ys);
            double u = 0.0;
            double v = 0.0;
            std::size_t term = 0;
            for (unsigned degree = 0; degree <= order; ++degree) {
                for (unsigned y_degree = 0; y_degree <= degree; ++y_degree, ++term) {
                    u += params[term] * xs[degree - y_degree] * ys[y_degree];
                    v += params[coefficients + term] * xs[degree - y_degree] * ys[y_degree];
                }
            }
            output[2 * i] = u;
            output[2 * i + 1] = v;
        }
        finite &= std::isfinite(output[2 * i]) & std::isfinite(output[2 * i + 1]);
    }
    return finite;
}

bool polynomial_constraints(const double* src, const double* dst,
                             const double* weights, std::size_t count, unsigned order,
                             double* equations) {
    if (order > polynomial_max_order || count == 0) return false;
    double maximum = 0.0;
    if (weights) {
        for (std::size_t i = 0; i < count; ++i) {
            if (!std::isfinite(weights[i]) || weights[i] < 0.0) return false;
            maximum = std::max(maximum, weights[i]);
        }
        if (maximum == 0.0) return false;
    }
    const auto coefficients = terms(order);
    const auto columns = 2 * coefficients + 1;
    bool finite = true;
    for (std::size_t i = 0; i < count; ++i) {
        double xs[polynomial_max_order + 1], ys[polynomial_max_order + 1];
        powers(src[2 * i], order, xs);
        powers(src[2 * i + 1], order, ys);
        auto* u = equations + i * columns;
        auto* v = equations + (count + i) * columns;
        std::fill_n(u + coefficients, coefficients, 0.0);
        std::fill_n(v, coefficients, 0.0);
        const double weight = weights ? std::sqrt(weights[i] / maximum) : 1.0;
        std::size_t term = 0;
        for (unsigned degree = 0; degree <= order; ++degree) {
            for (unsigned j = 0; j <= degree; ++j, ++term) {
                const double value = weight * (xs[degree - j] * ys[j]);
                u[term] = v[coefficients + term] = value;
                finite &= std::isfinite(value);
            }
        }
        u[columns - 1] = weight * dst[2 * i];
        v[columns - 1] = weight * dst[2 * i + 1];
        finite &= std::isfinite(u[columns - 1]) & std::isfinite(v[columns - 1]);
    }
    return finite;
}

}  // namespace skimagexx
