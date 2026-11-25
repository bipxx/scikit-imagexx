#include "skimagexx/fisher_vector.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace skimagexx {
namespace {

class DiagonalDensity {
public:
    explicit DiagonalDensity(DiagonalMixtureView model)
        : model_(model), precision_(model.components * model.dimensions),
          normalization_(model.components) {}

    bool prepare() {
        constexpr double log_two_pi = 1.83787706640934548356;
        for (std::size_t k = 0; k < model_.components; ++k) {
            if (!(model_.weights[k] > 0.) || !std::isfinite(model_.weights[k]))
                return false;
            double log_det = 0.;
            for (std::size_t j = 0; j < model_.dimensions; ++j) {
                const auto index = k * model_.dimensions + j;
                const double cholesky = model_.precision_cholesky[index];
                if (!(cholesky > 0.) || !std::isfinite(cholesky) ||
                    !std::isfinite(model_.means[index]))
                    return false;
                precision_[index] = cholesky * cholesky;
                if (!std::isfinite(precision_[index])) return false;
                log_det += std::log(cholesky);
            }
            normalization_[k] = log_det + std::log(model_.weights[k]) -
                                0.5 * static_cast<double>(model_.dimensions) * log_two_pi;
        }
        return true;
    }

    double scores(const double* row, double* output) const {
        const auto dimensions = model_.dimensions;
        double peak = -std::numeric_limits<double>::infinity();
        for (std::size_t k = 0; k < model_.components; ++k) {
            const auto offset = k * dimensions;
            const double* mean = model_.means + offset;
            const double* precision = precision_.data() + offset;
            double distance = 0.;
            for (std::size_t j = 0; j < dimensions; ++j) {
                const double delta = row[j] - mean[j];
                distance += (delta * delta) * precision[j];
            }
            output[k] = normalization_[k] - 0.5 * distance;
            peak = std::max(peak, output[k]);
        }
        return peak;
    }

    double log_density(const double* row, double* output) const {
        const double peak = scores(row, output);
        if (model_.components == 1) return peak;
        double sum = 0.;
        for (std::size_t k = 0; k < model_.components; ++k)
            sum += std::exp(output[k] - peak);
        return peak + std::log(sum);
    }

private:
    DiagonalMixtureView model_;
    std::vector<double> precision_;
    std::vector<double> normalization_;
};

struct Moments {
    Moments(std::size_t components, std::size_t dimensions)
        : mass(components), first(components * dimensions),
          second(components * dimensions), squares(dimensions) {}

    template<class ProbabilityAt>
    void add(const double* row, ProbabilityAt probability_at) {
        const auto width = squares.size();
        for (std::size_t j = 0; j < width; ++j)
            squares[j] = row[j] * row[j];
        auto* first_row = first.data();
        auto* second_row = second.data();
        for (std::size_t k = 0; k < mass.size(); ++k, first_row += width, second_row += width) {
            const double probability = probability_at(k);
            mass[k] += probability;
            for (std::size_t j = 0; j < width; ++j) {
                first_row[j] += probability * row[j];
                second_row[j] += probability * squares[j];
            }
        }
    }

    std::vector<double> mass;
    std::vector<double> first;
    std::vector<double> second;
    std::vector<double> squares;
};

} // namespace

bool diagonal_gmm_expectation(const double* descriptors, std::size_t rows,
                              DiagonalMixtureView model, double* log_density,
                              double* log_responsibilities) {
    if (!rows || !model.components || !model.dimensions) return false;
    DiagonalDensity density(model);
    if (!density.prepare()) return false;
    for (std::size_t i = 0; i < rows; ++i) {
        auto* scores = log_responsibilities + i * model.components;
        const double total = density.log_density(descriptors + i * model.dimensions, scores);
        if (!std::isfinite(total)) return false;
        log_density[i] = total;
        for (std::size_t k = 0; k < model.components; ++k)
            scores[k] -= total;
    }
    return true;
}

bool diagonal_gmm_maximization(const double* descriptors, std::size_t rows,
                               std::size_t dimensions, std::size_t components,
                               const double* log_responsibilities, double regularization,
                               double* weights, double* means, double* covariances,
                               double* precision_cholesky) {
    if (!rows || !dimensions || !components) return false;
    Moments moments(components, dimensions);
    for (std::size_t i = 0; i < rows; ++i) {
        const auto* log_row = log_responsibilities + i * components;
        moments.add(descriptors + i * dimensions, [log_row](std::size_t k) {
            return log_row[k] == 0. ? 1. : std::exp(log_row[k]);
        });
    }
    constexpr double prior = 10. * std::numeric_limits<double>::epsilon();
    for (auto& mass : moments.mass) mass += prior;
    const double mass = std::accumulate(moments.mass.begin(), moments.mass.end(), 0.);
    for (std::size_t k = 0; k < components; ++k) {
        weights[k] = moments.mass[k] / mass;
        for (std::size_t j = 0; j < dimensions; ++j) {
            const auto index = k * dimensions + j;
            const double mean = moments.first[index] / moments.mass[k];
            const double variance = moments.second[index] / moments.mass[k] - mean * mean;
            const double covariance = variance + regularization;
            if (!(covariance > 0.) || !std::isfinite(covariance)) return false;
            means[index] = mean;
            covariances[index] = covariance;
            precision_cholesky[index] = 1. / std::sqrt(covariance);
        }
    }
    return true;
}

bool fisher_vector(const double* descriptors, std::size_t rows, DiagonalMixtureView model,
                   const double* covariances, bool improved, double alpha, double* output) {
    if (!rows || !model.components || !model.dimensions) return false;
    const auto components = model.components;
    const auto dimensions = model.dimensions;
    const auto matrix_size = components * dimensions;
    DiagonalDensity density(model);
    if (!density.prepare()) return false;
    Moments moments(components, dimensions);
    std::vector<double> probabilities(components > 1 ? components : 0);
    for (std::size_t i = 0; i < rows; ++i) {
        const auto* row = descriptors + i * dimensions;
        if (components > 1) {
            const double peak = density.scores(row, probabilities.data());
            if (!std::isfinite(peak)) return false;
            double sum = 0.;
            for (double& score : probabilities) {
                score = std::exp(score - peak);
                sum += score;
            }
            moments.add(row, [&probabilities, sum](std::size_t k) {
                return probabilities[k] / sum;
            });
        } else {
            moments.add(row, [](std::size_t) { return 1.; });
        }
    }
    const double count = static_cast<double>(rows);
    const double sqrt_two = std::sqrt(2.);
    for (std::size_t k = 0; k < components; ++k) {
        const double weight = model.weights[k];
        const double sqrt_weight = std::sqrt(weight);
        const double mass = moments.mass[k] / count;
        output[k] = (mass - weight) / sqrt_weight;
        for (std::size_t j = 0; j < dimensions; ++j) {
            const auto index = k * dimensions + j;
            const double mean = model.means[index];
            const double covariance = covariances[index];
            if (!(covariance > 0.) || !std::isfinite(covariance)) return false;
            const double first = moments.first[index] / count;
            const double second = moments.second[index] / count;
            output[components + index] = (first - mass * mean) /
                                         (sqrt_weight * std::sqrt(covariance));
            output[components + matrix_size + index] =
                (-second - mass * (mean * mean) + mass * covariance + 2. * first * mean) /
                (sqrt_two * sqrt_weight * covariance);
        }
    }
    double norm_squared = 0., max_gradient = 0.;
    const auto size = components + 2 * matrix_size;
    for (std::size_t i = 0; i < size; ++i) {
        if (!std::isfinite(output[i])) return false;
        if (improved) {
            const double value = output[i];
            const double magnitude = std::abs(value);
            max_gradient = std::max(max_gradient, magnitude);
            const double scaled = alpha == 0.5 ? std::sqrt(magnitude) : std::pow(magnitude, alpha);
            output[i] = value == 0. ? 0. : std::copysign(scaled, value);
            norm_squared += output[i] * output[i];
        }
    }
    if (improved) {
        const double norm = std::sqrt(norm_squared);
        if (!(max_gradient > 1e-12) || !(norm > 0.) || !std::isfinite(norm)) return false;
        for (std::size_t i = 0; i < size; ++i) output[i] /= norm;
    }
    return true;
}

} // namespace skimagexx
