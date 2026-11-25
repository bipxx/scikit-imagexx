#include <skimagexx/fisher_vector.hpp>

#include <algorithm>
#include <cmath>

int main() {
    const double samples[] = {-1., 1., 3.};
    const double weights[] = {1.};
    const double means[] = {0.};
    const double covariance[] = {1.};
    const skimagexx::DiagonalMixtureView model{weights, means, covariance, 1, 1};
    double log_density[3], log_responsibilities[3];
    if (!skimagexx::diagonal_gmm_expectation(samples, 3, model, log_density,
                                            log_responsibilities)) return 1;
    if (!(log_density[0] == log_density[1] && log_density[2] < log_density[1]) ||
        !std::all_of(log_responsibilities, log_responsibilities + 3,
                     [](double value) { return value == 0.; })) return 2;

    double weight, mean, variance, precision;
    if (!skimagexx::diagonal_gmm_maximization(samples, 3, 1, 1, log_responsibilities,
                                             .25, &weight, &mean, &variance, &precision))
        return 3;
    if (weight != 1. || std::abs(mean - 1.) > 1e-12 ||
        std::abs(variance - (8. / 3. + .25)) > 1e-12 ||
        std::abs(precision - 1. / std::sqrt(variance)) > 1e-12) return 4;

    double result[3];
    if (!skimagexx::fisher_vector(samples, 3, model, covariance, false, .5, result))
        return 5;
    if (result[0] != 0. || result[1] != 1. ||
        std::abs(result[2] + 8. / (3. * std::sqrt(2.))) > 1e-12) return 6;
    if (!skimagexx::fisher_vector(samples, 3, model, covariance, true, .5, result) ||
        std::abs(std::hypot(result[1], result[2]) - 1.) > 1e-12) return 7;
    return 0;
}
