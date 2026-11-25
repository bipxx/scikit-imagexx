#pragma once

#include "image.hpp"

namespace skimagexx {

struct DiagonalMixtureView {
    const double* weights;
    const double* means;
    const double* precision_cholesky;
    std::size_t components;
    std::size_t dimensions;
};

// All matrices are contiguous and row-major. Outputs must not overlap inputs.
// The descriptor buffer has rows * dimensions values. Model matrices have
// components * dimensions values. False indicates unsuitable numeric values.
bool diagonal_gmm_expectation(const double* descriptors, std::size_t rows,
                              DiagonalMixtureView model, double* log_density,
                              double* log_responsibilities);
bool diagonal_gmm_maximization(const double* descriptors, std::size_t rows,
                               std::size_t dimensions, std::size_t components,
                               const double* log_responsibilities, double regularization,
                               double* weights, double* means, double* covariances,
                               double* precision_cholesky);

// The output has components * (1 + 2 * dimensions) values.
bool fisher_vector(const double* descriptors, std::size_t rows, DiagonalMixtureView model,
                   const double* covariances, bool improved, double alpha, double* output);

} // namespace skimagexx
