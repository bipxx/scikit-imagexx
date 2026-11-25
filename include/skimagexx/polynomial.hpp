#pragma once

#include "skimagexx/image.hpp"

namespace skimagexx {

// Coefficients are ordered by increasing total degree, starting with x^degree.
// Matrix and coordinates are contiguous, row-major, and must not overlap.
inline constexpr unsigned polynomial_max_order = 5;

bool polynomial_apply(const double* coords, std::size_t count, const double* params,
                      std::size_t coefficients, double* output);

// Construct the weighted homogeneous total-least-squares system. Passed weights
// are relative confidences, or nullptr for a unit weight on every observation.
bool polynomial_constraints(const double* src, const double* dst,
                            const double* weights, std::size_t count,
                            unsigned order, double* equations);

}  // namespace skimagexx
