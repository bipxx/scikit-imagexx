#pragma once

#include <cstddef>

namespace skimagexx {

// Input holds two disjoint logical rows of dimensions float64 coordinates.
// Admits 2D/3D bounded, finite integer-valued coordinates in nearest rounding.
// All buffers must be valid and disjoint. On false, output is unchanged.
bool line_model_nd_two_point(const double* input, std::size_t dimensions,
                            double* origin, double* direction) noexcept;

// Input is count coordinates. Origin and direction each have two elements;
// output has count * 2 elements in row order. Buffers are disjoint and valid.
// Returns false without writing on unsupported axis, bounds, or rounding.
bool line_model_nd_predict_2d(const double* input, std::size_t count,
                             const double* origin, const double* direction,
                             std::size_t axis, double* output) noexcept;

// Three-coordinate counterpart. Origin and direction each have three
// elements; output has count * 3 elements. The same validation applies.
bool line_model_nd_predict_3d(const double* input, std::size_t count,
                             const double* origin, const double* direction,
                             std::size_t axis, double* output) noexcept;

} // namespace skimagexx
