#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace skimagexx {

// Row-major, nonoverlapping input and output. All dimensions are positive.
// The gradient accepts rank two or three. The contour solvers require at least
// two samples per axis and a binary int8 initial level set. Float parameters
// follow the corresponding float32 or float64 NumPy arithmetic.
template<class T>
void inverse_gaussian_gradient_into(const T* image, T* output,
    const std::vector<std::size_t>& shape, double alpha, double sigma);

// The caller owns the curvature phase shared across successive contour calls.
// false starts with SI(IS(u)). Every smoothing operation reverses the phase.
template<class T>
void morphological_chan_vese_into(const T* image, const std::int8_t* initial,
    std::int8_t* output, const std::vector<std::size_t>& shape, std::size_t iterations,
    std::size_t smoothing, double lambda1, double lambda2, bool reverse_curvature);

// The balloon threshold has already been divided by abs(balloon) when active.
// The balloon direction is one of -1, 0, 1.
template<class T>
void morphological_geodesic_active_contour_into(const T* image, const std::int8_t* initial,
    std::int8_t* output, const std::vector<std::size_t>& shape, std::size_t iterations,
    std::size_t smoothing, double balloon_threshold, int balloon, bool reverse_curvature);

} // namespace skimagexx
