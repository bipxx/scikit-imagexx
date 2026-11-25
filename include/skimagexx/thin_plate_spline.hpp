#pragma once

#include <cstddef>
#include <vector>

namespace skimagexx {

inline constexpr std::size_t thin_plate_spline_max_controls = 128;

// Inputs contain count interleaved x/y pairs. The successful fit produces
// count + 3 interleaved weight pairs (radial weights, constant, x, y).
// A rejected fit leaves weights intact.
bool thin_plate_spline_fit(const double* src, const double* dst, std::size_t count,
                           std::vector<double>& weights);

// Buffers are contiguous and nonoverlapping. A rejected application must
// discard output. Zero query points may use null coordinate/output buffers.
bool thin_plate_spline_apply(const double* src, const double* weights, std::size_t count,
                             const double* coords, std::size_t rows, double* output);

} // namespace skimagexx
