#pragma once

#include "skimagexx/image.hpp"

namespace skimagexx {

enum class EpipolarScaling { rms, mean_radius, raw };

// All points have two coordinates. Matrices are row-major 3 x 3.
// Buffers must not overlap. Output has three values per source point.
bool epipolar_lines(const double* points, std::size_t count,
                    const double* matrix, double* output);
bool epipolar_residuals(const double* src, const double* dst, std::size_t count,
                        const double* matrix, double* output);

// Writes count x 9 constraints and the two conditioning matrices. A false
// result indicates nonfinite arithmetic or a degenerate conditioning scale.
bool epipolar_constraints(const double* src, const double* dst, std::size_t count,
                          EpipolarScaling scaling, double* constraints,
                          double* src_matrix, double* dst_matrix);

// Project the three singular values onto the selected rank constraint before
// reversing coordinate conditioning. u and vt are the row-major SVD factors.
void epipolar_matrix(const double* u, const double* singular_values, const double* vt,
                     const double* src_matrix, const double* dst_matrix,
                     bool essential, double* matrix);

}  // namespace skimagexx
