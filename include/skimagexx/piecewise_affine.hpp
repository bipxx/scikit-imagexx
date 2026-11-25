#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace skimagexx {

inline constexpr std::size_t piecewise_affine_max_controls = 256;

using PiecewiseSimplex = std::array<std::int32_t, 3>;
using PiecewiseTriangle = std::array<double, 6>;
using PiecewiseMatrix = std::array<double, 9>;

struct PiecewiseAffineMesh {
    std::vector<PiecewiseSimplex> simplices;
    std::vector<PiecewiseTriangle> triangles;
    std::vector<PiecewiseMatrix> matrices;
};

// Fits finite distinct planar controls with an unambiguous Delaunay mesh.
// Rejected fits leave mesh intact; inverse fitting swaps src and dst.
bool piecewise_affine_fit(const double* src, const double* dst, std::size_t count,
                          PiecewiseAffineMesh& mesh);

// Fits matrices for explicitly ordered simplices, preserving their order.
// Each input contains count interleaved x/y pairs.
bool piecewise_affine_fit_mesh(const double* src, const double* dst, std::size_t count,
                               const PiecewiseSimplex* simplices, std::size_t simplex_count,
                               PiecewiseAffineMesh& mesh);

// Triangle entries are the row-major 2x2 barycentric transform followed
// by the origin. Matrices are row-major homogeneous transforms. Rejected
// applications must discard output; zero query rows may use null buffers.
bool piecewise_affine_apply(const PiecewiseTriangle* triangles, const PiecewiseMatrix* matrices,
                            std::size_t simplex_count, const double* coords,
                            std::size_t rows, double* output);

} // namespace skimagexx
