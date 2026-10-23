#pragma once
#include <cstddef>
namespace skimagexx {
// projective-coordinate-v1: finite real double coordinates and homogeneous
// matrix, byte-strided input views, disjoint C-contiguous owned output supplied
// by caller. Matrix is (dimensions+1)^2, output is count*dimensions doubles.
// Zero divisors use double epsilon. Uses shared process-global workers.
void projective_apply_into(const double* points,std::ptrdiff_t point_row,
    std::ptrdiff_t point_column,const double* matrix,std::ptrdiff_t matrix_row,
    std::ptrdiff_t matrix_column,double* output,std::size_t count,std::size_t dimensions);
// Compatible checked companion: true means exceptional arithmetic requires
// the runtime NumPy provider. Floating-point environment is preserved per worker.
bool projective_apply_checked_into(const double* points,std::ptrdiff_t point_row,
    std::ptrdiff_t point_column,const double* matrix,std::ptrdiff_t matrix_row,
    std::ptrdiff_t matrix_column,double* output,std::size_t count,std::size_t dimensions);
namespace simd {
std::size_t projective_apply(const double* points,std::ptrdiff_t point_row,
    std::ptrdiff_t point_column,const double* matrix,std::ptrdiff_t matrix_row,
    std::ptrdiff_t matrix_column,double* output,std::size_t count,std::size_t dimensions);
}
}
