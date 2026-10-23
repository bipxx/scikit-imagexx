#pragma once
#include <cstddef>

namespace skimagexx {
enum class PairPrecision { native, wide_scalar, wide_scalar_narrow_square };
// Upper-triangular fields in row-major order, each holding count native,
// contiguous float32/64 values. Output has dimension*count values, with the
// eigenvalue axis first, in decreasing order, unless interleaved is true (then
// the eigenvalue axis is last and increasing). Matrices have dimension1..64.
// pair_formula uses the ordered two-by-two array formula. Other dimensions use
// a scaled double Jacobi iteration. The precision options preserve scalar
// promotion at the Python boundary without depending on Python in this library.
// False means nonfinite/unsafe data or nonconvergence; output must then be
// discarded. Empty fields are allowed when count==0. Global workers default1.
bool symmetric_eigenvalues_into(const void* const* fields,void* output,
    std::size_t count,std::size_t dimension,bool single,bool pair_formula=true,
    PairPrecision precision=PairPrecision::native,bool interleaved=false);
}
