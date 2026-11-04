#pragma once
#include <skimagexx/draw_coordinates.hpp>
#include <cstddef>
#include <cstdint>

namespace skimagexx::detail {
// Prepared once at the typed boundary. Preserve this geometry between count
// and fill. The reference uses signed32-bit coordinates and integer squares.
struct DrawLineAAGeometry {
    std::int32_t r0, c0, r1, c1;
    std::int32_t dr, dc, sign_r, sign_c;
    double distance;
};
// Refuse before narrowing/arithmetic if endpoints lack signed32-bit neighbor
// headroom or dr*dr+dc*dc exceeds INT32_MAX. Refusal leaves output unchanged.
// Successful geometry computes the literal sqrt once (point distance is1).
bool draw_line_aa_prepare(std::intptr_t r0, std::intptr_t c0,
                          std::intptr_t r1, std::intptr_t c1,
                          DrawLineAAGeometry& output);
// Exact source emission count, including ordered side pixels. Dependent
// traversal is serial and respects the caller's already-validated worker budget.
std::size_t draw_line_aa_count(const DrawLineAAGeometry&, int workers);
// Caller validates fresh/disjoint aligned coordinate/weight buffers, nonzero
// element strides, complete signed spans and capacity equal to count. No
// repeated intake checks or allocations. Output order and rounded64-bit
// division then1.-distance follow the original0.26 Cython/NumPy stages.
void draw_line_aa_into(const DrawLineAAGeometry&, DrawCoordinatePair,
                       double* weights, std::ptrdiff_t weight_step, int workers);
}
