#pragma once
#include <skimagexx/draw_coordinates.hpp>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace skimagexx::detail {
struct EllipsePerimeterRequest {
    std::intptr_t row=0,column=0,row_radius=0,column_radius=0;
    double orientation=0;
    bool clipped=false;
    double height=0,width=0;
};
// Owns source-ordered coordinates, including duplicates. Do not mutate between
// prepare and fill. No Python values, borrowed operands or cached worker limit.
struct EllipsePerimeter {
    EllipsePerimeterRequest request;
    std::vector<std::intptr_t> rows,columns;
    bool scalar_result=false; // Both radii0: two owning0D public arrays, skip clip.
};
// Initial64bit-intp domain: orientation0, radii0..32768, checked integer origin
// translations and <=1Mpoints. Exact integer recurrence is computed ONCE here;
// intermediate/progress bounds are checked at this boundary. Nonzero finite
// |orientation|<=.5 supports ONLY origin0/radii1, via literal source half-weight
// Bezier segments, not a general rotated-ellipse primitive. Other rotations are
// an explicit remaining dependency. Shape bounds are finite in[-2^52,2^52];
// zero-radius point ignores clipping. No floating-to-integer shape conversion.
// Refusal leaves output intact; allocation failure propagates normally.
bool prepare_ellipse_perimeter(const EllipsePerimeterRequest&,EllipsePerimeter&);
inline std::size_t ellipse_perimeter_count(const EllipsePerimeter& p) noexcept {
    return p.rows.size();
}
// Caller provides fresh/disjoint aligned intp storage of exact count capacity;
// signed/gapped element steps and spans/owners validated once by caller. Empty
// outputs allow null. Caller snapshots existing global budget once(default1).
// Independent copies may SIMD/partition for >=262144 coordinatewords; ordered
// recurrence remains serial. General Bezier/domain completeness is NOT claimed.
void ellipse_perimeter_fill(const EllipsePerimeter&,DrawCoordinatePair,int workers) noexcept;
}
