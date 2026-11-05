#pragma once
#include <skimagexx/draw_coordinates.hpp>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace skimagexx::detail {
struct BezierCurveRequest {
    std::intptr_t r0=0,c0=0,r1=0,c1=0,r2=0,c2=0;
    double weight=1;
    bool clipped=false;
    double height=0,width=0;
};
// Private prepared storage preserves source traversal, duplicates, segment
// concatenation and stable shape filtering. Both public arrays are owning 1D
// intp even for a point or empty result. There is no scalar/0D special case.
struct BezierCurve {
    BezierCurveRequest request;
    std::vector<std::intptr_t> rows,columns;
};
// Boundary admission: 64-bit intp, nearest x87/SSE rounding, FTZ/DAZ disabled,
// coordinates within +/-2^20, finite weight in [0,16], finite shape bounds
// within +/-2^52. Literal source extrema/sqrt/division/floor(+.5)/int32 casts
// are evaluated in order. Each numeric cast, progress and output bound is
// checked before publishing; <=1M points and <=2^21 intermediate coordinates.
// Nonunit rational weights, both extrema and the source's one-half recursive
// return are part of the implementation, not line/unit-weight substitutes.
// Unsupported or exceptional arithmetic returns false leaving output intact;
// allocation failure propagates. Conversion/protocol/error providers belong
// to the frontend; this boundary does not call NumPy or the original algorithm.
bool prepare_bezier_curve(const BezierCurveRequest&,BezierCurve&);
inline std::size_t bezier_curve_count(const BezierCurve& p) noexcept {
    return p.rows.size();
}
// Fresh, disjoint aligned output storage has exact count capacity; signed
// element steps and lifetimes are validated once by caller. Empty may be null.
// Caller snapshots global worker budget once (default1). Ordered recurrence
// is serial; independent output copies may SIMD/partition at262144 words.
void bezier_curve_fill(const BezierCurve&,DrawCoordinatePair,int workers) noexcept;
}
