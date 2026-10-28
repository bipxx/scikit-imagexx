#pragma once
#include <cstddef>
#include <cstdint>

namespace skimagexx::detail {
struct ApproximatePolygonInput {
    const double* data;
    std::size_t count;
    std::ptrdiff_t row_stride,column_stride; // signed strides IN DOUBLES
};
struct ApproximatePolygonTrig { double sin_angle,cos_angle; };
using ApproximatePolygonTrigProvider=ApproximatePolygonTrig(*)(void* context,double dr,double dc);
// Boundary invariants: count>=1; stable float64 coordinates of shape
// (count,2); both signed spans and products addressable. keep[count] uint8 and
// distances[count] double are writable, fresh/disjoint from all input/storage.
// The caller validates these invariants once and snapshots workers>=1. No
// arbitrary point-count cap is imposed. Native numerical admission checks finite
// coordinates and finite strictly-positive tolerance once under its FP capture;
// the caller does not duplicate that scan. Signaling NaNs are classified by bits.
// the public frontend retains the literal tolerance<=0 input-identity route.
// The primitive owns mask initialization, segment distances, deterministic
// first-maximum split and a right-then-left push stack (left processed first).
// Optional trig runs ONLY on the caller thread and supplies pure float64
// sin(-atan2(dr,dc)), cos(-atan2(dr,dc)), in that source order. Its context must
// outlive this call; it must not mutate coordinates/output or worker settings.
// Exceptions propagate after scoped FP restoration. Python can supply its exact
// NumPy scalar elementary primitives; nullptr uses standalone platform libm,
// which is an explicit compatibility boundary, not a NumPy-equivalence claim.
// Requires FE_TONEAREST and capturable FP state. Returns false for other modes,
// failed capture, nonfinite arithmetic/trig or FE_INVALID/DIVBYZERO/OVERFLOW/
// UNDERFLOW. FP flags/rounding are restored. On false discard all keep/scratch
// contents; no partial-output credit. Inexact alone is permitted. Native work
// uses no Python, caller callbacks except trig, pool, or global worker mutation.
// Final Python coords[keep,:] gather remains an explicit provider boundary.
bool approximate_polygon_select(const ApproximatePolygonInput&,double tolerance,
    std::uint8_t* keep,double* distances,int workers,
    ApproximatePolygonTrigProvider trig=nullptr,void* context=nullptr);
}
