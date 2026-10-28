#pragma once
#include <cstddef>

namespace skimagexx::detail {
struct SubdividePolygonInput {
    const double* data;
    std::size_t count; // AFTER literal circular-last-row stripping
    std::ptrdiff_t row_stride,column_stride; // signed strides IN DOUBLES
};
// Prepared input is stable native float64 with logical shape(count,2). The
// caller validates signed spans, products, alignment and fresh/disjoint writable
// output capacity once, and snapshots workers>=1. No arbitrary point-count cap.
// Literal degree validation and closure comparison precede preparation. Only
// pure bool preserve_ends enters here; user truth callbacks stay in the literal
// frontend so they are not evaluated before convolution. No SciPy callback.
// L for degrees1..7 is 2,2,3,3,4,4,5. M=count for circular(same+wrap), otherwise
// count-L+1(valid). Output shape is (2*M+(circular?1:preserve_ends?2:0),2),
// contiguous float64. Circular count excludes the original duplicate last row;
// open count is the original input count. Output must not overlap input.
// Owns both weighted convolutions, increasing-tap +0 separate multiply/add,
// odd/even interleave, computed-first-row closure and ordinary endpoint copies.
// Native checks finite input and its numerical/FP partition once under scoped
// capture. Invalid degree, count0, short valid input, non-nearest rounding,
// capture failure or INVALID/DIVBYZERO/OVERFLOW/UNDERFLOW refuse with false;
// discard private output on false. Inexact alone is permitted. Caller and worker
// FP flags/rounding are restored. No Python, allocation, pool or global-worker
// mutation. Unsupported types/protocols/nonfinite routes remain literal providers.
bool subdivide_polygon_into(const SubdividePolygonInput&,unsigned degree,
    bool circular,bool preserve_ends,double* output,int workers);
}
