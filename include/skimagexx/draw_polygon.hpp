#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace skimagexx::detail {
constexpr std::size_t draw_polygon_max_vertices = 4096;
constexpr std::size_t draw_polygon_max_scan = 16*1024*1024;
constexpr std::size_t draw_polygon_max_edge_tests = 64*1024*1024;

// Source-normalized contiguous float64 columns, equal nonzero vertex count.
// Borrowed only for prepare; the successful plan retains no input pointers.
struct DrawPolygonVertices {
    const double* rows = nullptr;
    const double* columns = nullptr;
    std::size_t count = 0;
};
// Inclusive integer scan bounds AFTER source-ordered reductions/conversions
// and optional shape clipping. Those numeric reductions remain an explicit
// frontend provider gap, not owned by this first typed raster interface.
struct DrawPolygonScan {
    std::array<std::intptr_t,2> first{};
    std::array<std::intptr_t,2> last{};
};
struct DrawPolygonPlan {
    std::array<std::intptr_t,2> first{};
    std::array<std::size_t,2> shape{};
    // ONE classifier raster in canonical row-major scan order, labels0/1/2/3.
    // A selected pixel is label!=0. No coordinate-copy or second bool raster.
    std::vector<std::uint8_t> labels;
    // Exclusive row prefix, shape[0]+1 entries; back()==selected count.
    std::vector<std::size_t> row_offsets;
    std::size_t count = 0;
};
// Finite coordinates: zero or abs in [2^-128,2^20]; scan first in[0,2^20],
// last<=2^20 (negative last means empty), vertices/count/products within the
// limits above, nearest rounding/gradual underflow. Refusal leaves plan intact.
// Reuse literal existing polygon_label with xp=columns,yp=rows and query(c,r),
// retaining epsilon, two ray tests/divisions, vertex short-circuit and labels.
// Classify once; plan owns only labels+row prefix. Invalid/unassessed Python
// inputs/errors must be handled before this typed boundary, not fabricated.
bool draw_polygon_prepare(const DrawPolygonVertices&,const DrawPolygonScan&,
                          DrawPolygonPlan& output,int workers);
// Caller owns two disjoint aligned contiguous intp[count] outputs. Stable C
// logical order, row first then column; plan remains unchanged, empty no writes.
void draw_polygon_coords_into(const DrawPolygonPlan&,std::intptr_t* rows,
                              std::intptr_t* columns,int workers);
// Caller owns disjoint C uint8[rows*columns], initially arbitrary. Validate
// shape/product and that all selected coordinates fit, then zero/scatter
// canonical bool1 for label!=0. Does not allocate intermediate coordinates.
void draw_polygon_mask_into(const DrawPolygonPlan&,std::size_t rows,
                            std::size_t columns,std::uint8_t* output,int workers);
// Existing optional global-worker model/default1: frontend snapshots once and
// passes that budget through prepare/fill. Independent row spans can parallelize
// beyond the existing threshold; ordered prefix preserves emission order.
}
