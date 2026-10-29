#pragma once
#include <cstddef>
#include <cstdint>

namespace skimagexx::detail {
// All pointers, counts/products, signed coordinate additions and element-step
// spans are validated once by the caller. Outputs are fresh/disjoint, aligned
// intptr_t storage; callers retain every owner and snapshot the global worker
// policy into workers. Zero-count buffers may be null. No Python/trig calls.
struct DrawCoordinatePair {
    std::intptr_t* rows;
    std::intptr_t* columns;
    std::ptrdiff_t row_step;
    std::ptrdiff_t column_step;
};
// Exact output count max(abs(r1-r0),abs(c1-c0))+1. Endpoint differences and
// twice either difference fit intptr_t. The literal Bresenham order, >=0 tie,
// strict steep swap and final original endpoint store are preserved.
void draw_line_into(std::intptr_t r0,std::intptr_t c0,std::intptr_t r1,
                    std::intptr_t c1,DrawCoordinatePair output,int workers);
// lengths are nonnegative prepared axis lengths; count is their exact product
// (zero for ndim==0). Each outputs[d] is a C-contiguous grid with the first two
// shape axes swapped, exactly np.meshgrid's default xy order. ndim is dynamic.
// Every starts[d]+coordinate fits intptr_t. Empty axes write no coordinates.
void draw_rectangle_into(const std::intptr_t* starts,const std::size_t* lengths,
                         std::size_t ndim,std::size_t count,
                         std::intptr_t* const* outputs,int workers);
struct DrawEllipseGrid {
    std::size_t rows,columns;
    double center_row,center_column,row_radius,column_radius,sin_alpha,cos_alpha;
    std::intptr_t row_offset,column_offset;
};
// rows/columns are source-prepared clipped bounding dimensions, their product
// is addressable, integer grid coordinates are exactly representable in double,
// and offsets+each emitted index fit intptr_t. Finite shifted centers/trig and
// strictly-positive finite radii are the numerical admission partition.
// mask[rows*columns] and row_prefix[rows+1] are fresh/disjoint caller storage.
// Source association: a=(r*cos+c*sin)/row_radius,
// b=(r*sin-c*cos)/column_radius; inside=(a*a+b*b)<1, no FMA/reassociation.
// On true row_prefix[0]==0, its final entry is exact point count, and mask is
// complete. FP state is restored on every exit. False means discard all private
// mask/prefix contents and use the fresh literal provider tail; no partial
// result may be published. Non-nearest rounding/capture or exceptional FP
// arithmetic refuses; inexact alone is permitted. No trig/libm replacement.
bool draw_ellipse_mask_into(const DrawEllipseGrid&,std::uint8_t* mask,
                            std::size_t* row_prefix,int workers);
// After a successful unchanged mask/count call, output has capacity equal to
// row_prefix[rows]. Emit source row-major hits with original bounding offsets.
// Output may be interleaved Nx2 (element steps2) or separate strided columns.
void draw_ellipse_emit_into(const DrawEllipseGrid&,const std::uint8_t* mask,
                            const std::size_t* row_prefix,
                            DrawCoordinatePair output,int workers);
}
