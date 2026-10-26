#pragma once
#include <cstddef>
#include <cstdint>

namespace skimagexx {
inline constexpr const char* footprint_shapes_interface="footprint-shapes-v1";
// Private prepared stages, not a full public-API capability. All spans are
// caller-owned, C-contiguous and disjoint. Lengths count elements except bytes.
// Validation precedes every store. Finite geometry is bounded to <=2^20 axes;
// unsupported public intake remains in the source-faithful Python shell.
struct FootprintBytes {void* data;std::size_t bytes;};
struct FootprintConstBytes {const void* data;std::size_t bytes;};
void footprint_shapes_ellipse(FootprintBytes output,std::size_t rows,std::size_t columns,
                              double height,double width);
// Convex hull of nonzero pixel centers expanded by the four half-pixel edge
// offsets used by original convex_hull_image. Output is a byte0/1 mask. This is
// intentionally a finite2D stage, not a general geometry API.
void footprint_shapes_hull(FootprintConstBytes input,FootprintBytes output,
                           std::size_t rows,std::size_t columns);
// Actual original mgrid arrays, not reconstructed radius coordinates. Each
// source contains count native binary64 values; expression is (absX+absY)+absZ.
// Complex grids use NumPy-prepared magnitudes. Inclusive=false implements the
// strict real-part boundary when the finite complex radius has negative imag.
void footprint_shapes_l1(FootprintConstBytes x,FootprintConstBytes y,FootprintConstBytes z,
                         FootprintBytes output,std::size_t count,double radius,bool inclusive=true);
// Exact builtin nonnegative integer-radius generator. Axis validation also
// bounds the original dense3-grid allocation (24*axis^3 <= PTRDIFF_MAX), which
// guarantees exact binary64 mgrid coordinates and L1 sums. Output is byte0/1.
std::size_t footprint_shapes_octahedron_axis(std::size_t radius);
void footprint_shapes_octahedron(FootprintBytes output,std::size_t radius);
// Reference-free dtype atoms. Numeric/string zero and one are prepared by the
// original NumPy conversion stage. No object pointer/reference copying.
void footprint_shapes_atoms(FootprintConstBytes mask,FootprintBytes output,
                            FootprintConstBytes zero,FootprintConstBytes one,
                            std::size_t count,std::size_t itemsize);
}
