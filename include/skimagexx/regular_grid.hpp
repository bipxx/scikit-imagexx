#pragma once
#include <cstddef>
#include <cstdint>
#include "components.hpp"
#include <stdexcept>
#include <vector>
namespace skimagexx::detail {
// Compatibility frontier retained from the source-first prototype. The public
// operation below handles typed shape views and returns owned slice metadata.
enum class RegularGridSliceKind { all, strided };
struct RegularGridSlice {
    RegularGridSliceKind kind=RegularGridSliceKind::all;
    std::int64_t start=0;
    std::int64_t step=1;
};
// This compatibility adapter's first typed frontier: positive int64 dimensions,
// exact product<=2^53; positive builtin integer points<=2^53. A source-faithful
// full-grid branch returns `all`; computed starts/steps construct NumPy intp
// scalar slice fields at the Python boundary. stop is ALWAYS None.
// Source floor(step/2) precedes ties-to-even round(step); sorting/axis inversion,
// root/power rounding and FP ownership retain explicit admission partitions.
// Metadata work is ordered and tiny: no artificial SIMD or process worker pool.
// Caller retains shape/output storage. Refusal changes no outputs or FP state.
// Existing regular_seeds binds normalized positive axes to RegularSeedsAxis;
// it will remain an unchanged provider consumer until root integrates later.
bool regular_grid_slices_i64(const std::int64_t* shape,std::size_t ndim,
                            std::uint64_t n_points,RegularGridSlice* output);
}

namespace skimagexx {
using RegularGridSlice=detail::RegularGridSlice;
using RegularGridSliceKind=detail::RegularGridSliceKind;
enum class RegularGridErrorKind { division_zero, complex_power, unsupported };
struct RegularGridError:std::runtime_error {
    RegularGridErrorKind kind;
    RegularGridError(RegularGridErrorKind k,const char* text):std::runtime_error(text),kind(k){}
};
struct RegularGridPrepared {
    double space_size=1;
    std::vector<double> sorted_dimensions;
    std::vector<std::uint64_t> sorted_words;
    std::vector<std::size_t> inverse_order;
    PixelType source_type=PixelType::int64;
    unsigned default_integer_bytes=8;
};
// A shape is an ordinary bounded1D ComponentsView, including negative strides
// and swapped integer storage. No image allocation or worker pool is used.
// default_integer_bytes is explicit source reduction metadata (NOT intp).
RegularGridPrepared regular_grid_prepare(const ComponentsView& shape,unsigned default_integer_bytes=8);
double regular_grid_suffix_product(const RegularGridPrepared&,std::size_t first);
// Finish is AFTER the caller's early comparison. Python must compare its exact
// integer point operand before converting it for floating division/root work.
std::vector<RegularGridSlice> regular_grid_finish(const RegularGridPrepared&,double n_points);
// Complete standalone operation; output all-slices have None-equivalent fields,
// strided outputs carry signed64 start/step and an always-None stop.
std::vector<RegularGridSlice> regular_grid(const ComponentsView& shape,double n_points,unsigned default_integer_bytes=8);
}
