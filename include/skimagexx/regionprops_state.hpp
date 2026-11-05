#pragma once
#include "components.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace skimagexx::detail {
struct RegionPropsBounds {
    std::uint64_t label=0;
    std::array<std::size_t,3> start{},stop{};
    std::size_t initial_population=0;
};
// Metadata owns only shape/stride vectors and frozen constructor geometry.
// labels.data is BORROWED: caller retains its original owner/addressable storage
// and keeps dtype/shape/strides stable. Pixel mutations are deliberately visible
// to subsequent fresh-mask calls. No copy of label pixels or public class/cache.
struct RegionPropsState {
    ComponentsView labels;
    std::vector<RegionPropsBounds> regions;
    int workers=1; // Constructor snapshot metadata; masks read the current budget.
};
// One typed entry validation: nonempty2D/3D native-endian eight integer dtypes,
// no half/bool/float; representable signed byte spans/products/intp coordinates.
// Caller supplies genuinely addressable storage for the view. Refusal returns
// false without changing state: it does not invoke a numerical provider.
// Snapshots process-wide workers once. Sorted positive IDs, C-order geometry;
// background/nonpositive labels ignored, disconnected equal IDs grouped.
bool prepare_regionprops_state(const ComponentsView&,RegionPropsState&);
// Frozen state and region index are internal invariants. Output is fresh,
// disjoint contiguous uint8[product(stop-start)], canonical0/1. Returns CURRENT
// population, which may differ from immutable initial_population after mutation.
// Independent mask rows snapshot the current global budget per call; ordered
// discovery stays serial. Later global-budget changes affect existing states.
std::size_t regionprops_mask_into(const RegionPropsState&,std::size_t region,
                                 std::uint8_t* output);
// Same mask/count contract, directly into caller-validated signed byte strides
// for the frozen bounding-box shape. No intermediate C-layout mask. Caller
// proves complete disjoint output spans; public binding supplies owned C/F.
std::size_t regionprops_mask_strided_into(const RegionPropsState&,std::size_t region,
    std::uint8_t* output,const std::array<std::ptrdiff_t,3>& output_strides);
// Count current caller-owned mask before allocating coords, including cached
// masks modified through the eventual public image array. Nonzero means true.
std::size_t regionprops_mask_population(const std::uint8_t*,std::size_t count) noexcept;
// Caller-owned mask is stable from population count through fill; output has
// population*ndim aligned disjoint intp elements. Emits absolute unshifted
// coordinates as C-order Nxndim. Never uses initial_population for capacity.
// NumPy offset promotion, spacing/area types, cache/aliases/errors and owning
// allocation are explicit future frontend gates; no FP arithmetic here.
void regionprops_coords_into(const RegionPropsState&,std::size_t region,
                             const std::uint8_t* mask,std::intptr_t* output) noexcept;
// Read the CURRENT provided mask using signed byte strides and emit directly
// into caller-validated coordinate byte strides. Canonical C logical order is
// preserved even for F masks; point_step/dimension_step are BYTES. Capacity and
// mask stability remain count/fill boundary invariants. No temporary/copy.
void regionprops_coords_strided_into(const RegionPropsState&,std::size_t region,
    const std::uint8_t* mask,const std::array<std::ptrdiff_t,3>& mask_strides,
    std::intptr_t* output,std::ptrdiff_t point_step,std::ptrdiff_t dimension_step) noexcept;
// Owned scalar arithmetic; NumPy scalar boxing, cache/dependency evaluation,
// spacing normalization and FP-policy delivery remain frontend responsibilities.
enum RegionPropsScalarFp : unsigned {
    regionprops_fp_invalid=1, regionprops_fp_divide=2,
    regionprops_fp_overflow=4, regionprops_fp_underflow=8
};
struct RegionPropsScalar64 { std::uint64_t bits; unsigned fp_status; };
// Exact count <= min(2^53, INTP_MAX); pixel_area is positive zero or finite
// [2^-256,2^256]. Reuse mask_population for area and CURRENT image.size for
// area_bbox. No constructor geometry/initial population or cached num_pixels.
RegionPropsScalar64 regionprops_scaled_count_f64(std::size_t count,double pixel_area);
// Visible area and visible area_bbox, evaluated separately in source order.
// Operands are positive zero or finite [2^-256,2^309]. Required 0/0 returns
// bits=0xfff8000000000000 and invalid status. Positive/0 reports divide status.
RegionPropsScalar64 regionprops_extent_f64(double area,double area_bbox);
// Both typed entries reject values outside this first frontier and require
// nearest rounding/gradual underflow (invalid_argument). They allocate nothing,
// run serially, read no worker state, and restore caller fenv including flags.
// Status bits above are SDK-local: NumPy callback invalid uses flag8, not1.
}
