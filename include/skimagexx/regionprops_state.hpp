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

// coords_scaled source phases; declarations only in this design packet.
// Shared first frontier: ndim2/3, product(shape)<=2^24, each extent<=2^20,
// |current origin|<=2^20. Caller holds a stable, addressable bool mask and
// count=its CURRENT nonzero population. No constructor population or origin.
// All coordinate buffers are column-major Nxndim: element(d*N+p). The binding
// supplies fresh owning F arrays; for N=0 it supplies the observed zero-stride
// ndarray metadata and may pass nullptr here. No entry retains any pointer.
// Capacity/byte-span/overlap checks belong to the external binding/SDK caller,
// once before using each new externally read operand. Internal stages consume
// established invariants; private intermediates must not be externally mutated.
inline constexpr std::size_t regionprops_coords_scaled_max_elements=1u<<24;
inline constexpr std::size_t regionprops_coords_scaled_max_extent=1u<<20;
inline constexpr std::intptr_t regionprops_coords_scaled_max_origin=1<<20;
// For the two f64 stages, new spacing/offset doubles are signed zero or finite,
// with nonzero magnitude
// in [2^-256,2^256]. With the integer bounds above, multiply/add stay normal or
// exact zero (minimum nonzero quantum >=2^-308), and magnitudes remain <2^278.
// Reuse the existing nearest/gradual FP guard including independent MXCSR round
// bits. Numeric refusal/unsupported FP state returns false BEFORE any writes.
// Accepted f64 work has no error-policy event; no NumPy geterr/callback delivery.
// Same positive worker snapshot is passed through all stages; no global reread.
// Small arrays stay serial; dense independent math may use the existing
// partition/SIMD machinery, strict noncontracted multiply/add, and exact scalar
// dispatch. A parallel implementation must install/restore the caller FP state
// per worker; never assume an old OpenMP worker retained the caller's rounding.

// 1. Materialize local argwhere coordinates BEFORE reading current slice starts.
// Reuse the existing C-logical mask walk; mask strides are signed BYTES. N must
// be established from this same stable mask before allocation/fill.
bool regionprops_coords_scaled_local_into(const std::uint8_t* mask,
    std::size_t ndim,const std::array<std::size_t,3>& shape,
    const std::array<std::ptrdiff_t,3>& mask_strides,std::size_t count,
    std::intptr_t* local_coordinates,int workers);
// 2. Add CURRENT source-normalized intp origins BEFORE reading _spacing.
// In-place is allowed only for the private owning index buffer. False leaves
// every coordinate unchanged so the literal source addition can run once.
bool regionprops_coords_scaled_origin_into(std::intptr_t* coordinates,
    std::size_t count,std::size_t ndim,const std::array<std::size_t,3>& shape,
    const std::array<std::intptr_t,3>& origins,int workers);
// 3. Complete source f64 multiplication BEFORE reading _offset. Input absolute
// coordinates obey stage2 bounds. Output is disjoint, caller-owned F double.
// Integer-spacing results are NOT this f64 interface; retain literal suffix.
bool regionprops_coords_scaled_multiply_f64_into(const std::intptr_t* coordinates,
    std::size_t count,std::size_t ndim,const std::array<double,3>& spacing,
    double* output,int workers);
// 4. Add the subsequently captured offset in-place to the private F double
// result. False leaves it unchanged for the literal source addition once.
bool regionprops_coords_scaled_add_f64_into(double* values,std::size_t count,
    std::size_t ndim,const std::array<double,3>& offset,int workers);
}
