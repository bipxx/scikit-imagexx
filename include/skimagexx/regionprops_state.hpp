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
}
