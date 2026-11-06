#pragma once
#include "components.hpp"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace skimagexx::detail {
enum class FindBoundaryMode { thick, inner, outer, subpixel };
struct FindBoundaryParameters {
    FindBoundaryMode mode=FindBoundaryMode::thick;
    unsigned connectivity=1; // Already clamped to [1,rank], as the source footprint.
    std::uint64_t background_bits=0;
    // False means no value of the input dtype equals the public background.
    // Signed labels/background use sign-extended int64 bit words. Bool input
    // means source bool->uint8 conversion, hence logical0/1 and sentinel255.
    bool background_representable=true;
};
constexpr std::size_t find_boundaries_max_elements=16*1024*1024;
constexpr std::size_t find_boundaries_max_neighbor_visits=64*1024*1024;
// First typed frontier: native-endian bool or eight integer dtypes, rank1..8,
// nonempty extents, addressable signed byte strides. Ordinary int64/uint64
// require EVERY label in [-2^53,2^53] (unsigned lower bound0); their outer
// mode is refused because generated dtype-max backgrounds can cast outside
// range in upstream double-staged min/max intermediate stores. Subpixel
// accepts full integer widths: it never enters SciPy morphology. Outside
// this exact-promotion frontier, source rounds labels through double and
// casts extrema back at each pass; do not claim exact-integer equivalence.
// Nonexact-but-in-range wide-label behavior remains a future typed partition.
// No half/float or swapped inputs. This shape helper reads no pixels/worker state, and changes shape
// only on success. Subpixel extent is exactly2*s-1; other modes preserve shape.
bool find_boundaries_shape(const ComponentsView&,FindBoundaryMode,
                           std::vector<std::size_t>& output_shape);
// Caller retains the input owner and stable descriptor/pixels for the call;
// provides disjoint, nonoverlapping uint8 storage covering the derived shape
// at output_strides (BYTES). SDK raw pointers imply caller-proved capacity;
// validation cannot discover an unknown allocation extent. Refusal is before
// writes and never invokes a provider. Output is canonicalbool0/1.
// Thick is source neighborhood max!=min. Inner masks it by background.
// Outer also uses max(full-neighbors)!=min(background-replaced-by-dtype-max),
// including nonzero/high backgrounds; do not substitute object adjacency.
// Subpixel preserves the source dtype-max sentinel collision: at interstitial
// sites it requires two distinct corner labels excluding that sentinel.
// Independent output spans may partition with the existing worker dispatcher.
// workers0 snapshots the existing process budget ONCE; defaultglobal1. A
// positive argument is an already captured1..1024 budget, not a new pool.
// Within the exact-promotion frontier, integer decisions make no FP change. SIMD claims
// require actual compiler evidence; no ordered reduction or float conversion.
bool find_boundaries_into(const ComponentsView&,const FindBoundaryParameters&,
    std::uint8_t* output,const std::vector<std::ptrdiff_t>& output_strides,
    int workers=0);
}
