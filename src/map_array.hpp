#pragma once
#include "skimagexx/map_array.hpp"

namespace skimagexx::simd {
// LUT has four padded cells, with uint32 slots for itemsize1/2/4, uint64 for8,
// and two uint64 lanes/cell for16. Indices are width-masked before unsigned
// range testing. Missing labels produce all-zero output words.
std::size_t map_array_dense(const std::uint64_t* indices, std::size_t count,
    std::uint64_t mask, std::uint64_t minimum, std::size_t cells,
    const void* table, std::size_t itemsize, void* output, std::ptrdiff_t output_stride);
// At most eight snapshotted mapping entries; comparisons and selection are
// integer-only, with later duplicate keys overwriting earlier matches.
std::size_t map_array_small(const detail::MapArrayInput&, const void* values,
    std::size_t begin, std::size_t count, detail::MapArrayOutput);
}
