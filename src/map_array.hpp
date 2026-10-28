#pragma once
#include <cstddef>
#include <cstdint>

namespace skimagexx::detail {
struct MapArrayInput {
    const std::uint64_t* indices;
    std::size_t count;
    const std::uint64_t* keys;
    const void* values;
    std::size_t key_count;
    std::size_t itemsize;
    unsigned index_bits;
    std::ptrdiff_t values_stride;
};
struct MapArrayOutput {
    void* data;
    std::ptrdiff_t stride;
};
// Boundary preconditions: native uint64 indices/keys are immutable snapshots;
// bits is 8/16/32/64, itemsize is 1/2/4/8/16, and all count/stride byte spans
// are addressable. Values may alias output: this function snapshots them once
// BEFORE any output write. Output elements do not overlap one another.
// The binder snapshots the existing global worker setting (1..1024). No
// temporary global thread-state changes or Python calls occur in this kernel.
void map_array_into(const MapArrayInput&, MapArrayOutput, int workers);
}

namespace skimagexx::simd {
// LUT has four padded cells, with uint32 slots for itemsize1/2/4, uint64 for8,
// and two uint64 lanes/cell for16. Indices are width-masked before unsigned
// range testing. Missing labels produce all-zero output words.
std::size_t map_array_dense(const std::uint64_t* indices, std::size_t count,
    std::uint64_t mask, std::uint64_t minimum, std::size_t cells,
    const void* table, std::size_t itemsize, void* output);
}
