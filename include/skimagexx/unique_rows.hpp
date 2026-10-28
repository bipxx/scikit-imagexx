#pragma once
#include <cstddef>
#include <cstdint>

namespace skimagexx::detail {
struct UniqueRowsInput {
    const unsigned char* rows;
    std::size_t row_count;
    std::size_t row_bytes;
};
// Boundary invariants: positive row_bytes; stable C-contiguous readable rows,
// exact/addressable row_count*row_bytes; row_count fits int64. Output holds
// row_count native int64 indices, is disjoint from input, and workers >= 1.
// Return the number of unique indices in the output prefix. Compare unsigned
// raw row bytes, order equal rows by original index, and retain the first.
// No dtype arithmetic, hashing, floating operations, or Python callbacks.
std::size_t unique_rows_indices(const UniqueRowsInput& input,
                                std::int64_t* output_indices,int workers);
}

