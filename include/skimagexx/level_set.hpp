#pragma once
#include <cstddef>
#include <cstdint>

namespace skimagexx::detail {
struct LevelSetShape {
    const std::size_t* dimensions;
    std::size_t ndim;
    std::size_t count;
};
// The boundary establishes an exact count/product, addressable disjoint int8
// output, workers >= 1, and stable dimensions for the duration of the call.
// ndim == 0 represents one scalar element; any empty axis represents count 0.
// square_size is a nonzero signed int64; negative values use floor division.
void checkerboard_level_set_into(LevelSetShape shape, std::int64_t square_size,
                                 std::int8_t* output, int workers);
// Shared boundary predicate: finite binary64 radius and integer-center squared
// distance sums <= 2^53, so int64 source sums and double sums are both exact.
// Shape/count validation belongs to the binder, before calling this predicate.
bool disk_level_set_geometry_admissible(LevelSetShape shape,
                                        const std::int64_t* center, double radius);
// Requires geometry_admissible and the storage/worker invariants above.
// Preserves incoming floating-point state. false means discard the ENTIRE
// private output and obtain a fresh result from the literal NumPy provider.
// No partial output may escape when rounding/state capture refuses admission.
bool disk_level_set_into(LevelSetShape shape, const std::int64_t* center,
                         double radius, std::int8_t* output, int workers);
}

