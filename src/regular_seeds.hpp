#pragma once
#include <cstddef>
#include <cstdint>

namespace skimagexx::detail {
enum class RegularSeedsType { boolean, u8, i8, u16, i16, u32, i32, u64, i64, f32, f64 };
struct RegularSeedsAxis {
    std::size_t start;
    std::size_t step;
    std::size_t count;
    std::size_t output_stride; // ELEMENTS, not bytes
};
struct RegularSeedsGrid {
    const RegularSeedsAxis* axes;
    std::size_t ndim;
    std::size_t selected_count;
    std::size_t output_count;
};
// Boundary establishes stable normalized positive-step axes, addressable native
// contiguous output, disjoint descriptor storage, selected_count = product of
// axis counts (rank zero means one), and every selected offset in output.
// Labels fit int64; f32 labels <= 2^24 and f64 labels <= 2^53. Integer labels
// wrap modulo destination width; signed representations are two's complement.
// workers is the caller's once-captured global policy, >= 1. No FP arithmetic
// except exact integer-to-float conversion. Unique destinations permit parallel
// assignment. No validation, allocation, global worker mutation, or callbacks.
// assign preserves all unselected bytes; Python passes its already-zeroed array.
void regular_seeds_assign(const RegularSeedsGrid&, void* output, RegularSeedsType, int workers);
// Standalone C++ convenience: zero the entire output, then assign once.
void regular_seeds_into(const RegularSeedsGrid&, void* output, RegularSeedsType, int workers);
}

namespace skimagexx::simd {
void regular_seeds_zero(void* output, std::size_t bytes);
void regular_seeds_u32_run(std::uint32_t* output, std::size_t count, std::uint64_t first);
void regular_seeds_u64_run(std::uint64_t* output, std::size_t count, std::uint64_t first);
}
