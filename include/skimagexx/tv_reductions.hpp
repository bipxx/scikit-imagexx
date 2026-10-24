#pragma once

#include <cstddef>
#include <cstdint>

namespace skimagexx {

// Private numerical stages, not a public denoising capability. The two profiles
// are audited against exact NumPy releases, not inferred from a version range.
inline constexpr const char* tv_reductions_interface = "tv-reductions-v1";
enum class TvReductionType { float32, float64 };
enum class TvReductionProfile { numpy_1244_buffered, numpy_254_aligned_contiguous };
struct TvReductionInput {
    const void* data;
    std::size_t count;
    TvReductionType type;
};
struct TvReductionPlan {
    TvReductionProfile profile;
    std::size_t buffer_elements; // Sampled by caller at this SUM provider stage.
};
enum TvReductionFpStatus : unsigned {
    tv_fp_invalid = 1, tv_fp_divide = 2, tv_fp_overflow = 4, tv_fp_underflow = 8
};
struct TvReductionResult {
    std::uint64_t bits; // float32 uses low 32 bits, upper bits zero.
    unsigned fp_status;
};

// Native-endian, naturally aligned, contiguous F32/F64 only. The caller owns
// storage and ensures its lifetime. Empty input may be null. No allocations,
// conversions, warnings, callbacks, scalar parameter evaluation or worker pool.
// Requires nearest rounding and gradual underflow (no FTZ/DAZ). Operation flags
// are returned while the caller's original floating environment is restored.
// The literal typed association is invariant under global worker settings.
TvReductionResult tv_sum_contiguous(TvReductionInput input, TvReductionPlan plan);

// Separate stage: the caller must dispatch square-stage FP policy BEFORE SUM.
// Output has input.count/type, natural alignment, and must be disjoint from the
// input. All descriptor validation is completed before any output store.
unsigned tv_square_contiguous_into(TvReductionInput input, void* output);

} // namespace skimagexx
