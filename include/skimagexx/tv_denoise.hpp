#pragma once

#include "skimagexx/tv_chambolle.hpp"
#include "skimagexx/tv_reductions.hpp"
#include <cstddef>
#include <cstdint>

namespace skimagexx {
inline constexpr const char* tv_denoise_interface = "tv-chambolle-finite-v1";
// Private finite-loop capabilities; never substitute stage63 for loop proof.
inline constexpr unsigned tv_denoise_capabilities = 3; // native F32/F64
struct TvDenoiseParameters {
    double weight;
    double eps;
    std::size_t max_iterations;
    TvReductionPlan reduction;
};
struct TvDenoiseWorkspace {
    TvPlanes p;
    TvPlanes g;
    TvView d;
    TvView out;
    TvView squared;
    TvView norm;
    // Pre-marshalled readonly aliases avoid allocating descriptor vectors in
    // the loop. Guard requires identical addresses/types/shapes/strides.
    TvConstPlanes p_read;
    TvConstPlanes g_read;
    TvConstView out_read;
    TvConstView norm_read;
};
struct TvDenoiseTrace {
    std::uint64_t square_sum_bits;
    std::uint64_t norm_sum_bits;
    std::uint64_t energy_bits;
    std::uint64_t difference_bits;
    std::uint64_t threshold_bits;
    bool energy_is_float32; // only F32 NumPy2.5.4; otherwise F64
    bool stopped;
};
struct TvDenoiseResult {
    std::size_t iterations; // actual update bodies executed, including break
    bool input_alias;      // true iff one body; caller returns input object
};

// Complete prewrite guard; nullptr means eligible, otherwise an immutable
// descriptive reason. No allocations, stores, scalar callbacks or FP arithmetic.
// Input: arbitrary native F32/F64 byte strides, readonly, rank1..4, nonempty,
// count<=2^30, abs<=2^20. Input and all workspaces have disjoint byte extents.
// Workspaces are naturally aligned C spatial/plane arrays of identical type and
// shape; p/g/d are caller-initialized POSITIVE zero. Caller retains ownership.
// weight=[2^-20,2^20], eps=[0,2^20], iterations=[1,20000]; exact two audited
// NumPy profiles and actual positive buffer_elements. Nearest/gradual FP only.
const char* tv_denoise_ineligibility(const TvConstView& input,
    const TvDenoiseWorkspace& workspace, const TvDenoiseParameters& parameters,
    TvDenoiseTrace* trace = nullptr, std::size_t trace_capacity = 0) noexcept;

// Rechecks guard BEFORE ANY WRITE; rejects with invalid_argument(reason).
// Executes divergence/out, square/sum, gradient/norm/sum, dual update and strict
// stopping entirely in C++. F32 energy uses double for1.24.4 and typed weak-scalar
// rounding at every operation for2.5.4. F64 uses double for both. p/g stage worker
// policy is the existing sole global state; reduction association deterministic.
// Optional caller-owned trace capacity must cover max_iterations. No callback,
// channel handling, allocation, warning-handler or arbitrary-argument contract.
// One iteration still performs all energy/dual updates before input_alias=true.
TvDenoiseResult tv_denoise_finite_into(const TvConstView& input,
    const TvDenoiseWorkspace& workspace, const TvDenoiseParameters& parameters,
    TvDenoiseTrace* trace = nullptr, std::size_t trace_capacity = 0);
} // namespace skimagexx
