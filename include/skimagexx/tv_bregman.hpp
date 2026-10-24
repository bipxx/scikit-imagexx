#pragma once
#include <cstddef>
#include <cstdint>

namespace skimagexx {
inline constexpr const char* tv_bregman_interface = "tv-bregman-prepared-v1";
inline constexpr unsigned tv_bregman_capabilities = 3; // private typed F32/F64
inline constexpr bool tv_bregman_parallel_sweep = false;
enum class TvBregmanType { float32, float64 };
struct TvBregmanImage {
    const void* data;
    TvBregmanType type;
    std::size_t rows, cols, channels, elements;
};
struct TvBregmanBuffer { void* data; std::size_t elements; };
struct TvBregmanWorkspace {
    TvBregmanBuffer out, dx, dy, bx, by;
};
struct TvBregmanParameters {
    double weight; // caller supplies the already dtype-converted weight value
    int max_iterations; // source Cint; <=0 still performs initialization
    double eps;
    bool isotropic;
};
struct TvBregmanTrace {
    std::uint64_t rmse_bits; // ordered F64 fold of typed tx*tx, sqrt(sum/N)
    bool stopped; // !(rmse>eps), including NaN if arithmetic produces it
};
struct TvBregmanResult {
    std::size_t iterations;
    double rmse; // DBL_MAX when no update body executes
    int initialization_workers; // actual team, never a parallel-sweep claim
};

// Native aligned denseC input (rows,cols,channels), rows/cols>=2, channels>=1.
// Every caller-owned workspace has exact (rows+2)*(cols+2)*channels elements
// of input type. Initial out is caller supplied finite data: zeros for the
// first public call, previous channel_out for subsequent calls. Initial aux
// words are unrestricted and completely overwritten. All six byte extents and optional trace are disjoint. Corners
// of out retain their initial words. Four aux buffers copy CURRENT out before
// resetting its interior/fixed halos: callers must preserve channel_out reuse.
// Input/current out must be finite. Positive typed weight and derived lambda/
// norm must remain finite/nonzero; finite eps>=0. Nearest/gradual FP profile.
// No allocation, shape/provider coercion, channel-axis, return-view ownership,
// warning/error text or unsafe-original-domain compatibility claim.
// One invocation jointly processes all channels in r/c/k order with one RMSE.
// A future public channel loop calls channels=1 sequentially with reused out;
// each invocation resets RMSE to DBL_MAX. No hidden/global channel state.
// Returns nullptr when eligible; validates BEFORE ANY WRITE.
const char* tv_bregman_ineligibility(const TvBregmanImage& input,
    const TvBregmanWorkspace& workspace,const TvBregmanParameters& parameters,
    TvBregmanTrace* trace=nullptr,std::size_t trace_capacity=0) noexcept;

// Rechecks the guard before writes. Snapshot sole image.cpp global workers;
// optional parallel SIMD initialization only. Gauss-Seidel r/c/k updates and
// the residual fold remain serial and preserve source association. F32
// isotropic +1 denominator/division and anisotropic reciprocal use F64.
TvBregmanResult tv_bregman_into(const TvBregmanImage& input,
    const TvBregmanWorkspace& workspace,const TvBregmanParameters& parameters,
    TvBregmanTrace* trace=nullptr,std::size_t trace_capacity=0);
} // namespace skimagexx
