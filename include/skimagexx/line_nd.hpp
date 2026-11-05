#pragma once
#include <array>
#include <cstddef>
#include <cstdint>

namespace skimagexx::detail {
enum class LineNDType { float16, float32, float64, int8, uint8, int64, uint64 };
// Explicit NumPy scalar promotion policy for _round_safe's first % 1.
// Standalone SDK defaults to legacy float64 (NumPy1 semantics). Python binding
// selects from its actual scalar result dtype once at its runtime boundary.
enum class LineNDScalarPromotion { legacy_float64, weak_input };
struct LineNDVector {
    const void* data;
    std::size_t size;
    std::ptrdiff_t stride; // bytes; signed, gapped and unaligned storage allowed
    LineNDType type;
};
struct LineNDFPEvents {
    bool divide_underflow = false;
    bool multiply_underflow = false;
};
struct LineNDPlan {
    std::size_t dimensions = 0, points = 0, rounded_dimensions = 0;
    LineNDType precision = LineNDType::float64;
    LineNDScalarPromotion scalar_promotion = LineNDScalarPromotion::legacy_float64;
    bool endpoint = false, integer = true, zero_step = false;
    std::array<double, 8> start{}, stop{}, delta{}, step{};
    LineNDFPEvents events{};
};
// Typed boundary: equal input dtypes; vectors of length1 broadcast, dimensions
// 1..8. Caller proves complete native-endian byte spans and ordinary nearest
// rounding. Read operands once, retaining original-dtype subtraction/abs for
// count before linspace's floating conversion. Refuse unsupported/nonfinite,
// unsafe extent/integer-cast domains without changing plan. Scalar form and
// NumPy conversion/protocol/error/storage semantics remain the Python boundary.
bool line_nd_prepare(LineNDVector start, LineNDVector stop, bool endpoint,
                     bool integer, LineNDPlan& plan,
                     LineNDScalarPromotion promotion = LineNDScalarPromotion::legacy_float64);
// Caller owns a complete nonoverlapping output buffer. Steps are BYTES; usual
// point-major backing uses point_step=D*itemsize, dimension_step=itemsize.
// Integer outputs are intptr_t; floating outputs match plan.precision (half is
// uint16 storage). All coordinates are emitted from prepared snapshots.
// Worker limit is already snapshotted once by caller (global default1). Small
// outputs remain serial. Independent coordinates permit optional thresholded
// parallel/SIMD work; no private pools/global worker lookup here. Floating
// status is reported as stage facts for caller's NumPy FP callback policy;
// this entry never calls Python and restores the calling floating environment.
LineNDFPEvents line_nd_into(const LineNDPlan&, void* output,
                          std::ptrdiff_t point_step, std::ptrdiff_t dimension_step,
                          int workers);
}
