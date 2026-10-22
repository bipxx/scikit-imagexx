#pragma once
#include "convert.hpp"
#include <vector>
namespace skimagexx {
// Views have one element stride per output axis; zero strides broadcast.
// Storage is native endian, aligned, and must remain immutable during calls.
// Output buffers are aligned, contiguous, and do not overlap either input.
struct HysteresisView {const void* data;PixelType type;std::vector<std::ptrdiff_t> strides;};
void hysteresis_minimum_into(const HysteresisView&,const HysteresisView&,void*,const std::vector<std::size_t>&);
// Same-type comparisons, plus exact signed64/unsigned64 mixed comparisons.
void hysteresis_greater_into(const HysteresisView&,const HysteresisView&,std::uint8_t*,const std::vector<std::size_t>&);
// Face connectivity in arbitrary dimensions. Strong pixels outside the weak
// mask select the entire background, matching upstream label-zero handling.
// Buffers must not overlap. Empty shapes with a zero extent are no-ops.
void hysteresis_connect_into(const std::uint8_t*,const std::uint8_t*,std::uint8_t*,const std::vector<std::size_t>&);
// Convenience entry for contiguous same-type images and scalar/full bounds.
void hysteresis_threshold_into(const void*,const void*,const void*,std::uint8_t*,PixelType,
    const std::vector<std::size_t>&,bool low_scalar=true,bool high_scalar=true);
}
