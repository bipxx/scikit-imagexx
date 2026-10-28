#pragma once
#include "skimagexx/convert.hpp"
#include <cstddef>
#include <vector>

namespace skimagexx::detail {
// Prepared native aligned C float32/64, positive rank1..3, disjoint same-type
// output. Reflect, origin0; even windows use left=window/2. Size1 still runs
// the source recurrence. DP accumulation, division and per-output typed cast.
// False means discard the private output and use the source primitive:
// nonfinite/subnormal/extreme input, unsupported FP environment or arithmetic.
// Caller and worker floating environments are restored. Global worker snapshot.
bool uniform_filter1d_into(const void* input,void* output,PixelType type,
    const std::vector<std::size_t>& shape,std::size_t axis,std::size_t window,int workers);
}
