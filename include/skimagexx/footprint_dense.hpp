#pragma once
#include <cstddef>

namespace skimagexx {
enum class DenseFootprint { rectangle, diamond, disk, ball };
// Caller-owned, disjoint C-contiguous output; atoms are one complete output
// element, without Python references. Axis costs contain width entries. For
// diamond they are |coordinate-radius|; for disk/ball coordinate squared.
// Output count is width^2 (diamond/disk) or width^3 (ball). Rectangle permits
// any count and no axis. The global worker limit is shared with every kernel.
void footprint_dense_into(void* output, std::size_t count, std::size_t itemsize,
                          const void* zero, const void* one, const double* axis,
                          std::size_t width, DenseFootprint kind, double limit);
namespace simd {
std::size_t footprint_dense_row(void* output, std::size_t itemsize,
                               const void* zero, const void* one,
                               const double* axis, std::size_t width,
                               double middle, double outer, bool ball, double limit);
}
}
