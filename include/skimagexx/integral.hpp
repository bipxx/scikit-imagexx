#pragma once
#include "convert.hpp"
#include <cstddef>
#include <cstdint>
#include <vector>
namespace skimagexx {
// Ordered, in-place axis-prefix stages after a raw input cast. Integer outputs
// use modulo-2^64 accumulation, including signed output bit patterns. Floating
// outputs preserve each addition's precision and the first element's sign.
// Dense/aligned, separate storage; target: i64/u64/f32/f64. Integer targets
// require integer inputs; f32 currently requires f32 input. Empty shapes allowed.
// Optional strides are positive dense element strides in logical axis order;
// input and output must have the same physical axis order. Empty means C order.
void integral_image_into(const void*,void*,PixelType,PixelType,const std::vector<std::size_t>&,
                         const std::vector<std::size_t>& strides={});
// Raw signed/unsigned 64-bit line, with an arbitrary byte stride (including
// negative/zero and unaligned storage). Swap reverses each input word's bytes.
// Output is contiguous and aligned; modulo-2^64 sums preserve signed bit patterns.
// Input/output must be separate except that exact dense in-place storage is safe.
// Optional global workers compute chunk prefixes and then apply exact offsets.
void integral_line_u64_into(const void*,std::uint64_t*,std::size_t,std::ptrdiff_t,bool swap=false);
}
