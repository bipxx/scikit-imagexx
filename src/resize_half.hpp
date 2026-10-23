#pragma once
#include "resize_range.hpp"
#include <cstdint>

namespace skimagexx::detail {
constexpr bool half_nan(std::uint16_t value){return (value&0x7fff)>0x7c00;}
constexpr bool half_zero(std::uint16_t value){return (value&0x7fff)==0;}
constexpr std::uint16_t half_key(std::uint16_t value){return value&0x8000 ? std::uint16_t(~value) : std::uint16_t(value^0x8000);}
constexpr std::uint16_t half_from_key(std::uint16_t key){return key&0x8000 ? std::uint16_t(key^0x8000) : std::uint16_t(~key);}
constexpr bool half_less(std::uint16_t a,std::uint16_t b){return !(half_zero(a)&&half_zero(b)) && half_key(a)<half_key(b);}
// Private binary16 identity finish. Storage contains native raw words, never
// host float arithmetic, so signed zeros and NaN payloads remain unchanged.
// Copy accepts unaligned contiguous source bytes; all buffers are disjoint.
void half_copy_into(const void*,std::uint16_t*,std::size_t,bool swap);
ResizeRangeInfo half_range_into(const std::uint16_t*,std::size_t,std::uint16_t*,std::uint16_t*);
// Both policies preserve zero ties and existing NaN pixels. Policy 0 ignores
// NaN bounds; policy 1 replaces non-NaN pixels with the first NaN bound.
void half_clip_into(std::uint16_t*,std::size_t,std::uint16_t,std::uint16_t,int policy);
}
#ifdef SKIMAGEXX_AVX2
namespace skimagexx::simd {
std::size_t half_copy(const void*,std::uint16_t*,std::size_t,bool);
std::size_t half_range(const std::uint16_t*,std::size_t,std::uint16_t*,std::uint16_t*,detail::ResizeRangeInfo&);
std::size_t half_clip(std::uint16_t*,std::size_t,std::uint16_t,std::uint16_t,int);
}
#endif
