#pragma once
#include "skimagexx/convert.hpp"
namespace skimagexx::detail {
struct ResizeRangeInfo {
    bool has_value=false,has_nan=false,positive_zero=false,negative_zero=false;
};
// Native aligned contiguous input; same-typed scalar outputs. Ignores NaNs,
// preserving integer precision and reporting NaNs/zero signs separately.
// Empty input leaves bounds untouched. All-NaN input writes quiet NaNs.
// Mixed signed zeros have numerically correct, unspecified tie signs. The
// Python bridge resolves observable tie signs with the runtime NumPy C loop.
ResizeRangeInfo resize_range_into(const void*,std::size_t,PixelType,void*,void*);
}
#ifdef SKIMAGEXX_AVX2
namespace skimagexx::simd {
std::size_t resize_range(const void*,std::size_t,PixelType,void*,void*,detail::ResizeRangeInfo&);
}
#endif
