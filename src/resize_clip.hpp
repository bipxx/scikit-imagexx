#pragma once
#include "skimagexx/convert.hpp"
namespace skimagexx::detail {
enum class ResizeClipPolicy { legacy, propagating, modern };
// Private in-place resize clipping stage. Image and scalar bounds have the
// same native type; image storage is aligned, contiguous and writable.
// Legacy replaces ties and ignores NaN bounds. Propagating replaces ties and
// propagates NaN bounds while preserving existing NaN pixels. Modern preserves
// ties and fills with the first NaN bound, including existing NaN pixels.
// Bool uses raw unsigned byte ordering, as NumPy clip does even for unusual
// noncanonical image/bound bytes. Ordinary bool bounds still produce 0/1.
void resize_clip_into(void*,std::size_t,PixelType,const void*,const void*,ResizeClipPolicy);
}
#ifdef SKIMAGEXX_AVX2
namespace skimagexx::simd {
std::size_t resize_clip(void*,std::size_t,PixelType,const void*,const void*,bool replace_ties);
std::size_t resize_clip_nan(void*,std::size_t,PixelType,const void*);
}
#endif
