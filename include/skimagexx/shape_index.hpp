#pragma once
#include <cstddef>
namespace skimagexx {
// Three contiguous Hessian fields (rr,rc,cc); separate float32/64 output.
// Flat regions produce NaN. Global workers are optional and default to one.
// False requests whole-function fallback for unsafe intermediate arithmetic.
bool shape_index_from_hessian_into(const void* const*,void*,std::size_t,bool single);
}
