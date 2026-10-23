#pragma once
#include <cstddef>
namespace skimagexx {
// Upper-triangle homogeneous F32/F64 Hessian fields, matrix dimension 1..5.
// Project eigenvalues through a circulant matrix (diagonal1, offdiagonal alpha),
// select the first largest absolute response, clip nonnegative and normalize
// this scale by its global maximum before taking the previous pixel maximum.
// Dense/aligned separate buffers; previous may be null (zero). Output precision
// matches fields. False means unsafe/nonfinite arithmetic; discard output.
// Optional global worker cap defaults to one; reduction order cannot affect max.
bool meijering_accumulate_into(const void* const*,const void*,void*,std::size_t,
                              std::size_t,bool single,double alpha);
}
