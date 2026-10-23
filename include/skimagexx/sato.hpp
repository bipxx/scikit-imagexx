#pragma once
#include "convert.hpp"
#include <cstddef>
namespace skimagexx {
// Raw floating cast (no integer normalization), followed by optional negation.
// Native dense/aligned buffers, separate storage. Single output requires F32 input.
void ridge_prepare_into(const void*,PixelType,void*,std::size_t,bool single,bool negative);
// Upper-triangle Hessian fields, homogeneous F32/F64. Uses the largest n-1
// eigenvalues, nonnegative clipping, geometric mean, scale, then pixel maximum.
// n=2/3, decreasing eigenvalue order. Values retain field precision through
// product/root; value_single selects scale precision. Output/previous may differ
// in precision; null previous means zeros. All buffers dense/aligned and separate.
// False means unsafe arithmetic/nonconvergence; discard output. Global cap default1.
bool sato_accumulate_into(const void* const*,const void*,void*,std::size_t,std::size_t,
                         bool field_single,bool previous_single,bool output_single,
                         bool value_single,double factor);
}
