#pragma once
#include "montage.hpp"
namespace skimagexx::detail {
// Original generic fill/copy implementation, isolated from runtime dispatch.
void montage_generic_into(const MontageInput&,MontageGrid,const void*,MontageOutput,int);
}
namespace skimagexx::simd {
// Disjoint prepared row: width4-byte words at positive byte stride8.
void montage_stride2_words32(const unsigned char*,unsigned char*,std::size_t);
}
