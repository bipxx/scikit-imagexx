#pragma once
#include <cstddef>
namespace skimagexx::simd {
// Prepared disjoint row: width raw 4-byte words at positive byte stride8.
// Reads stay within (width-1)*8+4 bytes; output holds width*4 bytes.
void montage_stride2_words32(const unsigned char*,unsigned char*,std::size_t);
}
