#pragma once
#include "skimagexx/unique_rows.hpp"

namespace skimagexx::simd {
int unique_rows_compare(const unsigned char* first,const unsigned char* second,
                        std::size_t count);
}
