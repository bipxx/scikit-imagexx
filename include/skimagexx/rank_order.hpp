#pragma once
#include "convert.hpp"
#include <vector>
namespace skimagexx {
// Native contiguous aligned real input. Labels use an unsigned integer type
// wide enough for count-1 and must not overlap input. Representatives are
// original input positions in ascending unique-value order. Empty input is
// rejected. NaNs and mixed signed zeros return false before writing labels:
// their upstream representatives/ranks depend on NumPy's unstable argsort.
bool rank_order_into(const void* input,std::size_t count,PixelType source,
    void* labels,PixelType label_type,std::vector<std::size_t>& representatives);
}
