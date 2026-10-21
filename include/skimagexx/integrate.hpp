#pragma once
#include "convert.hpp"
#include <vector>
namespace skimagexx {
struct IntegralTableView {
    const void* data=nullptr;
    PixelType type=PixelType::float64;
    bool half=false; // uint16 storage interpreted as IEEE binary16
    bool swap=false;
    std::vector<std::size_t> shape;
    std::vector<std::ptrdiff_t> strides; // bytes, including zero/negative strides
};
// Lower corners contain upstream's normalized start-1. A negative selected
// lower coordinate suppresses that entire corner. Upper coordinates are valid
// nonnegative indices. Metadata is contiguous [windows, rank]; the corner
// count 2**rank must fit size_t. Runtime grows exponentially with rank.
// Output is separate contiguous double storage. Returns false for nonfinite
// sampled values or overflowing arithmetic; output is then unspecified.
bool integrate_windows_into(const IntegralTableView&,const std::int64_t*,const std::int64_t*,double*,std::size_t windows);
}
