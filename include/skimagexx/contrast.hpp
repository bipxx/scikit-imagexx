#pragma once
#include "convert.hpp"
#include <array>
#include <vector>

namespace skimagexx {
struct ContrastView {
    const void* data=nullptr;
    PixelType type=PixelType::float64;
    bool half=false; // Binary16 uses uint16 storage.
    bool swap=false;
    std::vector<std::size_t> shape;
    std::vector<std::ptrdiff_t> strides; // Signed bytes; broadcasts allowed.
};
// Nonempty boolean storage; true if all logical values have the same truth.
bool contrast_boolean_constant(const ContrastView&);
// Linear percentile interpolation at two virtual logical ranks in [0,n-1].
// Reads raw values without intensity scaling. Input is unchanged. Sorted-value
// subtraction uses the input precision, including signed integer modular
// differences and binary16 rounding. Returns false for exceptional floating
// arithmetic; output is then unspecified. Parallelism uses the global limit.
bool contrast_percentiles(const ContrastView&,const std::array<double,2>& ranks,
                          std::array<double,2>& output);
}
