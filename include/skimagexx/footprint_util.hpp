#pragma once
#include <cstddef>
#include <vector>
namespace skimagexx {
// Raw fixed-width records only: the caller retains all reference-bearing
// dtypes in a reference-aware provider. Buffers do not overlap. Output is
// densely allocated, in C or F order, with the same record representation.
struct FootprintPadView {
    const void* input=nullptr;
    void* output=nullptr;
    const void* zero=nullptr;
    std::size_t itemsize=0;
    std::vector<std::size_t> shape,before;
    std::vector<std::ptrdiff_t> input_strides,output_strides;
    bool fortran=false;
};
void footprint_pad_into(const FootprintPadView&);
}
