#pragma once
#include <cstddef>
namespace skimagexx {
enum class CornerResponse {harris,noble,shi_tomasi,foerstner};
// Three contiguous aligned floating tensor fields (rr, rc, cc). The first
// output is required; Foerstner also requires a second output. Buffers must
// not overlap. Input float32 intermediates retain their precision, even when
// a typed Harris/Noble scalar promotes the final output to float64 (wide).
// False means unsafe arithmetic: discard outputs and use compatibility.
bool corner_response_into(const void* const* fields,void* output,void* second,
    std::size_t count,bool single,CornerResponse method,double parameter=0,
    bool wide=false);
}
