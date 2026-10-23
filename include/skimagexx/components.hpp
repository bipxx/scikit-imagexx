#pragma once
#include "convert.hpp"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace skimagexx {
inline constexpr unsigned components_interface_version=1;
struct ComponentsView {
    const void* data=nullptr;
    PixelType type=PixelType::uint8;
    bool half=false,swap=false;
    std::vector<std::size_t> shape;
    std::vector<std::ptrdiff_t> strides;
};
struct ComponentsConversionInfo {bool invalid_cast=false;};
// V1 preparation is separate from component labeling so Python can reproduce
// NumPy cast warnings/errors before rank/background/connectivity validation.
// Outputs are native, aligned, contiguous C-order and do not alias input.
ComponentsConversionInfo components_prepare_index_into(const ComponentsView&,std::intptr_t*);
void components_prepare_bool_into(const ComponentsView&,std::uint8_t*);
// Prepared bool keys are canonical 0/1; background1 labels original False
// components. Other numeric input has already been converted to intptr.
// Output storage is aligned C-order, disjoint from prepared input. The final
// IDs follow first logical C-order occurrence regardless of worker count.
std::size_t components_label_index_into(const std::intptr_t*,std::intptr_t*,
    const std::vector<std::size_t>&,std::intptr_t background,int connectivity);
std::size_t components_label_bool_into(const std::uint8_t*,void*,bool wide_output,
    const std::vector<std::size_t>&,std::uint8_t background,int connectivity);
}
#ifdef SKIMAGEXX_AVX2
namespace skimagexx::simd {
std::size_t components_prepare_index(const void*,std::intptr_t*,std::size_t,PixelType,bool swap);
std::size_t components_prepare_bool(const std::uint8_t*,std::uint8_t*,std::size_t);
std::size_t components_equal_prefix(const void*,std::size_t,const void*,std::size_t bytes);
std::size_t components_fill(void*,std::size_t,const void*,std::size_t bytes);
}
#endif
