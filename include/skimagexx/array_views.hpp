#pragma once
#include "crop.hpp"
namespace skimagexx {
// Complete borrowed operations: payload/type ownership stays with the caller.
// Input must describe live storage; signed/zero strides and readonly ownership
// are retained. No allocation or element reads. Positive integral parameters.
// Block and window dimensions append after indexing dimensions, as in NumPy.
CropView view_as_blocks(const CropView&, const std::vector<std::ptrdiff_t>& block);
CropView view_as_windows(const CropView&, const std::vector<std::ptrdiff_t>& window,
                         const std::vector<std::ptrdiff_t>& step);
}
