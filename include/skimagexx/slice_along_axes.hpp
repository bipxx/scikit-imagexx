#pragma once
#include "crop.hpp"
namespace skimagexx {
struct SliceAlongAxesPair {std::ptrdiff_t first=0,last=0;};
// Complete operation for ordinary typed metadata and POD storage:
// - empty bounds -> original-input identity; ignores axes/copy entirely;
// - nullptr axes -> default0..ndim-1; explicit vector must match bounds count;
// - duplicate-axis rejection precedes range validation; negative axes reject;
// - negative bounds use modulo axis length BEFORE ordering/bounds checks;
// - Python big integers/protocols are handled in the ordered frontend;
// - basic views retain strides and caller-owned input lifetime;
// - copy requests produce owned C-order POD words through crop_copy_into.
// Nonempty typed operations require nonzero itemsize, as crop.hpp. Zero-word
// NumPy representation is a frontend provider; no POD bytes exist to copy.
// Python identity/base/copy protocols and object references are provider domains.
struct SliceAlongAxesResult {
    enum class Kind { identity, borrowed, owned };
    Kind kind=Kind::identity;
    CropPlan plan;
    std::vector<unsigned char> payload;
    CropView view() const {auto out=plan.view;if(kind==Kind::owned)out.data=payload.empty()?nullptr:payload.data();return out;}
};
SliceAlongAxesResult slice_along_axes(const CropView& image,
    const std::vector<SliceAlongAxesPair>& bounds,
    const std::vector<std::ptrdiff_t>* axes=nullptr,bool copy=false);
}
